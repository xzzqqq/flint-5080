use crate::{
    config::Config,
    engine_ffi::{Engine, EngineStatus, Telemetry},
    generation::{
        FinishReason, GenerationError, GenerationEvent, GenerationRequest, GenerationStream,
        before_deadline, collect_generation, start_generation,
    },
    parser::{ParseDelta, ParsedToolCall},
    template::ChatTemplate,
    tokenizer::Tokenizer,
};
use axum::{
    Json, Router,
    extract::{FromRequest, Request, State},
    http::StatusCode,
    middleware::{self, Next},
    response::{IntoResponse, Response, Sse, sse::Event},
    routing::{get, post},
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    convert::Infallible,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::time::Instant;
use tokio_stream::StreamExt;

pub struct AppState {
    pub config: Config,
    pub engine: Arc<Engine>,
    pub tokenizer: Arc<Tokenizer>,
    pub template: Arc<ChatTemplate>,
}

pub fn build_router(state: Arc<AppState>) -> Router {
    let v1 = Router::new()
        .route("/models", get(models))
        .route("/chat/completions", post(chat))
        .route_layer(middleware::from_fn_with_state(state.clone(), authorize));
    Router::new()
        .nest("/v1", v1)
        .route("/health", get(|| async { Json(json!({"status":"ok"})) }))
        .route("/ready", get(ready))
        .with_state(state)
}

async fn authorize(State(s): State<Arc<AppState>>, req: Request, next: Next) -> Response {
    if let Some(key) = &s.config.api_key {
        let bearer = req
            .headers()
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.split_once(' '));
        if !bearer
            .is_some_and(|(scheme, token)| scheme.eq_ignore_ascii_case("bearer") && token == key)
        {
            let error = json!({
                "message": "Invalid API key",
                "type": "authentication_error",
                "code": "invalid_api_key"
            });
            return (StatusCode::UNAUTHORIZED, Json(json!({"error": error}))).into_response();
        }
    }
    next.run(req).await
}

async fn ready(State(s): State<Arc<AppState>>) -> Response {
    match s
        .engine
        .stats_json()
        .ok()
        .and_then(|s| serde_json::from_str::<Value>(&s).ok())
    {
        Some(engine) => Json(json!({"status":"ready","model":s.config.model_name,"engine":engine}))
            .into_response(),
        None => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"status":"unavailable"})),
        )
            .into_response(),
    }
}
async fn models(State(s): State<Arc<AppState>>) -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": [{"id": s.config.model_name, "object": "model", "created": 0, "owned_by": "flint-5080"}]
    }))
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Stop {
    One(String),
    Many(Vec<String>),
}
#[derive(Default, Deserialize)]
struct StreamOptions {
    #[serde(default)]
    include_usage: bool,
}

#[derive(Deserialize)]
struct ChatRequest {
    model: Option<String>,
    messages: Vec<Value>,
    tools: Option<Vec<Value>>,
    tool_choice: Option<Value>,
    reasoning_effort: Option<Value>,
    #[serde(default)]
    stream: bool,
    stream_options: Option<StreamOptions>,
    max_completion_tokens: Option<u32>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<i32>,
    seed: Option<u64>,
    stop: Option<Stop>,
    n: Option<u32>,
}
impl ChatRequest {
    fn generation(self, cfg: &Config) -> Result<GenerationRequest, GenerationError> {
        if self.model.as_ref().is_some_and(|m| m.trim().is_empty())
            || self.n.is_some_and(|n| n != 1)
        {
            return Err(GenerationError::Validation(
                "model must be non-empty and n must equal 1".into(),
            ));
        }
        if let Some(choice) = &self.tool_choice
            && !(matches!(choice.as_str(), Some("none" | "required" | "auto"))
                || choice["type"] == "function"
                    && choice["function"]["name"]
                        .as_str()
                        .is_some_and(|n| !n.is_empty()))
        {
            return Err(GenerationError::Validation("invalid tool_choice".into()));
        }
        Ok(GenerationRequest {
            messages: self.messages,
            tools: self.tools,
            tool_choice: self.tool_choice,
            reasoning_effort: self.reasoning_effort,
            max_tokens: self
                .max_completion_tokens
                .or(self.max_tokens)
                .unwrap_or(cfg.default_max_tokens),
            temperature: self.temperature.unwrap_or(cfg.temperature),
            top_p: self.top_p.unwrap_or(1.0),
            top_k: self.top_k.unwrap_or(0),
            seed: self.seed.unwrap_or(0),
            stop_strings: match self.stop {
                Some(Stop::One(s)) => vec![s],
                Some(Stop::Many(v)) => v,
                None => vec![],
            },
        })
    }
}

fn error_body(error: &GenerationError) -> (StatusCode, Value) {
    let (status, kind, code) = match error {
        GenerationError::Internal(_) => (StatusCode::INTERNAL_SERVER_ERROR, "server_error", None),
        GenerationError::Timeout => (StatusCode::GATEWAY_TIMEOUT, "timeout_error", None),
        GenerationError::Engine(e) if e.status == EngineStatus::ContextOverflow => (
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            Some("context_length_exceeded"),
        ),
        GenerationError::Engine(e) if e.status != EngineStatus::InvalidInput => {
            (StatusCode::INTERNAL_SERVER_ERROR, "server_error", None)
        }
        _ => (StatusCode::BAD_REQUEST, "invalid_request_error", None),
    };
    (
        status,
        json!({"error":{"type":kind,"code":code,"message":error.to_string()}}),
    )
}

struct Completion {
    id: String,
    created: u64,
    model: String,
    streaming: bool,
    telemetry: bool,
    include_usage: bool,
    logged: bool,
}
impl Completion {
    fn frame(&self, choices: Value) -> Value {
        let object = if self.streaming {
            "chat.completion.chunk"
        } else {
            "chat.completion"
        };
        json!({"id":self.id,"object":object,"model":self.model,"created":self.created,"choices":choices})
    }
    fn usage(&self, body: &mut Value, t: Telemetry) {
        body["usage"] = json!({
            "prompt_tokens": t.prefill_tokens,
            "completion_tokens": t.decode_tokens,
            "total_tokens": u64::from(t.prefill_tokens) + u64::from(t.decode_tokens),
            "prompt_tokens_details": {"cached_tokens": t.cached_prefix_tokens}
        });
        if self.telemetry {
            body["x_flint_telemetry"] = json!(t);
        }
    }
    fn log(&mut self, status: u16, reason: Option<FinishReason>, t: Telemetry) {
        self.logged = true;
        tracing::info!(
            req_id = %self.id, model = %self.model, stream = self.streaming,
            prompt_tokens = t.prefill_tokens, cached_tokens = t.cached_prefix_tokens,
            decode_tokens = t.decode_tokens, prefill_ms = t.prefill_ms, decode_ms = t.decode_ms,
            finish_reason = ?reason, status, "chat completion"
        );
    }
    fn error(&mut self, e: GenerationError) -> (StatusCode, Value) {
        let (status, body) = error_body(&e);
        self.log(
            status.as_u16(),
            None,
            match e {
                GenerationError::Engine(e) => e.telemetry,
                _ => Telemetry::default(),
            },
        );
        (status, body)
    }
    fn chunk(&self, delta: Value, reason: Option<FinishReason>) -> Value {
        self.frame(json!([{"index":0,"delta":delta,"finish_reason":reason}]))
    }
    fn sse(mut self, mut stream: GenerationStream) -> Response {
        let events = async_stream::stream! {
            yield Ok::<_, Infallible>(sse_event(self.chunk(json!({"role": "assistant"}), None)));
            while let Some(event) = stream.next().await {
                match event {
                    Ok(GenerationEvent::Delta(delta)) => {
                        let delta = match delta {
                            ParseDelta::Content(s) => json!({"content": s}),
                            ParseDelta::Reasoning(s) => json!({"reasoning_content": s}),
                            ParseDelta::ToolCall(c) => json!({"tool_calls": [tool_call(c, true)]}),
                        };
                        yield Ok(sse_event(self.chunk(delta, None)));
                    }
                    Ok(GenerationEvent::Finished { finish_reason, telemetry }) => {
                        self.log(200, Some(finish_reason), telemetry);
                        yield Ok(sse_event(self.chunk(json!({}), Some(finish_reason))));
                        if self.telemetry || self.include_usage {
                            let mut body = self.frame(json!([]));
                            self.usage(&mut body, telemetry);
                            yield Ok(sse_event(body));
                        }
                        break;
                    }
                    Err(e) => {
                        let (_, body) = self.error(e);
                        drop(stream);
                        yield Ok(sse_event(body));
                        break;
                    }
                }
            }
            yield Ok(Event::default().data("[DONE]"));
        };
        Sse::new(events).into_response()
    }
}
impl Drop for Completion {
    fn drop(&mut self) {
        if !self.logged {
            self.log(499, None, Telemetry::default());
        }
    }
}

fn sse_event(body: Value) -> Event {
    Event::default().data(body.to_string())
}
fn tool_call(c: ParsedToolCall, indexed: bool) -> Value {
    let mut call =
        json!({"id":c.id,"type":"function","function":{"name":c.name,"arguments":c.arguments}});
    if indexed {
        call["index"] = json!(c.index);
    }
    call
}

async fn chat(State(s): State<Arc<AppState>>, request: Request) -> Response {
    static SEQUENCE: AtomicU64 = AtomicU64::new(1);
    let deadline = s
        .config
        .request_timeout
        .and_then(|d| Instant::now().checked_add(d));
    let telemetry = request
        .headers()
        .get("x-flint-telemetry")
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let mut completion = Completion {
        id: format!(
            "chatcmpl-{created}-{}",
            SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ),
        created,
        model: s.config.model_name.clone(),
        streaming: false,
        telemetry,
        include_usage: false,
        logged: false,
    };
    let generate = async {
        let extract = Json::<ChatRequest>::from_request(request, &s);
        let body = before_deadline(deadline, extract)
            .await?
            .map_err(|e| GenerationError::Validation(e.body_text()))?
            .0;
        completion.streaming = body.stream;
        completion.include_usage = body
            .stream_options
            .as_ref()
            .is_some_and(|o| o.include_usage);
        let req = body.generation(&s.config)?;
        let (_, stream) = start_generation(
            s.engine.clone(),
            s.tokenizer.clone(),
            s.template.clone(),
            req,
            deadline,
        )
        .await?;
        Ok::<_, GenerationError>(stream)
    };
    let result = match generate.await {
        Ok(stream) if completion.streaming => return completion.sse(stream),
        Ok(stream) => collect_generation(stream).await,
        Err(e) => Err(e),
    };
    match result {
        Ok((msg, reason, t)) => {
            let mut message = json!({"role":"assistant","content":msg.content});
            if let Some(text) = msg.reasoning_content {
                message["reasoning_content"] = json!(text);
            }
            if !msg.tool_calls.is_empty() {
                message["tool_calls"] = json!(
                    msg.tool_calls
                        .into_iter()
                        .map(|c| tool_call(c, false))
                        .collect::<Vec<_>>()
                );
            }
            let mut body =
                completion.frame(json!([{"index":0,"message":message,"finish_reason":reason}]));
            completion.usage(&mut body, t);
            completion.log(200, Some(reason), t);
            Json(body).into_response()
        }
        Err(e) => {
            let (status, body) = completion.error(e);
            (status, Json(body)).into_response()
        }
    }
}

#[cfg(all(test, not(feature = "link-engine")))]
mod tests;
