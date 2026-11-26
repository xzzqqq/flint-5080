use crate::{
    engine_ffi::{
        Engine, EngineError, EngineStatus, EngineStopReason, Event, EventReceiver, Request,
        Telemetry,
    },
    parser::{ParseDelta, ParsedMessage, ResponseParser},
    template::{ChatTemplate, TemplateError},
    tokenizer::Tokenizer,
};
use futures_core::Stream;
use serde::Serialize;
use serde_json::Value;
use std::{
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};
use tokio::time::{Instant, timeout_at};
use tokio_stream::StreamExt;

pub struct GenerationRequest {
    pub call_id_prefix: String,
    pub messages: Vec<Value>,
    pub tools: Option<Vec<Value>>,
    pub tool_choice: Option<Value>,
    pub reasoning_effort: Option<Value>,
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub seed: u64,
    pub stop_strings: Vec<String>,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
}

pub enum GenerationEvent {
    Delta(ParseDelta),
    Finished {
        finish_reason: FinishReason,
        telemetry: Telemetry,
    },
}

#[derive(Debug, thiserror::Error)]
pub enum GenerationError {
    #[error("{0}")]
    Template(#[from] TemplateError),
    #[error("{0}")]
    Validation(String),
    #[error("{0}")]
    Engine(#[from] EngineError),
    #[error("{0}")]
    Internal(String),
    #[error("request timed out")]
    Timeout,
}

pub type GenerationStream =
    Pin<Box<dyn Stream<Item = Result<GenerationEvent, GenerationError>> + Send>>;

// The deadline must cancel inference even when TCP backpressure stops polling
// the body. The guard and timer share a single claim on the cancellation.
struct CancelGuard {
    engine: Arc<Engine>,
    id: u64,
    active: Arc<AtomicBool>,
    timer: Option<tokio::task::JoinHandle<()>>,
}
impl CancelGuard {
    fn new(engine: Arc<Engine>, id: u64, deadline: Option<Instant>) -> Self {
        let active = Arc::new(AtomicBool::new(true));
        let timer = deadline.map(|at| {
            let active = active.clone();
            let engine = engine.clone();
            tokio::spawn(async move {
                tokio::time::sleep_until(at).await;
                if active.swap(false, Ordering::Relaxed) {
                    engine.cancel(id);
                }
            })
        });
        Self {
            engine,
            id,
            active,
            timer,
        }
    }
    fn disarm(&mut self) -> bool {
        let active = self.active.swap(false, Ordering::Relaxed);
        if let Some(timer) = self.timer.take() {
            timer.abort();
        }
        active
    }
    fn cancel(&mut self) {
        if self.disarm() {
            self.engine.cancel(self.id);
        }
    }
}
impl Drop for CancelGuard {
    fn drop(&mut self) {
        self.cancel();
    }
}

// Only retain a suffix that can still become a stop string. All split points
// are UTF-8 boundaries and the retained length is strictly below max_stop_len.
struct StopFilter {
    stops: Vec<String>,
    pending: String,
    matched: bool,
}
impl StopFilter {
    fn push(&mut self, delta: ParseDelta) -> Option<ParseDelta> {
        let ParseDelta::Content(text) = delta else {
            return Some(delta);
        };
        if self.stops.is_empty() {
            return Some(ParseDelta::Content(text));
        }
        self.pending.push_str(&text);
        let end = if let Some(pos) = self.stops.iter().filter_map(|s| self.pending.find(s)).min() {
            self.matched = true;
            self.pending.truncate(pos);
            pos
        } else {
            let max = self.stops.iter().map(String::len).max().unwrap();
            let start = self.pending.len().saturating_sub(max - 1);
            let start = self.pending.ceil_char_boundary(start);
            self.pending[start..]
                .char_indices()
                .map(|(i, _)| start + i)
                .find(|&i| self.stops.iter().any(|s| s.starts_with(&self.pending[i..])))
                .unwrap_or(self.pending.len())
        };
        (end > 0).then(|| ParseDelta::Content(self.pending.drain(..end).collect()))
    }
}

pub(crate) async fn before_deadline<T>(
    deadline: Option<Instant>,
    future: impl Future<Output = T>,
) -> Result<T, GenerationError> {
    let Some(at) = deadline else {
        return Ok(future.await);
    };
    let value = timeout_at(at, future)
        .await
        .map_err(|_| GenerationError::Timeout)?;
    // A ready channel can win the timer race, including a cancellation callback
    // from the independent deadline task.
    if Instant::now() >= at {
        return Err(GenerationError::Timeout);
    }
    Ok(value)
}

async fn receive(
    rx: &mut EventReceiver,
    deadline: Option<Instant>,
    guard: &mut CancelGuard,
) -> Result<Event, GenerationError> {
    let item = before_deadline(deadline, rx.recv()).await?;
    if matches!(item, Some(Ok(Event::Done(..)) | Err(_))) {
        guard.disarm();
    }
    item.unwrap_or_else(|| {
        Err(EngineError {
            status: EngineStatus::Internal,
            message: "missing terminal event".into(),
            telemetry: Telemetry::default(),
        })
    })
    .map_err(Into::into)
}

async fn drain_cancelled(
    rx: &mut EventReceiver,
    mut latest: Telemetry,
    deadline: Option<Instant>,
) -> Result<Telemetry, GenerationError> {
    // Cancellation may race with token batches already in flight. The terminal
    // callback owns the final counters; use the latest snapshot if the channel closes.
    while let Some(event) = before_deadline(deadline, rx.recv()).await? {
        match event {
            Ok(Event::Tokens(_, t)) => latest = t,
            Ok(Event::Done(t, _)) => return Ok(t),
            Err(e) => return Ok(e.telemetry),
        }
    }
    Ok(latest)
}

pub async fn start_generation(
    engine: Arc<Engine>,
    tokenizer: Arc<Tokenizer>,
    template: Arc<ChatTemplate>,
    req: GenerationRequest,
    deadline: Option<Instant>,
) -> Result<(u64, GenerationStream), GenerationError> {
    if req.messages.is_empty() || req.messages.iter().any(|m| m["role"].as_str().is_none()) {
        return Err(GenerationError::Validation(
            "messages must contain chat messages with roles".into(),
        ));
    }
    if req.stop_strings.len() > 4 || req.stop_strings.iter().any(String::is_empty) {
        return Err(GenerationError::Validation(
            "stop accepts up to four non-empty strings".into(),
        ));
    }
    if !req.temperature.is_finite()
        || req.temperature < 0.0
        || !(0.0..=1.0).contains(&req.top_p)
        || req.top_k < 0
    {
        return Err(GenerationError::Validation(
            "invalid sampling parameters".into(),
        ));
    }
    let encoder = tokenizer.clone();
    let encode = tokio::task::spawn_blocking(move || {
        let tools = if req.tool_choice.as_ref().is_some_and(|v| v == "none") {
            None
        } else {
            req.tools.as_deref()
        };
        let prompt = template.render(&req.messages, tools, req.reasoning_effort.as_ref())?;
        encoder
            .encode(&prompt)
            .map_err(|e| GenerationError::Validation(e.to_string()))
    });
    // A timed-out or disconnected caller leaves only CPU work behind. Engine
    // submission stays on this side of the await, under the request's lifetime.
    let ids = before_deadline(deadline, encode)
        .await?
        .map_err(|e| GenerationError::Internal(e.to_string()))??;
    let (id, mut rx) = engine.submit_stream(&Request {
        token_ids: &ids,
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: req.top_k,
        seed: req.seed,
        stop_token_ids: &[],
        ignore_eos: false,
    })?;
    let mut guard = CancelGuard::new(engine, id, deadline);
    // Before committing SSE headers, observe the first worker callback so
    // prompt admission errors and prefill timeouts can set the HTTP status.
    let first = receive(&mut rx, deadline, &mut guard).await?;
    let stream = async_stream::try_stream! {
        let mut event = first;
        let prefix = if req.call_id_prefix.is_empty() {
            "call".into()
        } else {
            req.call_id_prefix
        };
        let mut parser = ResponseParser::with_id_prefix(tokenizer.special.clone(), prefix);
        let mut decoder = tokenizer.stream_decoder();
        let mut filter = StopFilter { stops: req.stop_strings, pending: String::new(), matched: false };
        let mut calls = false;
        let (telemetry, finish_reason) = loop {
            let (deltas, telemetry, done) = match event {
                Event::Tokens(ids, t) => {
                    let mut deltas = Vec::new();
                    for id in ids {
                        deltas.extend(parser.push_token(id, &decoder.push(id)));
                    }
                    (deltas, t, None)
                }
                Event::Done(t, reason) => {
                    let tail = parser.push_token(tokenizer.special.eos_token_id, &decoder.finish());
                    (tail, t, Some(reason))
                }
            };
            for delta in deltas {
                let filtered = filter.push(delta);
                if filter.matched {
                    guard.cancel();
                }
                if let Some(delta) = filtered {
                    calls |= matches!(delta, ParseDelta::ToolCall(_));
                    yield GenerationEvent::Delta(delta);
                }
                if filter.matched { break; }
            }
            if filter.matched {
                let telemetry = if done.is_some() {
                    telemetry
                } else {
                    drain_cancelled(&mut rx, telemetry, deadline).await?
                };
                break (telemetry, FinishReason::Stop);
            }
            if let Some(reason) = done {
                let finish_reason = match reason {
                    EngineStopReason::MaxTokens | EngineStopReason::ContextLimit => FinishReason::Length,
                    _ if calls => FinishReason::ToolCalls,
                    EngineStopReason::None | EngineStopReason::Eos | EngineStopReason::StopToken => FinishReason::Stop,
                };
                break (telemetry, finish_reason);
            }
            event = receive(&mut rx, deadline, &mut guard).await?;
        };
        if !filter.pending.is_empty() && !filter.matched {
            yield GenerationEvent::Delta(ParseDelta::Content(filter.pending));
        }
        yield GenerationEvent::Finished { finish_reason, telemetry };
    };
    Ok((id, Box::pin(stream)))
}

pub async fn collect_generation(
    mut stream: GenerationStream,
) -> Result<(ParsedMessage, FinishReason, Telemetry), GenerationError> {
    let mut message = ParsedMessage::default();
    while let Some(event) = stream.next().await {
        match event? {
            GenerationEvent::Delta(delta) => message.extend([delta]),
            GenerationEvent::Finished {
                finish_reason,
                telemetry,
            } => return Ok((message, finish_reason, telemetry)),
        }
    }
    Err(GenerationError::Internal(
        "generation ended without a finish reason".into(),
    ))
}
