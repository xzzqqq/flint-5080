use super::*;
use crate::{
    config::Options,
    engine_ffi::{
        EngineError, EngineStopReason, Event as EngineEvent, FLINT_ENGINE_ABI_VERSION,
        Request as EngineRequest,
    },
    tokenizer::tests::byte_tokenizer,
};
use axum::{body::Body, http::Request as HttpRequest};
use http_body_util::BodyExt;
use std::{path::PathBuf, sync::Mutex, time::Duration};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    time::{sleep, timeout},
};
use tower::ServiceExt;

fn state(engine: Engine) -> AppState {
    let config = Options {
        model_dir: Some(PathBuf::from("mini")),
        ..Default::default()
    }
    .resolve(None)
    .unwrap();
    AppState {
        metrics: Default::default(),
        engine: Arc::new(engine),
        config,
        tokenizer: Arc::new(byte_tokenizer()),
        template: Arc::new(
            ChatTemplate::from_str("{% if tools %}tools {% endif %}{{ messages[0].content }}")
                .unwrap(),
        ),
    }
}

fn script(text: &str) -> AppState {
    state(script_engine(text, EngineStopReason::Eos))
}

fn script_engine(text: &str, end_reason: EngineStopReason) -> Engine {
    let tokenizer = byte_tokenizer();
    let ids = tokenizer.encode(text).unwrap();
    let eos = tokenizer.special.eos_token_id;
    Engine::with_stub(move |req, tx| {
        let mut t = Telemetry {
            prefill_tokens: req.token_ids.len() as u32,
            cached_prefix_tokens: 2,
            prefill_ms: 1.5,
            decode_ms: 2.25,
            ..Default::default()
        };
        let mut reason = end_reason;
        let mut batch = Vec::new();
        for id in ids.iter().copied().take(req.max_tokens as usize) {
            if id == eos && !req.ignore_eos {
                reason = EngineStopReason::Eos;
                break;
            }
            t.decode_tokens += 1;
            batch.push(id);
            if batch.len() == 8 {
                tx.send(Ok(EngineEvent::Tokens(std::mem::take(&mut batch), t)))
                    .unwrap();
            }
        }
        if !batch.is_empty() {
            tx.send(Ok(EngineEvent::Tokens(batch, t))).unwrap();
        }
        if t.decode_tokens >= req.max_tokens {
            reason = EngineStopReason::MaxTokens;
        }
        tx.send(Ok(EngineEvent::Done(t, reason))).unwrap();
    })
}
fn chat_request(extra: Value, telemetry: Option<&str>) -> HttpRequest<Body> {
    let mut body = json!({"messages":[{"role":"user","content":"hello"}]});
    body.as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    let mut request =
        HttpRequest::post("/v1/chat/completions").header("content-type", "application/json");
    if let Some(v) = telemetry {
        request = request.header("X-Flint-Telemetry", v);
    }
    request.body(Body::from(body.to_string())).unwrap()
}
async fn query(router: Router, req: HttpRequest<Body>) -> (StatusCode, String) {
    let res = router.oneshot(req).await.unwrap();
    let status = res.status();
    let body = res.into_body().collect().await.unwrap().to_bytes();
    (status, String::from_utf8(body.to_vec()).unwrap())
}
fn chunks(body: &str) -> Vec<Value> {
    let mut data = body
        .lines()
        .filter_map(|s| s.strip_prefix("data: "))
        .collect::<Vec<_>>();
    assert_eq!(data.pop(), Some("[DONE]"));
    data.into_iter()
        .map(|s| serde_json::from_str(s).unwrap())
        .collect()
}

#[tokio::test]
async fn chat_formats_and_usage_opt_in() {
    let text = "<|content_thinking|>想<|end_message|><|content_text|>你好!";
    let router = build_router(Arc::new(script(text)));
    for header in [None, Some("false"), Some("1"), Some("TrUe")] {
        let opted = matches!(header, Some("1" | "TrUe"));
        let (status, body) = query(
            router.clone(),
            chat_request(json!({"model":"alias"}), header),
        )
        .await;
        assert_eq!(status, 200);
        let body: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(body["model"], "mini");
        assert_eq!(body["object"], "chat.completion");
        assert_eq!(
            body["choices"][0]["message"],
            json!({"role":"assistant","content":"你好!","reasoning_content":"想"})
        );
        assert_eq!(body["choices"][0]["finish_reason"], "stop");
        assert_eq!(body["usage"]["prompt_tokens_details"]["cached_tokens"], 2);
        assert_eq!(
            body["usage"]["total_tokens"].as_u64(),
            Some(5 + body["usage"]["completion_tokens"].as_u64().unwrap())
        );
        assert_eq!(body.get("x_flint_telemetry").is_some(), opted);
        for usage in [false, true] {
            let res = router
                .clone()
                .oneshot(chat_request(
                    json!({"stream":true,"stream_options":{"include_usage":usage}}),
                    header,
                ))
                .await
                .unwrap();
            assert_eq!(res.headers()["content-type"], "text/event-stream");
            let bytes = res.into_body().collect().await.unwrap().to_bytes();
            let mut frames = chunks(std::str::from_utf8(&bytes).unwrap());
            if opted || usage {
                let trailer = frames.pop().unwrap();
                assert_eq!(trailer["choices"], json!([]));
                assert_eq!(trailer["usage"], body["usage"]);
                assert_eq!(
                    trailer.get("x_flint_telemetry"),
                    body.get("x_flint_telemetry")
                );
            }
            assert!(
                frames.iter().all(|c| c["object"] == "chat.completion.chunk"
                    && c.get("x_flint_telemetry").is_none())
            );
            assert_eq!(
                frames[0]["choices"][0]["delta"],
                json!({"role":"assistant"})
            );
            assert_eq!(frames.pop().unwrap()["choices"][0]["finish_reason"], "stop");
            let join = |key: &str| {
                frames
                    .iter()
                    .filter_map(|c| c["choices"][0]["delta"][key].as_str())
                    .collect::<String>()
            };
            assert_eq!(join("content"), "你好!");
            assert_eq!(join("reasoning_content"), "想");
        }
    }
}

#[tokio::test]
async fn parallel_tools_and_length_priority() {
    let tokens = r#"<|content_invoke_tool_json|>{"name":"lookup","args":{"s":"你"}}<|end_message|><|content_invoke_tool_json|>{"name":"ping","args":[]}<|end_message|>"#;
    let count = byte_tokenizer().encode(tokens).unwrap().len();
    let router = build_router(Arc::new(script(tokens)));
    for limit in [count, count + 1] {
        let expected = if limit == count {
            "length"
        } else {
            "tool_calls"
        };
        for stream in [false, true] {
            let (status, body) = query(
                router.clone(),
                chat_request(
                    json!({"stream":stream,"max_tokens":1,"max_completion_tokens":limit}),
                    None,
                ),
            )
            .await;
            assert_eq!(status, 200);
            let (frame, calls) = if stream {
                let mut frames = chunks(&body);
                let frame = frames.pop().unwrap();
                let calls = frames
                    .into_iter()
                    .filter_map(|c| {
                        c["choices"][0]["delta"]["tool_calls"]
                            .as_array()
                            .map(|a| a[0].clone())
                    })
                    .collect::<Vec<_>>();
                (frame, calls)
            } else {
                let body: Value = serde_json::from_str(&body).unwrap();
                let calls = body["choices"][0]["message"]["tool_calls"]
                    .as_array()
                    .unwrap()
                    .clone();
                (body, calls)
            };
            assert_eq!(frame["choices"][0]["finish_reason"], expected);
            let (_, seq) = frame["id"].as_str().unwrap().rsplit_once('-').unwrap();
            let created = frame["created"].as_u64().unwrap();
            assert_eq!(calls.len(), 2);
            for (i, call) in calls.iter().enumerate() {
                assert_eq!(call["type"], "function");
                assert_eq!(call["id"], format!("call_{created}_{seq}_{i}"));
                assert_eq!(call["index"].as_u64(), stream.then_some(i as u64));
            }
            assert_eq!(
                calls[0]["function"],
                json!({"name":"lookup","arguments":r#"{"s":"你"}"#})
            );
            assert_eq!(
                calls[1]["function"],
                json!({"name":"ping","arguments":"[]"})
            );
        }
    }
}

#[tokio::test]
async fn stop_truncates_across_tokens_and_uses_terminal_or_latest_telemetry() {
    for terminal in [true, false] {
        let app = state(Engine::with_stub(move |_, tx| {
            for batch in "好了🚀ENDsecret".as_bytes().chunks(8) {
                tx.send(Ok(EngineEvent::Tokens(
                    batch.iter().map(|&b| b.into()).collect(),
                    Telemetry {
                        decode_tokens: 12,
                        decode_ms: 3.0,
                        ..Default::default()
                    },
                )))
                .unwrap();
            }
            if terminal {
                tx.send(Err(EngineError {
                    status: EngineStatus::Cancelled,
                    message: "cancelled".into(),
                    telemetry: Telemetry {
                        decode_tokens: 16,
                        decode_ms: 7.0,
                        ..Default::default()
                    },
                }))
                .unwrap();
            }
        }));
        let engine = app.engine.clone();
        let (_, body) = query(
            build_router(Arc::new(app)),
            chat_request(
                json!({"stream":true,"max_tokens":12,"stop":["END","🚀EN"]}),
                Some("1"),
            ),
        )
        .await;
        let mut frames = chunks(&body);
        let telemetry = frames.pop().unwrap()["x_flint_telemetry"].clone();
        assert_eq!(telemetry["decode_ms"], if terminal { 7.0 } else { 3.0 });
        assert_eq!(frames.pop().unwrap()["choices"][0]["finish_reason"], "stop");
        let text = frames
            .iter()
            .filter_map(|c| c["choices"][0]["delta"]["content"].as_str())
            .collect::<String>();
        assert_eq!(text, "好了");
        assert_eq!(engine.cancelled_ids(), vec![1]);
    }
    let (_, body) = query(
        build_router(Arc::new(script("🚀E"))),
        chat_request(json!({"stop":"🚀END"}), None),
    )
    .await;
    assert_eq!(
        serde_json::from_str::<Value>(&body).unwrap()["choices"][0]["message"]["content"],
        "🚀E"
    );
}

#[tokio::test]
async fn forwards_sampling_and_tools_to_template() {
    let prompts = Arc::new(Mutex::new(Vec::new()));
    let captured = prompts.clone();
    let engine = Engine::with_stub(move |r: &EngineRequest, tx| {
        captured.lock().unwrap().push((
            r.token_ids.to_vec(),
            r.temperature,
            r.max_tokens,
            r.top_p,
            r.top_k,
            r.seed,
        ));
        tx.send(Ok(EngineEvent::Done(
            Telemetry::default(),
            EngineStopReason::Eos,
        )))
        .unwrap();
    });
    let mut app = state(engine);
    app.config.temperature = 0.4;
    app.config.default_max_tokens = 9;
    let router = build_router(Arc::new(app));
    for choice in [
        json!("none"),
        json!("auto"),
        json!("required"),
        json!({"type":"function","function":{"name":"test"}}),
    ] {
        let (status, _) = query(
            router.clone(),
            chat_request(
                json!({"tool_choice":choice,"tools":[{}],"seed":8,"top_k":3,"top_p":0.8}),
                None,
            ),
        )
        .await;
        assert_eq!(status, 200);
        let (ids, temp, limit, p, k, seed) = prompts.lock().unwrap().pop().unwrap();
        assert_eq!((temp, limit, p, k, seed), (0.4, 9, 0.8, 3, 8));
        assert_eq!(
            byte_tokenizer().decode(&ids, true).unwrap(),
            if choice == "none" {
                "hello"
            } else {
                "tools hello"
            }
        );
    }
}

#[tokio::test]
async fn auth_probes_and_models() {
    let mut app = script("hello");
    app.config.api_key = Some("test".into());
    let router = build_router(Arc::new(app));
    for (path, status) in [("/health", "ok"), ("/ready", "ready")] {
        let (code, body) = query(
            router.clone(),
            HttpRequest::get(path).body(Body::empty()).unwrap(),
        )
        .await;
        assert_eq!(code, 200);
        let body: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(body["status"], status);
        if path == "/ready" {
            assert_eq!(body["engine"]["abi_version"], FLINT_ENGINE_ABI_VERSION);
            assert_eq!(body["model"], "mini");
        }
    }
    for token in [None, Some("Bearer wrong"), Some("Bearer test")] {
        let mut req = HttpRequest::get("/v1/models");
        if let Some(v) = token {
            req = req.header("authorization", v);
        }
        let (code, body) = query(router.clone(), req.body(Body::empty()).unwrap()).await;
        let body: Value = serde_json::from_str(&body).unwrap();
        if token == Some("Bearer test") {
            assert_eq!(code, 200);
            assert_eq!(body["data"][0]["owned_by"], "flint-5080");
            assert_eq!(body["data"][0]["id"], "mini");
        } else {
            assert_eq!(code, 401);
            assert_eq!(body["error"]["type"], "authentication_error");
        }
    }
    assert_eq!(query(router, chat_request(json!({}), None)).await.0, 401);
}

#[tokio::test]
async fn errors_before_headers() {
    let router = build_router(Arc::new(script("hello")));
    for invalid in [
        json!({"n":2}),
        json!({"model":""}),
        json!({"messages":[]}),
        json!({"messages":null}),
        json!({"top_p":2}),
        json!({"stop":[""]}),
        json!({"stop":["1","2","3","4","5"]}),
        json!({"tool_choice":{"function":{}}}),
        json!({"messages":[{"role":"assistant","tool_calls":[{"function":{"arguments":"bad"}}]}]}),
    ] {
        let (code, body) = query(router.clone(), chat_request(invalid, None)).await;
        assert_eq!(code, 400);
        assert_eq!(
            serde_json::from_str::<Value>(&body).unwrap()["error"]["type"],
            "invalid_request_error"
        );
    }
    for (status, http, expected) in [
        (EngineStatus::ContextOverflow, 400, "invalid_request_error"),
        (EngineStatus::InvalidInput, 400, "invalid_request_error"),
        (EngineStatus::Internal, 500, "server_error"),
    ] {
        let app = state(Engine::with_stub(move |_, tx| {
            tx.send(Err(EngineError {
                status,
                message: "failure".into(),
                telemetry: Telemetry::default(),
            }))
            .unwrap();
        }));
        let router = build_router(Arc::new(app));
        for stream in [true, false] {
            let (code, body) =
                query(router.clone(), chat_request(json!({"stream":stream}), None)).await;
            assert_eq!(code, http);
            let error = serde_json::from_str::<Value>(&body).unwrap()["error"].clone();
            assert_eq!(error["type"], expected);
            if status == EngineStatus::ContextOverflow {
                assert_eq!(error["code"], "context_length_exceeded");
            }
        }
    }
}

fn stalled(first: bool) -> AppState {
    state(Engine::with_stub(move |_, tx| {
        if first {
            tx.send(Ok(EngineEvent::Tokens(vec![65], Telemetry::default())))
                .unwrap();
        }
        tokio::spawn(async move {
            tx.closed().await;
        });
    }))
}
#[tokio::test(start_paused = true)]
async fn timeouts_cancel_before_and_after_headers() {
    for stream in [false, true] {
        for first in [false, true] {
            let mut app = stalled(first);
            app.config.request_timeout = Some(Duration::from_secs(1));
            let engine = app.engine.clone();
            let (code, body) = query(
                build_router(Arc::new(app)),
                chat_request(json!({"stream":stream}), None),
            )
            .await;
            let error = if stream && first {
                assert_eq!(code, 200);
                chunks(&body).pop().unwrap()
            } else {
                assert_eq!(code, 504);
                serde_json::from_str(&body).unwrap()
            };
            assert_eq!(error["error"]["type"], "timeout_error");
            assert_eq!(engine.cancelled_ids(), vec![1]);
        }
    }
}

#[tokio::test]
async fn tcp_disconnect_cancels_while_waiting_for_next_token() {
    let app = stalled(true);
    let engine = app.engine.clone();
    let router = build_router(Arc::new(app));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    let mut socket = TcpStream::connect(addr).await.unwrap();
    let body = r#"{"messages":[{"role":"user","content":"hi"}],"stream":true}"#;
    socket.write_all(format!("POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",body.len(),body).as_bytes()).await.unwrap();
    let mut received = Vec::new();
    timeout(Duration::from_secs(2), async {
        while !String::from_utf8_lossy(&received).contains("assistant") {
            let mut buf = [0; 1024];
            let n = socket.read(&mut buf).await.unwrap();
            assert_ne!(n, 0);
            received.extend_from_slice(&buf[..n]);
        }
    })
    .await
    .unwrap();
    drop(socket);
    timeout(Duration::from_secs(1), async {
        while engine.cancelled_ids().is_empty() {
            sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    assert_eq!(engine.cancelled_ids(), vec![1]);
    server.abort();
}

#[tokio::test(start_paused = true)]
async fn deadline_cancels_when_sse_body_is_not_polled() {
    let mut app = stalled(true);
    app.config.request_timeout = Some(Duration::from_secs(1));
    let engine = app.engine.clone();
    let response = build_router(Arc::new(app))
        .oneshot(chat_request(json!({"stream":true}), None))
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    sleep(Duration::from_secs(2)).await;
    assert_eq!(engine.cancelled_ids(), vec![1]);
    drop(response);
    assert_eq!(engine.cancelled_ids(), vec![1]);
}

#[tokio::test(start_paused = true)]
async fn deadline_includes_reading_request_body() {
    let mut app = script("hello");
    app.config.request_timeout = Some(Duration::from_secs(1));
    let delayed = async_stream::stream! {
        sleep(Duration::from_secs(5)).await;
        yield Ok::<_, Infallible>(axum::body::Bytes::from_static(b"{}"));
    };
    let req = HttpRequest::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from_stream(delayed))
        .unwrap();
    let (code, body) = query(build_router(Arc::new(app)), req).await;
    assert_eq!(code, 504);
    assert_eq!(
        serde_json::from_str::<Value>(&body).unwrap()["error"]["type"],
        "timeout_error"
    );
}

#[tokio::test(start_paused = true)]
async fn stop_waits_for_delayed_cancellation_telemetry() {
    let app = state(Engine::with_stub(|_, tx| {
        tokio::spawn(async move {
            for byte in b"prefix--" {
                sleep(Duration::from_millis(10)).await;
                tx.send(Ok(EngineEvent::Tokens(
                    vec![i32::from(*byte)],
                    Telemetry::default(),
                )))
                .unwrap();
            }
            sleep(Duration::from_secs(2)).await;
            tx.send(Err(EngineError {
                status: EngineStatus::Cancelled,
                message: "cancelled".into(),
                telemetry: Telemetry {
                    decode_tokens: 8,
                    decode_ms: 42.5,
                    ..Default::default()
                },
            }))
            .unwrap();
        });
    }));
    let engine = app.engine.clone();
    let (code, body) = query(
        build_router(Arc::new(app)),
        chat_request(json!({"stop": "--", "max_tokens": 8}), Some("1")),
    )
    .await;
    assert_eq!(code, 200);
    let body: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(body["choices"][0]["message"]["content"], "prefix");
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
    assert_eq!(body["x_flint_telemetry"]["decode_ms"], 42.5);
    assert_eq!(engine.cancelled_ids(), vec![1]);
}

#[tokio::test]
async fn eos_at_token_limit_finishes_with_stop() {
    let router = build_router(Arc::new(script("a<|content_model_end_sampling|>")));
    let (_, body) = query(router, chat_request(json!({"max_tokens": 2}), None)).await;
    let body: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
    assert_eq!(body["choices"][0]["message"]["content"], "a");
    assert_eq!(body["usage"]["completion_tokens"], 1);
}

#[tokio::test]
async fn tool_call_ids_resolve_across_turns() {
    let replies = ["lookup", "ping"].map(|name| {
        let call = format!(
            r#"<|content_invoke_tool_json|>{{"name":"{name}","args":{{}}}}<|end_message|>"#
        );
        byte_tokenizer().encode(&call).unwrap()
    });
    let replies = Mutex::new(replies.into_iter());
    let mut app = state(Engine::with_stub(move |_, tx| {
        let ids = replies.lock().unwrap().next().unwrap();
        tx.send(Ok(EngineEvent::Tokens(ids, Telemetry::default())))
            .unwrap();
        tx.send(Ok(EngineEvent::Done(
            Telemetry::default(),
            EngineStopReason::Eos,
        )))
        .unwrap();
    }));
    let golden: Value =
        serde_json::from_str(include_str!("../../../workloads/chat_golden.json")).unwrap();
    let template =
        Arc::new(ChatTemplate::from_str(golden["chat_template"].as_str().unwrap()).unwrap());
    app.template = template.clone();
    let router = build_router(Arc::new(app));
    let mut history = vec![json!({"role":"user","content":"check both"})];
    let mut ids = Vec::new();
    for result in ["found", "pong"] {
        let (status, body) = query(
            router.clone(),
            chat_request(json!({"messages":history}), None),
        )
        .await;
        assert_eq!(status, 200);
        let body: Value = serde_json::from_str(&body).unwrap();
        let message = &body["choices"][0]["message"];
        let id = &message["tool_calls"][0]["id"];
        ids.push(id.as_str().unwrap().to_owned());
        history.push(message.clone());
        history.push(json!({"role":"tool","tool_call_id":id,"content":result}));
    }
    assert_ne!(ids[0], ids[1]);
    let prompt = template.render(&history, None, None).unwrap();
    assert!(prompt.contains("<|message_tool|>lookup<|content_text|>found<|end_message|>"));
    assert!(prompt.contains("<|message_tool|>ping<|content_text|>pong<|end_message|>"));
}

#[tokio::test]
async fn context_window_finish_reason() {
    // Identical token counts can end at the context boundary or at a sampled stop.
    let reply = "ab";
    for (stop_reason, reason) in [
        (EngineStopReason::ContextLimit, "length"),
        (EngineStopReason::Eos, "stop"),
        (EngineStopReason::StopToken, "stop"),
    ] {
        let router = build_router(Arc::new(state(script_engine(reply, stop_reason))));
        let (status, body) = query(router, chat_request(json!({"max_tokens":10}), None)).await;
        assert_eq!(status, 200);
        let body: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(body["usage"]["prompt_tokens"], 5);
        assert_eq!(body["usage"]["completion_tokens"], reply.len());
        assert_eq!(body["choices"][0]["finish_reason"], reason);
    }
}

#[tokio::test]
async fn public_metrics_count_completed_responses_by_route_and_status() {
    let mut app = script("hello");
    app.config.api_key = Some("test".into());
    let router = build_router(Arc::new(app));
    let get = |path| HttpRequest::get(path).body(Body::empty()).unwrap();
    for path in ["/ready", "/health", "/health?probe=1", "/metrics"] {
        assert_eq!(query(router.clone(), get(path)).await.0, 200);
    }
    assert_eq!(query(router.clone(), get("/v1/models")).await.0, 401);
    let stream = || {
        let mut request = chat_request(json!({"stream":true}), None);
        request
            .headers_mut()
            .insert("authorization", "Bearer test".parse().unwrap());
        request
    };
    drop(router.clone().oneshot(stream()).await.unwrap());
    assert_eq!(query(router.clone(), stream()).await.0, 200);

    let response = router.oneshot(get("/metrics")).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers()["content-type"],
        "text/plain; version=0.0.4; charset=utf-8"
    );
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let text = std::str::from_utf8(&bytes).unwrap();
    assert!(text.contains("# TYPE flint_http_requests_total counter\n"));
    for (endpoint, status, count) in [
        ("/v1/chat/completions", "2xx", 1),
        ("/v1/models", "4xx", 1),
        ("/ready", "2xx", 1),
        ("/health", "2xx", 2),
        ("/metrics", "2xx", 1),
    ] {
        let expected = format!(
            "flint_http_requests_total{{endpoint=\"{endpoint}\",status=\"{status}\"}} {count}"
        );
        assert!(text.lines().any(|line| line == expected), "{expected}");
    }
}
