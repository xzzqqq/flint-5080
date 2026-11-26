use axum::{
    body::Body,
    extract::{MatchedPath, Request, State},
    http::{StatusCode, header::CONTENT_TYPE},
    middleware::Next,
    response::{IntoResponse, Response},
};
use serde::Deserialize;
use std::{
    fmt::Write,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};
use tokio_stream::StreamExt;

use crate::http::AppState;

const ENDPOINTS: [&str; 5] = [
    "/v1/chat/completions",
    "/v1/models",
    "/ready",
    "/health",
    "/metrics",
];
const CONTENT_TYPE_VALUE: &str = "text/plain; version=0.0.4; charset=utf-8";

#[derive(Default)]
pub struct HttpMetrics {
    counts: [[AtomicU64; 5]; ENDPOINTS.len()],
}

pub(crate) async fn count(
    State(state): State<Arc<AppState>>,
    path: MatchedPath,
    req: Request,
    next: Next,
) -> Response {
    let response = next.run(req).await;
    let Some(endpoint) = ENDPOINTS.iter().position(|p| *p == path.as_str()) else {
        return response;
    };
    let class = usize::from(response.status().as_u16() / 100 - 1);
    if response
        .headers()
        .get(CONTENT_TYPE)
        .is_some_and(|v| v == "text/event-stream")
    {
        // Headers are ready before generation finishes. Count only fully consumed streams.
        let (parts, body) = response.into_parts();
        let mut data = body.into_data_stream();
        let body = Body::from_stream(async_stream::stream! {
            while let Some(chunk) = data.next().await {
                if chunk.is_err() {
                    yield chunk;
                    return;
                }
                yield chunk;
            }
            state.metrics.counts[endpoint][class].fetch_add(1, Ordering::Relaxed);
        });
        return Response::from_parts(parts, body);
    }
    state.metrics.counts[endpoint][class].fetch_add(1, Ordering::Relaxed);
    response
}

#[derive(Default, Deserialize)]
#[serde(default)]
struct EngineStats {
    vram_total_bytes: u64,
    vram_free_bytes: u64,
    submitted_requests: u64,
    completed_requests: u64,
    cancelled_requests: u64,
    active_requests: u64,
    queued_requests: u64,
    total_prefill_tokens: u64,
    total_cached_prefix_tokens: u64,
    total_decode_tokens: u64,
    total_decode_rounds: u64,
    total_spec_accepted: u64,
    prefix_cache_hits: u64,
    prefix_cache_misses: u64,
    prefix_cache_saved_tokens: u64,
    prefix_cache_used_slots: u64,
    prefix_cache_slots: u64,
    graph_captures: u64,
    graph_launches: u64,
    speculative_active: bool,
}

fn sample(out: &mut String, name: &str, kind: &str, help: &str, value: u64) {
    writeln!(
        out,
        "# HELP flint_{name} {help}\n# TYPE flint_{name} {kind}\nflint_{name} {value}"
    )
    .unwrap();
}

impl HttpMetrics {
    fn response(&self, json: Option<&str>) -> Response {
        let stats = json
            .filter(|s| s.trim_start().starts_with('{'))
            .and_then(|s| serde_json::from_str::<EngineStats>(s).ok());
        let headers = [(CONTENT_TYPE, CONTENT_TYPE_VALUE)];
        let mut out = String::new();
        sample(
            &mut out,
            "up",
            "gauge",
            "Whether engine statistics are available.",
            stats.is_some().into(),
        );
        let Some(s) = stats else {
            return (StatusCode::SERVICE_UNAVAILABLE, headers, out).into_response();
        };

        for (name, help, value) in [
            (
                "vram_total_bytes",
                "Total GPU memory in bytes.",
                s.vram_total_bytes,
            ),
            (
                "vram_free_bytes",
                "Free GPU memory in bytes.",
                s.vram_free_bytes,
            ),
            (
                "vram_used_bytes",
                "Used GPU memory in bytes.",
                s.vram_total_bytes.saturating_sub(s.vram_free_bytes),
            ),
            (
                "requests_active",
                "Requests currently active.",
                s.active_requests,
            ),
            (
                "requests_queued",
                "Requests currently queued.",
                s.queued_requests,
            ),
            (
                "prefix_cache_slots_used",
                "Occupied prefix cache slots.",
                s.prefix_cache_used_slots,
            ),
            (
                "prefix_cache_slots_total",
                "Configured prefix cache slots.",
                s.prefix_cache_slots,
            ),
            (
                "speculative_active",
                "Whether speculative decoding is active.",
                s.speculative_active.into(),
            ),
        ] {
            sample(&mut out, name, "gauge", help, value);
        }
        for (name, help, value) in [
            (
                "requests_submitted_total",
                "Requests submitted to the engine.",
                s.submitted_requests,
            ),
            (
                "requests_completed_total",
                "Requests completed by the engine.",
                s.completed_requests,
            ),
            (
                "requests_cancelled_total",
                "Engine request cancellations.",
                s.cancelled_requests,
            ),
            (
                "prefill_tokens_total",
                "Prefill tokens reported by the engine.",
                s.total_prefill_tokens,
            ),
            (
                "cached_prefix_tokens_total",
                "Cached prefix tokens reported by the engine.",
                s.total_cached_prefix_tokens,
            ),
            (
                "decode_tokens_total",
                "Tokens decoded.",
                s.total_decode_tokens,
            ),
            (
                "decode_rounds_total",
                "Decode rounds executed.",
                s.total_decode_rounds,
            ),
            (
                "spec_accepted_tokens_total",
                "Draft tokens accepted during speculative decoding.",
                s.total_spec_accepted,
            ),
            (
                "prefix_cache_hits_total",
                "Prefix cache hits.",
                s.prefix_cache_hits,
            ),
            (
                "prefix_cache_misses_total",
                "Prefix cache misses.",
                s.prefix_cache_misses,
            ),
            (
                "prefix_cache_saved_tokens_total",
                "Tokens saved by prefix cache reuse.",
                s.prefix_cache_saved_tokens,
            ),
            (
                "cuda_graph_captures_total",
                "CUDA graph captures.",
                s.graph_captures,
            ),
            (
                "cuda_graph_launches_total",
                "CUDA graph launches.",
                s.graph_launches,
            ),
        ] {
            sample(&mut out, name, "counter", help, value);
        }

        out.push_str("# HELP flint_http_requests_total Completed HTTP responses by endpoint and status class.\n# TYPE flint_http_requests_total counter\n");
        for (endpoint, counts) in ENDPOINTS.iter().zip(&self.counts) {
            for (class, count) in (1..=5).zip(counts) {
                let value = count.load(Ordering::Relaxed);
                writeln!(out, "flint_http_requests_total{{endpoint=\"{endpoint}\",status=\"{class}xx\"}} {value}").unwrap();
            }
        }
        (headers, out).into_response()
    }
}

pub(crate) async fn scrape(State(s): State<Arc<AppState>>) -> Response {
    s.metrics.response(s.engine.stats_json().ok().as_deref())
}

#[cfg(test)]
mod tests {
    use super::*;
    use http_body_util::BodyExt;

    async fn body(response: Response) -> String {
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        String::from_utf8(bytes.to_vec()).unwrap()
    }

    #[tokio::test]
    async fn renders_engine_snapshot() {
        let json = r#"{
            "vram_total_bytes": 17179869184, "vram_free_bytes": 17,
            "submitted_requests": 1, "completed_requests": 2, "cancelled_requests": 3,
            "active_requests": 4, "queued_requests": 5, "total_prefill_tokens": 6,
            "total_cached_prefix_tokens": 7, "total_decode_tokens": 8,
            "total_decode_rounds": 9, "total_spec_accepted": 10,
            "prefix_cache_hits": 11, "prefix_cache_misses": 12,
            "prefix_cache_saved_tokens": 13, "prefix_cache_used_slots": 14,
            "prefix_cache_slots": 15, "graph_captures": 16, "graph_launches": 18,
            "speculative_active": true
        }"#;
        let text = body(HttpMetrics::default().response(Some(json))).await;
        assert!(text.ends_with('\n'));
        for (metric, kind, value) in [
            ("up", "gauge", 1),
            ("vram_total_bytes", "gauge", 17179869184u64),
            ("vram_free_bytes", "gauge", 17),
            ("vram_used_bytes", "gauge", 17179869167),
            ("requests_submitted_total", "counter", 1),
            ("requests_completed_total", "counter", 2),
            ("requests_cancelled_total", "counter", 3),
            ("requests_active", "gauge", 4),
            ("requests_queued", "gauge", 5),
            ("prefill_tokens_total", "counter", 6),
            ("cached_prefix_tokens_total", "counter", 7),
            ("decode_tokens_total", "counter", 8),
            ("decode_rounds_total", "counter", 9),
            ("spec_accepted_tokens_total", "counter", 10),
            ("prefix_cache_hits_total", "counter", 11),
            ("prefix_cache_misses_total", "counter", 12),
            ("prefix_cache_saved_tokens_total", "counter", 13),
            ("prefix_cache_slots_used", "gauge", 14),
            ("prefix_cache_slots_total", "gauge", 15),
            ("cuda_graph_captures_total", "counter", 16),
            ("cuda_graph_launches_total", "counter", 18),
            ("speculative_active", "gauge", 1),
        ] {
            let name = format!("flint_{metric}");
            let mut lines = text
                .lines()
                .skip_while(|line| !line.starts_with(&format!("# HELP {name} ")));
            assert!(!lines.next().unwrap().ends_with(' '));
            assert_eq!(lines.next().unwrap(), format!("# TYPE {name} {kind}"));
            assert_eq!(lines.next().unwrap(), format!("{name} {value}"));
        }
    }

    #[tokio::test]
    async fn missing_fields_default_to_zero_and_used_vram_saturates() {
        let text = body(HttpMetrics::default().response(Some(r#"{"vram_free_bytes":1}"#))).await;
        for sample in [
            "flint_up 1",
            "flint_vram_used_bytes 0",
            "flint_speculative_active 0",
            "flint_decode_tokens_total 0",
        ] {
            assert!(text.lines().any(|line| line == sample));
        }
    }

    #[tokio::test]
    async fn unavailable_or_invalid_stats_render_only_up() {
        for stats in [
            None,
            Some("{"),
            Some("[]"),
            Some(r#"{"graph_launches":"invalid"}"#),
        ] {
            let response = HttpMetrics::default().response(stats);
            assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
            assert_eq!(
                response.headers()[CONTENT_TYPE],
                "text/plain; version=0.0.4; charset=utf-8"
            );
            assert_eq!(
                body(response).await,
                "# HELP flint_up Whether engine statistics are available.\n# TYPE flint_up gauge\nflint_up 0\n"
            );
        }
    }
}
