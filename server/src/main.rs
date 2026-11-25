use clap::Parser;
use serde_json::Value;
use server::engine_ffi::{Engine, Event, FLINT_ENGINE_ABI_VERSION, Request};
use std::{error::Error, process::ExitCode};

#[derive(Parser)]
#[command(version, about = "Flint-5080 engine smoke runner")]
struct Cli {
    #[arg(long, required = true)]
    smoke: bool,
    #[arg(long, default_value = "{}")]
    config_json: String,
}

fn smoke(config: &str) -> Result<(), Box<dyn Error>> {
    let engine = Engine::new(config)?;
    let stats: Value = serde_json::from_str(&engine.stats_json()?)?;
    if stats["abi_version"] != FLINT_ENGINE_ABI_VERSION {
        return Err("unexpected engine ABI version in stats".into());
    }
    let config: Value = serde_json::from_str(config)?;
    let model = config["model_dir"]
        .as_str()
        .is_some_and(|path| !path.is_empty());
    let mode = if model { "model" } else { "smoke" };
    if stats["mode"] != mode {
        return Err("unexpected engine mode".into());
    }
    let prompt: &[i32] = if model {
        // short_math from workloads/golden_cases.json.
        &[
            200002, 200004, 34182, 279, 154554, 150628, 25, 220, 15, 200010, 200000, 200004, 1775,
            149884, 220, 68107, 196987, 220, 68169, 30, 154710, 149899, 150239, 149876, 150107, 13,
            200010,
        ]
    } else {
        &[1, 2, 3, 4]
    };
    let limit = if model { 4 } else { 1 };
    let (id, events) = engine.submit(&Request {
        token_ids: prompt,
        max_tokens: limit,
        ..Default::default()
    })?;
    let Some((Event::Done(t), chunks)) = events.split_last() else {
        return Err("missing terminal done".into());
    };
    let mut tokens = Vec::new();
    for chunk in chunks {
        let Event::Tokens(ids) = chunk else {
            return Err("premature done".into());
        };
        tokens.extend_from_slice(ids);
    }
    if tokens.len() != limit as usize
        || t.prefill_tokens != prompt.len() as u32
        || t.decode_tokens != limit
        || if model {
            t.decode_rounds != 3
                || !(t.prefill_ms > 0.0 && t.decode_ms > 0.0)
                || tokens.iter().any(|id| !(0..200058).contains(id))
        } else {
            tokens != [42] || t.decode_rounds != 1
        }
    {
        return Err("unexpected smoke tokens or telemetry".into());
    }
    let final_stats: Value = serde_json::from_str(&engine.stats_json()?)?;
    if final_stats["submitted_requests"] != 1
        || final_stats["completed_requests"] != 1
        || final_stats["cancelled_requests"] != 0
        || final_stats["queued_requests"] != 0
        || final_stats["active_requests"] != 0
    {
        return Err("unexpected request counters".into());
    }
    println!(
        "{}",
        serde_json::json!({"backend": stats.get("backend").unwrap_or(&stats["gpu_name"]),
                                        "mode":mode,"req_id": id, "tokens": tokens, "telemetry": t})
    );
    Ok(())
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match smoke(&cli.config_json) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("server smoke: {e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_requires_smoke_and_accepts_config_and_version() {
        assert!(Cli::try_parse_from(["server"]).is_err());
        let cli = Cli::try_parse_from(["server", "--smoke"]).unwrap();
        assert!(cli.smoke);
        assert_eq!(cli.config_json, "{}");
        let cli =
            Cli::try_parse_from(["server", "--config-json", r#"{"test":1}"#, "--smoke"]).unwrap();
        assert_eq!(cli.config_json, r#"{"test":1}"#);
        let err = Cli::try_parse_from(["server", "--version"]).err().unwrap();
        assert_eq!(err.kind(), clap::error::ErrorKind::DisplayVersion);
    }
}
