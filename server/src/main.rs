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
    let (id, events) = engine.submit(&Request {
        token_ids: &[1, 2, 3, 4],
        max_tokens: 1,
        ..Default::default()
    })?;
    let [Event::Tokens(tokens), Event::Done(t)] = events.as_slice() else {
        return Err("expected tokens followed by done".into());
    };
    if tokens != &[42] || t.prefill_tokens != 4 || t.decode_tokens != 1 || t.decode_rounds != 1 {
        return Err("unexpected smoke tokens or done telemetry".into());
    }
    engine.cancel(id);
    let final_stats: Value = serde_json::from_str(&engine.stats_json()?)?;
    if final_stats["submitted_requests"] != 1 || final_stats["cancelled_requests"] != 1 {
        return Err("unexpected request counters".into());
    }
    println!(
        "{}",
        serde_json::json!({"backend": stats.get("backend").unwrap_or(&stats["gpu_name"]),
                                        "req_id": id, "tokens": tokens, "telemetry": t})
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
