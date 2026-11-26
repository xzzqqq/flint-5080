mod checks;

use clap::Parser;
use server::{BoxError, engine_ffi::Engine, template::ChatTemplate, tokenizer::Tokenizer};
use std::{path::PathBuf, process::ExitCode};

#[derive(Parser)]
#[command(version, about = "Flint-5080 inference server")]
struct Cli {
    #[arg(long, conflicts_with = "check_chat_golden")]
    smoke: bool,
    #[arg(long, requires_all = ["model_dir", "golden"])]
    check_chat_golden: bool,
    #[command(flatten)]
    options: server::config::Options,
    #[arg(long, requires = "check_chat_golden")]
    golden: Option<PathBuf>,
    #[arg(long, requires = "check_chat_golden")]
    golden_cases: Option<PathBuf>,
    #[arg(long, requires = "smoke", default_value = "{}")]
    config_json: String,
}

#[tokio::main]
async fn serve(options: server::config::Options) -> Result<(), BoxError> {
    use server::http::{AppState, build_router};
    use std::sync::Arc;
    let config = options.resolve(std::env::var("FLINT_API_KEY").ok())?;
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_new(&config.log_level)?)
        .init();
    // Reserve the address before loading weights so a readiness probe cannot
    // mistake an existing listener for a second server still loading.
    let listener = tokio::net::TcpListener::bind(&config.bind).await?;
    let template = Arc::new(ChatTemplate::from_model_dir(&config.model_dir)?);
    let tokenizer = Arc::new(Tokenizer::from_model_dir(&config.model_dir)?);
    let engine = Arc::new(Engine::new(&config.engine_json())?);
    let router = build_router(Arc::new(AppState {
        metrics: Default::default(),
        config,
        template,
        tokenizer,
        engine,
    }));
    let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
    axum::serve(listener, router)
        .with_graceful_shutdown(async move {
            tokio::select! { _ = tokio::signal::ctrl_c() => (), _ = term.recv() => () }
        })
        .await?;
    Ok(())
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = if cli.check_chat_golden {
        checks::check_chat_golden(
            cli.options.model_dir.as_deref().unwrap(),
            cli.golden.as_deref().unwrap(),
            cli.golden_cases.as_deref(),
        )
    } else if cli.smoke {
        checks::smoke(&cli.config_json)
    } else {
        serve(cli.options)
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            let mode = if cli.smoke {
                "smoke"
            } else if cli.check_chat_golden {
                "check-chat-golden"
            } else {
                "serve"
            };
            eprintln!("server {mode}: {e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_accepts_serve_smoke_config_and_version() {
        assert!(Cli::try_parse_from(["server"]).is_ok());
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
