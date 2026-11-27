use clap::Parser;
use serde_json::{Value, json};
use server::config::Options;
use std::{fs, time::Duration};

#[derive(Parser)]
struct Cli {
    #[command(flatten)]
    options: Options,
}

#[test]
fn toml_cli_environment_and_engine_defaults() {
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("mini");
    fs::create_dir(&model).unwrap();
    let file = dir.path().join("server.toml");
    fs::write(&file,format!("model_dir = '{}'\ndraft_dir='draft'\napi_key='file'\nno_cuda_graph=true\nno_spec=true\nrequest_timeout_sec=0\nbind='localhost:1234'",model.display())).unwrap();
    let parse = |args: &[&str]| {
        Cli::parse_from(
            ["server", "--config", file.to_str().unwrap()]
                .into_iter()
                .chain(args.iter().copied()),
        )
        .options
    };
    let config = parse(&[]).resolve(None).unwrap();
    assert_eq!(config.api_key.as_deref(), Some("file"));
    assert_eq!(config.model_name, "mini");
    assert_eq!(config.temperature, 1.0);
    assert_eq!(config.default_max_tokens, 4096);
    assert_eq!(config.request_timeout, None);
    assert_eq!(
        serde_json::from_str::<Value>(&config.engine_json()).unwrap(),
        json!({"model_dir":model,"draft_model_dir":"draft","max_context":0,"prefix_cache_slots":3,"prefill_chunk_size":256,"use_cuda_graph":false,"use_speculative":false})
    );
    fs::write(
        model.join("generation_config.json"),
        r#"{"temperature":0.7}"#,
    )
    .unwrap();
    assert_eq!(
        parse(&[])
            .resolve(Some("env".into()))
            .unwrap()
            .api_key
            .as_deref(),
        Some("env")
    );
    let config = parse(&[
        "--draft-dir",
        "override",
        "--api-key",
        "cli",
        "--model-name",
        "alias",
        "--no-cuda-graph=false",
        "--no-spec=false",
        "--request-timeout-sec",
        "12",
    ])
    .resolve(Some("env".into()))
    .unwrap();
    assert_eq!(config.api_key.as_deref(), Some("cli"));
    assert_eq!(
        config.draft_dir.unwrap(),
        std::path::PathBuf::from("override")
    );
    assert_eq!(config.temperature, 0.7);
    assert_eq!(config.model_name, "alias");
    assert!(config.use_cuda_graph && config.use_speculative);
    let cli = |flags: &[&str]| {
        Cli::parse_from(
            ["server", "--model-dir", model.to_str().unwrap()]
                .into_iter()
                .chain(flags.iter().copied()),
        )
        .options
        .resolve(None)
        .unwrap()
    };
    assert!(cli(&[]).use_speculative);
    assert!(!cli(&["--no-spec"]).use_speculative);
    assert_eq!(config.request_timeout, Some(Duration::from_secs(12)));
    assert_eq!(config.bind, "localhost:1234");
}

#[test]
#[cfg(not(feature = "link-engine"))]
fn smoke_binary() {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_server"))
        .arg("--smoke")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&output.stdout).unwrap()["backend"],
        "stub"
    );
}
