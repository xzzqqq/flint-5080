use clap::Parser;
use serde::Deserialize;
use serde_json::Value;
use server::engine_ffi::{Engine, Event, FLINT_ENGINE_ABI_VERSION, Request};
use server::{
    template::ChatTemplate,
    tokenizer::{Tokenizer, TokenizerError},
};
use std::process::ExitCode;
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Parser)]
#[command(version, about = "Flint-5080 server checks")]
struct Cli {
    #[arg(
        long,
        required_unless_present = "check_chat_golden",
        conflicts_with = "check_chat_golden"
    )]
    smoke: bool,
    #[arg(long, requires_all = ["model_dir", "golden"])]
    check_chat_golden: bool,
    #[arg(long, requires = "check_chat_golden")]
    model_dir: Option<PathBuf>,
    #[arg(long, requires = "check_chat_golden")]
    golden: Option<PathBuf>,
    #[arg(long, requires = "check_chat_golden")]
    golden_cases: Option<PathBuf>,
    #[arg(long, requires = "smoke", default_value = "{}")]
    config_json: String,
}

fn smoke(config: &str) -> Result<(), TokenizerError> {
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

#[derive(Deserialize)]
struct Prompt {
    name: String,
    messages: Vec<Value>,
    tools: Option<Vec<Value>>,
    reasoning_effort: Option<Value>,
}

impl Prompt {
    fn render(&self, template: &ChatTemplate) -> Result<String, server::template::TemplateError> {
        template.render(
            &self.messages,
            self.tools.as_deref(),
            self.reasoning_effort.as_ref(),
        )
    }
}

#[derive(Deserialize)]
struct TemplateCase {
    #[serde(flatten)]
    prompt: Prompt,
    rendered: String,
    input_ids: Vec<i32>,
}

#[derive(Deserialize)]
struct TextCase {
    name: String,
    text: String,
    token_ids: Vec<i32>,
    decoded_keep_special: String,
    decoded_skip_special: String,
}

#[derive(Deserialize)]
struct ChatGolden {
    chat_template: String,
    template_cases: Vec<TemplateCase>,
    template_error_cases: Vec<Prompt>,
    tokenizer_cases: Vec<TextCase>,
    special_tokens: std::collections::BTreeMap<String, i32>,
}

fn check_chat_golden(
    model: &Path,
    path: &Path,
    reference: Option<&Path>,
) -> Result<(), TokenizerError> {
    let golden: ChatGolden = serde_json::from_slice(&fs::read(path)?)?;
    let template = ChatTemplate::from_model_dir(model)?;
    if fs::read_to_string(model.join("chat_template.jinja"))? != golden.chat_template {
        return Err("checkpoint chat_template.jinja differs from fixture".into());
    }
    let tok = Tokenizer::from_model_dir(model)?;
    for (name, id) in &golden.special_tokens {
        let actual = match name.as_str() {
            "eos_token_id" => vec![tok.special.eos_token_id],
            "pad_token_id" => vec![tok.special.pad_token_id],
            _ => tok.encode(&format!("<|{name}|>"))?,
        };
        if actual != [*id] || !tok.special.is_special(*id) {
            return Err(format!("special token mismatch for {name}").into());
        }
    }
    for case in &golden.template_cases {
        let rendered = case.prompt.render(&template)?;
        if rendered != case.rendered {
            return Err(format!("{}: rendered prompt mismatch", case.prompt.name).into());
        }
        if tok.encode(&rendered)? != case.input_ids {
            return Err(format!("{}: input_ids mismatch", case.prompt.name).into());
        }
    }
    for case in &golden.template_error_cases {
        if case.render(&template).is_ok() {
            return Err(format!("{}: expected a template error", case.name).into());
        }
    }
    for case in &golden.tokenizer_cases {
        if tok.encode(&case.text)? != case.token_ids {
            return Err(format!("{}: token_ids mismatch", case.name).into());
        }
        for (skip, expected) in [
            (false, &case.decoded_keep_special),
            (true, &case.decoded_skip_special),
        ] {
            if tok.decode(&case.token_ids, skip)? != *expected {
                return Err(format!("{}: decode mismatch (skip_special={skip})", case.name).into());
            }
        }
        let mut decoder = tok.stream_decoder();
        let mut stream = String::new();
        for &id in &case.token_ids {
            stream.push_str(&decoder.push(id));
        }
        stream.push_str(&decoder.finish());
        if stream != case.decoded_skip_special {
            return Err(format!("{}: stream decode mismatch", case.name).into());
        }
    }
    if let Some(path) = reference {
        #[derive(Deserialize)]
        struct Reference {
            cases: Vec<ReferenceCase>,
        }
        #[derive(Deserialize)]
        struct ReferenceCase {
            #[serde(flatten)]
            prompt: Prompt,
            input_ids: Vec<i32>,
        }
        let fixed: Reference = serde_json::from_slice(&fs::read(path)?)?;
        for case in fixed.cases {
            if tok.encode(&case.prompt.render(&template)?)? != case.input_ids {
                return Err(format!("{}: reference input_ids mismatch", case.prompt.name).into());
            }
        }
    }
    println!(
        "Chat golden passed: {} templates, {} expected errors, {} tokenizer cases (including streaming)",
        golden.template_cases.len(),
        golden.template_error_cases.len(),
        golden.tokenizer_cases.len()
    );
    Ok(())
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = if cli.check_chat_golden {
        check_chat_golden(
            cli.model_dir.as_deref().unwrap(),
            cli.golden.as_deref().unwrap(),
            cli.golden_cases.as_deref(),
        )
    } else {
        smoke(&cli.config_json)
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            let mode = if cli.smoke {
                "smoke"
            } else {
                "check-chat-golden"
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
