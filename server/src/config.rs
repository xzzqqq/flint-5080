use crate::BoxError;
use clap::Args;
use serde::Deserialize;
use serde_json::{Value, json};
use std::{fs, io::ErrorKind, path::PathBuf, time::Duration};

#[derive(Args, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Options {
    #[arg(long)]
    #[serde(skip)]
    pub config: Option<PathBuf>,
    #[arg(long)]
    pub bind: Option<String>,
    #[arg(long)]
    pub model_dir: Option<PathBuf>,
    #[arg(long)]
    pub model_name: Option<String>,
    #[arg(long)]
    pub api_key: Option<String>,
    #[arg(long)]
    pub max_context: Option<u32>,
    #[arg(long)]
    pub prefill_chunk_size: Option<u32>,
    #[arg(long)]
    pub prefix_cache_slots: Option<u32>,
    #[arg(long, num_args=0..=1, require_equals=true, default_missing_value="true")]
    pub no_cuda_graph: Option<bool>,
    #[arg(long)]
    pub default_max_tokens: Option<u32>,
    #[arg(long)]
    pub request_timeout_sec: Option<u64>,
    #[arg(long)]
    pub log_level: Option<String>,
}

pub struct Config {
    pub bind: String,
    pub model_dir: PathBuf,
    pub model_name: String,
    pub api_key: Option<String>,
    pub max_context: u32,
    pub prefill_chunk_size: u32,
    pub prefix_cache_slots: u32,
    pub use_cuda_graph: bool,
    pub default_max_tokens: u32,
    pub request_timeout: Option<Duration>,
    pub log_level: String,
    pub temperature: f32,
}

impl Options {
    pub fn resolve(self, env_api_key: Option<String>) -> Result<Config, BoxError> {
        let file: Options = match self.config {
            Some(path) => toml::from_str(&fs::read_to_string(path)?)?,
            None => Options::default(),
        };
        let model_dir = self
            .model_dir
            .or(file.model_dir)
            .ok_or("model_dir is required")?;
        let model_name = self
            .model_name
            .or(file.model_name)
            .or_else(|| {
                model_dir
                    .file_name()
                    .and_then(|s| s.to_str())
                    .map(String::from)
            })
            .or_else(|| {
                model_dir
                    .canonicalize()
                    .ok()?
                    .file_name()?
                    .to_str()
                    .map(String::from)
            })
            .ok_or("model_dir has no basename; set model_name")?;
        let temperature = match fs::read(model_dir.join("generation_config.json")) {
            Ok(bytes) => serde_json::from_slice::<Value>(&bytes)?["temperature"]
                .as_f64()
                .unwrap_or(1.0) as f32,
            Err(e) if e.kind() == ErrorKind::NotFound => 1.0,
            Err(e) => return Err(e.into()),
        };
        let seconds = self
            .request_timeout_sec
            .or(file.request_timeout_sec)
            .unwrap_or(1800);
        Ok(Config {
            bind: self
                .bind
                .or(file.bind)
                .unwrap_or_else(|| "127.0.0.1:8080".into()),
            model_dir,
            model_name,
            temperature,
            api_key: self.api_key.or(env_api_key).or(file.api_key),
            max_context: self.max_context.or(file.max_context).unwrap_or(0),
            prefill_chunk_size: self
                .prefill_chunk_size
                .or(file.prefill_chunk_size)
                .unwrap_or(256),
            prefix_cache_slots: self
                .prefix_cache_slots
                .or(file.prefix_cache_slots)
                .unwrap_or(3),
            use_cuda_graph: !self.no_cuda_graph.or(file.no_cuda_graph).unwrap_or(false),
            default_max_tokens: self
                .default_max_tokens
                .or(file.default_max_tokens)
                .unwrap_or(4096),
            request_timeout: (seconds != 0).then(|| Duration::from_secs(seconds)),
            log_level: self
                .log_level
                .or(file.log_level)
                .unwrap_or_else(|| "info".into()),
        })
    }
}

impl Config {
    pub fn engine_json(&self) -> String {
        json!({"model_dir": self.model_dir, "max_context": self.max_context,
            "prefill_chunk_size": self.prefill_chunk_size, "prefix_cache_slots": self.prefix_cache_slots,
            "use_cuda_graph": self.use_cuda_graph}).to_string()
    }
}
