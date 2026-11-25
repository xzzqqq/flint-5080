use minijinja::{Environment, Error, ErrorKind, context, value::Kwargs};
use regex::{Captures, Regex};
use serde_json::Value;
use std::{fs, path::Path};

pub type TemplateError = Error;

pub struct ChatTemplate {
    env: Environment<'static>,
}

fn invalid(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidOperation, message.into())
}

// HF overrides Jinja's HTML-safe filter with Python json.dumps(ensure_ascii=False).
fn hf_tojson(value: minijinja::Value, kwargs: Kwargs) -> Result<String, Error> {
    let mut value = serde_json::to_value(value).map_err(|e| invalid(e.to_string()))?;
    if kwargs.get::<Option<bool>>("sort_keys")?.unwrap_or(false) {
        value.sort_all_objects();
    }
    let separators = kwargs
        .get::<Option<Vec<String>>>("separators")?
        .unwrap_or_else(|| vec![", ".into(), ": ".into()]);
    let [comma, colon] = separators.as_slice() else {
        return Err(invalid("expected two JSON separators"));
    };
    kwargs.assert_all_used()?;
    let mut out = String::new();
    write_json(&value, comma, colon, &mut out);
    Ok(out)
}

fn write_json(value: &Value, comma: &str, colon: &str, out: &mut String) {
    match value {
        Value::Object(map) => {
            out.push('{');
            for (index, (key, value)) in map.iter().enumerate() {
                if index != 0 {
                    out.push_str(comma);
                }
                out.push_str(&serde_json::to_string(key).expect("JSON key"));
                out.push_str(colon);
                write_json(value, comma, colon, out);
            }
            out.push('}');
        }
        Value::Array(items) => {
            out.push('[');
            for (index, value) in items.iter().enumerate() {
                if index != 0 {
                    out.push_str(comma);
                }
                write_json(value, comma, colon, out);
            }
            out.push(']');
        }
        Value::Number(n) if n.is_f64() => out.push_str(&python_float(n.as_f64().unwrap())),
        _ => out.push_str(&value.to_string()),
    }
}

// Python repr uses a signed exponent with at least two digits.
fn python_float(value: f64) -> String {
    let repr = format!("{value:?}");
    match repr.split_once('e') {
        Some((mantissa, exp)) => format!("{mantissa}e{:+03}", exp.parse::<i32>().unwrap()),
        None => repr,
    }
}

impl ChatTemplate {
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(source: &str) -> Result<Self, TemplateError> {
        // Transparent blocks retain both whitespace controls and HF trim_blocks behavior.
        let tags = Regex::new(r"(\{%-?\s*)(endgeneration|generation)(\s*-?%\})").unwrap();
        let source = tags.replace_all(source, |m: &Captures<'_>| {
            let tag = if &m[2] == "generation" {
                "if true"
            } else {
                "endif"
            };
            format!("{}{tag}{}", &m[1], &m[3])
        });
        let mut env = Environment::new();
        env.set_formatter(|out, state, value| {
            if value.is_number() && !value.is_integer() {
                Ok(out.write_str(&python_float(f64::try_from(value.clone())?))?)
            } else {
                minijinja::escape_formatter(out, state, value)
            }
        });
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.add_function("raise_exception", |msg: String| -> Result<String, Error> {
            Err(invalid(msg))
        });
        env.add_filter("tojson", hf_tojson);
        // MiniJinja's built-in accepts the same case_sensitive keyword as Jinja.
        env.add_filter("dictsort", minijinja::filters::dictsort);
        env.add_template_owned("chat", source.into_owned())?;
        Ok(Self { env })
    }

    pub fn from_model_dir(dir: &Path) -> Result<Self, TemplateError> {
        let path = dir.join("chat_template.jinja");
        Self::from_str(
            &fs::read_to_string(&path).map_err(|e| invalid(format!("{}: {e}", path.display())))?,
        )
    }

    pub fn render(
        &self,
        messages: &[Value],
        tools: Option<&[Value]>,
        reasoning_effort: Option<&Value>,
    ) -> Result<String, TemplateError> {
        let mut messages = messages.to_vec();
        for message in &mut messages {
            if message["role"] != "assistant" {
                continue;
            }
            if let Some(calls) = message.get_mut("tool_calls").and_then(Value::as_array_mut) {
                for call in calls {
                    if let Some(args) = call
                        .get_mut("function")
                        .and_then(|f| f.get_mut("arguments"))
                    {
                        if let Some(s) = args.as_str() {
                            *args = serde_json::from_str(if s.is_empty() { "{}" } else { s })
                                .map_err(|e| {
                                    invalid(format!("invalid tool call arguments: {e}"))
                                })?;
                        }
                        if !args.is_object() {
                            return Err(invalid("tool call arguments must be an object"));
                        }
                    }
                }
            }
        }
        self.env
            .get_template("chat")?
            .render(context! { messages, tools, reasoning_effort })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hf_golden() {
        let golden: Value =
            serde_json::from_str(include_str!("../../workloads/chat_golden.json")).unwrap();
        let template = ChatTemplate::from_str(golden["chat_template"].as_str().unwrap()).unwrap();
        for group in ["template_cases", "template_error_cases"] {
            for row in golden[group].as_array().unwrap() {
                let result = template.render(
                    row["messages"].as_array().unwrap(),
                    row["tools"].as_array().map(Vec::as_slice),
                    row.get("reasoning_effort"),
                );
                if group == "template_cases" {
                    assert_eq!(result.unwrap(), row["rendered"], "{}", row["name"]);
                } else {
                    assert!(result.is_err(), "{}", row["name"]);
                }
            }
        }
    }
}
