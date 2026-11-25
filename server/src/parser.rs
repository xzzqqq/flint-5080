use crate::tokenizer::SpecialTokens;
use serde_json::Value;
use std::mem;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedToolCall {
    pub index: usize,
    pub id: String,
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseDelta {
    Reasoning(String),
    Content(String),
    ToolCall(ParsedToolCall),
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct ParsedMessage {
    pub reasoning_content: Option<String>,
    pub content: Option<String>,
    pub tool_calls: Vec<ParsedToolCall>,
}

impl Extend<ParseDelta> for ParsedMessage {
    fn extend<T: IntoIterator<Item = ParseDelta>>(&mut self, deltas: T) {
        for delta in deltas {
            match delta {
                ParseDelta::Reasoning(s) => self
                    .reasoning_content
                    .get_or_insert_with(String::new)
                    .push_str(&s),
                ParseDelta::Content(s) => self.content.get_or_insert_with(String::new).push_str(&s),
                ParseDelta::ToolCall(call) => self.tool_calls.push(call),
            }
        }
    }
}

pub trait TokenParser {
    fn push_token(&mut self, token_id: i32, text_delta: &str) -> Vec<ParseDelta>;
    fn finish(&mut self) -> Vec<ParseDelta>;
}

pub fn parse_stream_to_message<'a>(
    parser: &mut dyn TokenParser,
    tokens: impl IntoIterator<Item = (i32, &'a str)>,
) -> ParsedMessage {
    let mut message = ParsedMessage::default();
    for (id, text) in tokens {
        message.extend(parser.push_token(id, text));
    }
    message.extend(parser.finish());
    message
}

#[derive(Default)]
enum State {
    #[default]
    Idle,
    Header(String),
    Reasoning,
    Content,
    ToolCallJson {
        header: String,
        payload: String,
    },
}

pub struct ResponseParser {
    special: SpecialTokens,
    state: State,
    next_index: usize,
}

impl ResponseParser {
    pub fn new(special: SpecialTokens) -> Self {
        Self {
            special,
            state: State::Idle,
            next_index: 0,
        }
    }

    fn text(&mut self, text: &str, out: &mut Vec<ParseDelta>) {
        if text.is_empty() {
            return;
        }
        match &mut self.state {
            State::Header(name) => name.push_str(text),
            State::ToolCallJson { payload, .. } => payload.push_str(text),
            State::Reasoning => out.push(ParseDelta::Reasoning(text.into())),
            State::Idle | State::Content => {
                self.state = State::Content;
                out.push(ParseDelta::Content(text.into()));
            }
        }
    }

    pub fn push_token(&mut self, token_id: i32, text_delta: &str) -> Vec<ParseDelta> {
        let mut out = Vec::new();
        // StreamDecoder may flush a trailing incomplete character at a delimiter.
        let awaiting_type = matches!(self.state, State::Idle | State::Header(_));
        self.text(text_delta, &mut out);
        let s = &self.special;
        if token_id == s.end_message || token_id == s.eos_token_id {
            out.extend(self.finish());
        } else if awaiting_type {
            if token_id == s.message_model {
                self.state = State::Header(String::new());
            } else if token_id == s.content_thinking {
                self.state = State::Reasoning;
            } else if [
                s.content_text,
                s.content_tool_error,
                s.content_invoke_tool_text,
            ]
            .contains(&token_id)
            {
                self.state = State::Content;
            } else if token_id == s.content_invoke_tool_json {
                let header = match mem::take(&mut self.state) {
                    State::Header(name) => name,
                    _ => String::new(),
                };
                self.state = State::ToolCallJson {
                    header,
                    payload: String::new(),
                };
            }
        }
        out
    }

    pub fn finish(&mut self) -> Vec<ParseDelta> {
        let State::ToolCallJson { header, payload } = mem::take(&mut self.state) else {
            return Vec::new();
        };
        let call = serde_json::from_str::<Value>(&payload)
            .ok()
            .and_then(|value| {
                let name = match value.get("name") {
                    Some(name) => name.as_str()?,
                    None if !header.is_empty() => &header,
                    None => return None,
                };
                let arguments = value.get("args")?.to_string();
                let index = self.next_index;
                Some(ParsedToolCall {
                    index,
                    id: format!("call_{index}"),
                    name: name.into(),
                    arguments,
                })
            });
        match call {
            Some(call) => {
                self.next_index += 1;
                vec![ParseDelta::ToolCall(call)]
            }
            None => vec![ParseDelta::Content(payload)],
        }
    }
}

impl TokenParser for ResponseParser {
    fn push_token(&mut self, token_id: i32, text_delta: &str) -> Vec<ParseDelta> {
        self.push_token(token_id, text_delta)
    }
    fn finish(&mut self) -> Vec<ParseDelta> {
        self.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::tests::byte_tokenizer;
    use ParseDelta::*;

    #[test]
    fn decoder_flush_at_open_marker() {
        let tok = byte_tokenizer();
        let mut decoder = tok.stream_decoder();
        let mut parser = ResponseParser::new(tok.special.clone());
        let mut message = ParsedMessage::default();
        for id in [
            0xf0,
            tok.special.content_thinking,
            65,
            tok.special.end_message,
        ] {
            message.extend(parser.push_token(id, &decoder.push(id)));
        }
        message.extend(parser.finish());
        assert_eq!(message.content.as_deref(), Some("�"));
        assert_eq!(message.reasoning_content.as_deref(), Some("A"));
    }

    #[test]
    fn accumulate_repeated_fields() {
        let s = byte_tokenizer().special;
        let tokens = [
            (s.content_thinking, ""),
            (1, "a"),
            (s.end_message, ""),
            (s.content_text, ""),
            (2, "你"),
            (s.end_message, ""),
            (s.content_thinking, ""),
            (3, "b"),
            (s.eos_token_id, ""),
            (s.content_tool_error, ""),
            (4, "!"),
            (s.end_message, ""),
        ];
        assert_eq!(
            parse_stream_to_message(&mut ResponseParser::new(s), tokens),
            ParsedMessage {
                reasoning_content: Some("ab".into()),
                content: Some("你!".into()),
                tool_calls: Vec::new()
            }
        );
    }

    #[test]
    fn streaming_and_aggregate() {
        let s = byte_tokenizer().special;
        let thought = [
            (s.content_thinking, ""),
            (1, "考"),
            (2, "虑"),
            (s.end_message, ""),
        ];
        let text = [
            (s.message_model, ""),
            (4, "helper"),
            (s.content_text, ""),
            (1, "a"),
            (2, "b"),
            (s.eos_token_id, ""),
        ];
        let first = [
            (s.message_model, ""),
            (1, "lookup"),
            (s.content_invoke_tool_json, ""),
            (2, r#"{"args": "#),
            (3, r#"{"b": "查"}}"#),
            (s.end_message, ""),
        ];
        let second = [
            (s.content_invoke_tool_json, ""),
            (1, r#"{"name":"ping","args":[]}"#),
            (s.eos_token_id, ""),
        ];
        let reasoning = vec![Reasoning("考".into()), Reasoning("虑".into())];
        let content = vec![Content("a".into()), Content("b".into())];
        let lookup = ToolCall(ParsedToolCall {
            index: 0,
            id: "call_0".into(),
            name: "lookup".into(),
            arguments: r#"{"b":"查"}"#.into(),
        });
        let ping = ToolCall(ParsedToolCall {
            index: 1,
            id: "call_1".into(),
            name: "ping".into(),
            arguments: "[]".into(),
        });
        let cases = [
            ("thinking", thought.to_vec(), reasoning.clone()),
            ("text", text.to_vec(), content.clone()),
            (
                "thinking_text",
                [&thought[..], &text].concat(),
                [&reasoning[..], &content].concat(),
            ),
            ("single", first.to_vec(), vec![lookup.clone()]),
            (
                "parallel",
                [&first[..], &second].concat(),
                vec![lookup.clone(), ping.clone()],
            ),
            (
                "mixed",
                [&thought[..], &text, &first, &second].concat(),
                [reasoning, content, vec![lookup.clone(), ping]].concat(),
            ),
            (
                "tool_error",
                vec![(s.content_tool_error, ""), (1, "oops")],
                vec![Content("oops".into())],
            ),
            (
                "tool_text",
                vec![(s.content_invoke_tool_text, ""), (1, "call")],
                vec![Content("call".into())],
            ),
            ("headerless", vec![(1, "hi")], vec![Content("hi".into())]),
            (
                "literal",
                vec![(s.content_text, ""), (1, "<|content_thinking|>")],
                vec![Content("<|content_thinking|>".into())],
            ),
            (
                "malformed",
                vec![
                    (s.content_invoke_tool_json, ""),
                    (1, "bad"),
                    (s.end_message, ""),
                ],
                vec![Content("bad".into())],
            ),
            (
                "truncated",
                vec![(s.content_invoke_tool_json, ""), (1, "{")],
                vec![Content("{".into())],
            ),
            (
                "finish_call",
                first[..first.len() - 1].to_vec(),
                vec![lookup],
            ),
        ];
        for (name, tokens, expected) in cases {
            let mut parser = ResponseParser::new(s.clone());
            let mut actual = Vec::new();
            for &(id, text) in &tokens {
                let buffered =
                    matches!(parser.state, State::Header(_) | State::ToolCallJson { .. });
                let deltas = parser.push_token(id, text);
                if !s.is_special(id) && !text.is_empty() {
                    assert_eq!(deltas.len(), usize::from(!buffered), "{name}");
                }
                actual.extend(deltas);
            }
            actual.extend(parser.finish());
            assert_eq!(actual, expected, "{name}");
            assert!(parser.finish().is_empty());
            let mut message = ParsedMessage::default();
            message.extend(expected);
            assert_eq!(
                parse_stream_to_message(&mut ResponseParser::new(s.clone()), tokens),
                message,
                "{name}"
            );
        }
    }
}
