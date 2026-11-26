use std::{
    collections::{HashMap, HashSet},
    fs,
    path::Path,
    str,
};
use tokenizers::Tokenizer as Backend;

pub type TokenizerError = crate::BoxError;

#[derive(Clone, Debug)]
pub struct SpecialTokens {
    pub eos_token_id: i32,
    pub pad_token_id: i32,
    pub message_user: i32,
    pub message_model: i32,
    pub message_system: i32,
    pub message_tool: i32,
    pub content_text: i32,
    pub content_thinking: i32,
    pub content_tool_error: i32,
    pub content_xml: i32,
    pub content_invoke_tool_json: i32,
    pub content_invoke_tool_text: i32,
    pub end_message: i32,
    pub begin_of_text: i32,
    all: HashSet<i32>,
}

impl SpecialTokens {
    fn load(inner: &Backend) -> Result<Self, TokenizerError> {
        let all: HashSet<_> = inner
            .get_added_tokens_decoder()
            .into_iter()
            .filter(|(_, token)| token.special)
            .map(|(id, _)| id as i32)
            .collect();
        let id = |name: &str| -> Result<i32, TokenizerError> {
            inner
                .token_to_id(&format!("<|{name}|>"))
                .map(|id| id as i32)
                .filter(|id| all.contains(id))
                .ok_or_else(|| format!("missing special token {name}").into())
        };
        Ok(Self {
            eos_token_id: id("content_model_end_sampling")?,
            pad_token_id: id("endoftext")?,
            message_user: id("message_user")?,
            message_model: id("message_model")?,
            message_system: id("message_system")?,
            message_tool: id("message_tool")?,
            content_text: id("content_text")?,
            content_thinking: id("content_thinking")?,
            content_tool_error: id("content_tool_error")?,
            content_xml: id("content_xml")?,
            content_invoke_tool_json: id("content_invoke_tool_json")?,
            content_invoke_tool_text: id("content_invoke_tool_text")?,
            end_message: id("end_message")?,
            begin_of_text: id("begin_of_text")?,
            all,
        })
    }
    pub fn is_special(&self, id: i32) -> bool {
        self.all.contains(&id)
    }
}

pub struct Tokenizer {
    inner: Backend,
    pub special: SpecialTokens,
    bytes: HashMap<char, u8>,
}

fn byte_alphabet() -> HashMap<char, u8> {
    let mut next = 256;
    (0..=u8::MAX)
        .map(|b| {
            let c = match b {
                33..=126 | 161..=172 | 174..=255 => char::from(b),
                _ => {
                    next += 1;
                    char::from_u32(next - 1).unwrap()
                }
            };
            (c, b)
        })
        .collect()
}

impl Tokenizer {
    pub fn from_model_dir(dir: &Path) -> Result<Self, TokenizerError> {
        Self::from_bytes(&fs::read(dir.join("tokenizer.json"))?)
    }
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, TokenizerError> {
        let inner = Backend::from_bytes(bytes)?;
        Ok(Self {
            special: SpecialTokens::load(&inner)?,
            inner,
            bytes: byte_alphabet(),
        })
    }
    pub fn encode(&self, text: &str) -> Result<Vec<i32>, TokenizerError> {
        Ok(self
            .inner
            .encode(text, false)?
            .get_ids()
            .iter()
            .map(|&id| id as i32)
            .collect())
    }
    pub fn decode(&self, ids: &[i32], skip_special_tokens: bool) -> Result<String, TokenizerError> {
        let ids: Result<Vec<_>, _> = ids.iter().copied().map(u32::try_from).collect();
        self.inner.decode(&ids?, skip_special_tokens)
    }
    pub fn stream_decoder(&self) -> StreamDecoder<'_> {
        StreamDecoder {
            tokenizer: self,
            pending_bytes: Vec::new(),
        }
    }
}

pub struct StreamDecoder<'a> {
    tokenizer: &'a Tokenizer,
    pending_bytes: Vec<u8>,
}

impl StreamDecoder<'_> {
    /// For a special ID the only returned text is an incomplete preceding character, flushed lossily.
    pub fn push(&mut self, token_id: i32) -> String {
        if self.tokenizer.special.is_special(token_id) {
            return self.finish();
        }
        let Some(piece) = self.tokenizer.inner.id_to_token(token_id as u32) else {
            return String::new();
        };
        for c in piece.chars() {
            if let Some(b) = self.tokenizer.bytes.get(&c) {
                self.pending_bytes.push(*b);
            } else {
                self.pending_bytes
                    .extend_from_slice(c.encode_utf8(&mut [0; 4]).as_bytes());
            }
        }
        let mut bytes = self.pending_bytes.as_slice();
        let mut out = String::new();
        loop {
            match str::from_utf8(bytes) {
                Ok(text) => {
                    out.push_str(text);
                    bytes = &[];
                    break;
                }
                Err(e) => {
                    let (valid, rest) = bytes.split_at(e.valid_up_to());
                    out.push_str(str::from_utf8(valid).unwrap());
                    bytes = rest;
                    let Some(len) = e.error_len() else { break };
                    out.push(char::REPLACEMENT_CHARACTER);
                    bytes = &bytes[len..];
                }
            }
        }
        let consumed = self.pending_bytes.len() - bytes.len();
        self.pending_bytes.drain(..consumed);
        out
    }

    pub fn finish(&mut self) -> String {
        let text = String::from_utf8_lossy(&self.pending_bytes).into_owned();
        self.reset();
        text
    }
    pub fn reset(&mut self) {
        self.pending_bytes.clear();
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use serde_json::{Map, Value, json};

    pub(crate) fn byte_tokenizer() -> Tokenizer {
        let mut vocab: Map<String, Value> = byte_alphabet()
            .into_iter()
            .map(|(c, b)| (c.to_string(), json!(b)))
            .collect();
        let mut added = Vec::new();
        for name in [
            "endoftext",
            "content_model_end_sampling",
            "message_user",
            "message_model",
            "message_system",
            "message_tool",
            "content_text",
            "content_thinking",
            "content_tool_error",
            "content_xml",
            "content_invoke_tool_json",
            "content_invoke_tool_text",
            "end_message",
            "begin_of_text",
            "unused",
        ] {
            let id = vocab.len();
            let content = format!("<|{name}|>");
            vocab.insert(content.clone(), json!(id));
            added.push(json!({"id":id,"content":content,"special":true,"normalized":false,"single_word":false,"lstrip":false,"rstrip":false}));
        }
        let byte_level = json!({"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false});
        let file = json!({"version":"1.0","added_tokens":added,"pre_tokenizer":byte_level,"decoder":byte_level,
            "model":{"type":"BPE","vocab":vocab,"merges":[]}});
        Tokenizer::from_bytes(&serde_json::to_vec(&file).unwrap()).unwrap()
    }

    #[test]
    fn split_utf8_and_specials() {
        let tok = byte_tokenizer();
        let text = " 你好🚀!\n";
        let ids = tok.encode(text).unwrap();
        assert_eq!(ids, text.bytes().map(i32::from).collect::<Vec<_>>());
        assert_eq!(tok.decode(&ids, true).unwrap(), text);
        let mut d = tok.stream_decoder();
        for c in text.chars() {
            let bytes = c.encode_utf8(&mut [0; 4]).as_bytes().to_vec();
            for &b in &bytes[..bytes.len() - 1] {
                assert_eq!(d.push(b.into()), "");
            }
            assert_eq!(d.push((*bytes.last().unwrap()).into()), c.to_string());
        }
        assert_eq!(d.finish(), "");
        let ids = tok.encode("<|unused|>!").unwrap();
        assert!(tok.special.is_special(ids[0]));
        assert_eq!(tok.decode(&ids, false).unwrap(), "<|unused|>!");
        assert_eq!(tok.decode(&ids, true).unwrap(), "!");
        assert_eq!(d.push(ids[0]), "");
        assert_eq!(d.push(0xf0), "");
        assert_eq!(d.push(ids[0]), "�");
        assert_eq!(d.push(0xf0), "");
        d.reset();
        assert_eq!(d.finish(), "");
        assert_eq!(d.push(0x80), "�");
        assert_eq!(d.push(0xe4), "");
        assert_eq!(d.push(65), "�A");
        assert_eq!(d.push(0xf0), "");
        assert_eq!(d.finish(), "�");
        assert_eq!(d.finish(), "");
    }
}
