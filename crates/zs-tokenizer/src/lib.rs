// Copyright (c) 2026 zstep Contributors
// SPDX-License-Identifier: MIT

//! HuggingFace tokenizer, DeepSeek-V4.1 / ChatML chat template renderer, and incremental
//! streaming detokenizer for the `zstep` online serving path.
//!
//! This crate is independent from the SHM data plane: it turns OpenAI-style chat messages
//! into prompt token IDs and turns generated token IDs back into UTF-8 chunks with
//! cross-chunk `stop` string clipping.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

pub mod chat;
pub mod stream;

pub use chat::{
    ChatMessage, ChatTemplateFormat, ChatTemplateOptions, MessageContent, ThinkingMode,
};
pub use stream::{FinishReason, StreamDecoder, StreamDelta};

/// Errors surfaced by the tokenizer layer. The HTTP layer maps
/// [`TokenizerError::ContextLengthExceeded`] to `400 context_length_exceeded` and
/// [`TokenizerError::Unsupported`] / [`TokenizerError::Invalid`] to `400 invalid_parameter`.
#[derive(Debug, thiserror::Error)]
pub enum TokenizerError {
    #[error("model directory {path} does not contain tokenizer.json")]
    MissingTokenizerJson { path: PathBuf },
    #[error("failed to load tokenizer.json: {0}")]
    Load(String),
    #[error("failed to tokenize prompt: {0}")]
    Encode(String),
    #[error(
        "requested {requested_total} tokens ({prompt_tokens} prompt + {max_new_tokens} completion) exceeds max_model_len {max_model_len}"
    )]
    ContextLengthExceeded {
        prompt_tokens: usize,
        max_new_tokens: usize,
        requested_total: usize,
        max_model_len: usize,
    },
    #[error("unsupported request feature: {0}")]
    Unsupported(String),
    #[error("invalid request: {0}")]
    Invalid(String),
}

pub type Result<T> = std::result::Result<T, TokenizerError>;

/// Canonical DeepSeek-V4.1 control tokens (must match `encoding/encoding.py`).
pub const BOS_TOKEN: &str = "<｜begin▁of▁sentence｜>";
pub const EOS_TOKEN: &str = "<｜end▁of▁sentence｜>";
pub const SYSTEM_TOKEN: &str = "<｜System｜>";
pub const USER_TOKEN: &str = "<｜User｜>";
pub const ASSISTANT_TOKEN: &str = "<｜Assistant｜>";
pub const LATEST_REMINDER_TOKEN: &str = "<｜latest_reminder｜>";
pub const THINK_START_TOKEN: &str = "<think>";
pub const THINK_END_TOKEN: &str = "</think>";

/// Rendered prompt string plus the encoded token IDs for one message list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodedChat {
    pub prompt: String,
    pub token_ids: Vec<u32>,
}

#[derive(Debug, Default, Deserialize)]
struct ModelConfigEosProbe {
    #[serde(default)]
    model_type: Option<String>,
    #[serde(default)]
    eos_token_id: Option<serde_json::Value>,
}

fn push_unique_id(dst: &mut Vec<u32>, id: u32) {
    if !dst.contains(&id) {
        dst.push(id);
    }
}

fn collect_eos_from_json_value(dst: &mut Vec<u32>, value: &serde_json::Value) {
    match value {
        serde_json::Value::Number(num) => {
            if let Some(raw) = num.as_u64()
                && let Ok(id) = u32::try_from(raw)
            {
                push_unique_id(dst, id);
            }
        }
        serde_json::Value::Array(items) => {
            for item in items {
                collect_eos_from_json_value(dst, item);
            }
        }
        _ => {}
    }
}

/// Loaded HuggingFace tokenizer plus model metadata and EOS token IDs.
#[derive(Clone)]
pub struct Tokenizer {
    inner: Arc<tokenizers::Tokenizer>,
    eos_ids: Vec<u32>,
    model_type: Option<String>,
    pub model_dir: PathBuf,
}

impl std::fmt::Debug for Tokenizer {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Tokenizer")
            .field("model_dir", &self.model_dir)
            .field("model_type", &self.model_type)
            .field("eos_ids", &self.eos_ids)
            .finish_non_exhaustive()
    }
}

impl Tokenizer {
    /// Load `<model_dir>/tokenizer.json` and inspect `config.json` / `generation_config.json`
    /// for `model_type` and `eos_token_id`.
    pub fn from_model_dir(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref().to_path_buf();
        let tokenizer_path = model_dir.join("tokenizer.json");
        if !tokenizer_path.is_file() {
            return Err(TokenizerError::MissingTokenizerJson {
                path: tokenizer_path,
            });
        }
        let inner = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|error| TokenizerError::Load(error.to_string()))?;

        let mut eos_ids = Vec::new();
        let mut model_type = None;

        for filename in ["config.json", "generation_config.json"] {
            let path = model_dir.join(filename);
            if let Ok(raw) = std::fs::read_to_string(&path)
                && let Ok(probe) = serde_json::from_str::<ModelConfigEosProbe>(&raw)
            {
                if model_type.is_none() {
                    model_type = probe.model_type.filter(|s| !s.is_empty());
                }
                if let Some(eos_val) = &probe.eos_token_id {
                    collect_eos_from_json_value(&mut eos_ids, eos_val);
                }
            }
        }

        for candidate in [EOS_TOKEN, "<|im_end|>", "<|endoftext|>", "</s>"] {
            if let Some(id) = inner.token_to_id(candidate) {
                push_unique_id(&mut eos_ids, id);
            }
        }

        Ok(Self {
            inner: Arc::new(inner),
            eos_ids,
            model_type,
            model_dir,
        })
    }

    pub fn eos_token_ids(&self) -> &[u32] {
        &self.eos_ids
    }

    pub fn is_eos(&self, token_id: u32) -> bool {
        self.eos_ids.contains(&token_id)
    }

    pub fn model_type(&self) -> Option<&str> {
        self.model_type.as_deref()
    }

    /// Infer the chat template format when `--tokenizer-mode auto` is active.
    pub fn default_template_format(&self) -> ChatTemplateFormat {
        if self.model_type.as_deref() == Some("deepseek_v41") {
            return ChatTemplateFormat::DeepSeekV41;
        }
        if self.model_type.is_none()
            && self.inner.token_to_id(BOS_TOKEN).is_some()
            && self.inner.token_to_id(USER_TOKEN).is_some()
        {
            return ChatTemplateFormat::DeepSeekV41;
        }
        ChatTemplateFormat::ChatMl
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(false)
    }

    /// Encode raw text (no special tokens added; the chat renderer emits them itself).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|error| TokenizerError::Encode(error.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode a token sequence with special tokens filtered out.
    pub fn decode(&self, token_ids: &[u32]) -> String {
        self.inner.decode(token_ids, true).unwrap_or_default()
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    /// Render the prompt string for `messages` (no tokenizer round-trip).
    pub fn render_chat(
        &self,
        messages: &[ChatMessage],
        options: &ChatTemplateOptions,
    ) -> Result<String> {
        chat::render_chat(messages, options)
    }

    /// Render + encode a chat request.
    pub fn encode_chat(
        &self,
        messages: &[ChatMessage],
        options: &ChatTemplateOptions,
    ) -> Result<EncodedChat> {
        let prompt = chat::render_chat(messages, options)?;
        let token_ids = self.encode(&prompt)?;
        Ok(EncodedChat { prompt, token_ids })
    }

    /// Enforce `prompt + max_new_tokens <= max_model_len` before admitting a request.
    pub fn validate_lengths(
        &self,
        prompt_tokens: usize,
        max_new_tokens: usize,
        max_model_len: usize,
    ) -> Result<()> {
        if prompt_tokens == 0 {
            return Err(TokenizerError::Invalid(
                "prompt must not be empty".to_owned(),
            ));
        }
        let requested_total = prompt_tokens.saturating_add(max_new_tokens);
        if requested_total > max_model_len {
            return Err(TokenizerError::ContextLengthExceeded {
                prompt_tokens,
                max_new_tokens,
                requested_total,
                max_model_len,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/deepseek_v41_golden.json")
    }

    fn model_dir() -> Option<PathBuf> {
        std::env::var_os("ZSTEP_TOKENIZER_MODEL_DIR").map(PathBuf::from)
    }

    #[test]
    fn renders_deepseek_prompt_frames_without_model() {
        let messages = vec![
            ChatMessage::system("You are a helpful assistant."),
            ChatMessage::user("Capital of France?"),
        ];
        let prompt = chat::render_chat(&messages, &ChatTemplateOptions::chat()).expect("render");
        assert_eq!(
            prompt,
            "<｜begin▁of▁sentence｜><｜System｜>You are a helpful assistant.<｜User｜>Capital of France?<｜Assistant｜></think>"
        );
    }

    #[test]
    fn renders_chatml_prompt_frames_for_qwen35() {
        let messages = vec![
            ChatMessage::system("You are helpful."),
            ChatMessage::user("Hello!"),
        ];
        let options = ChatTemplateOptions::for_format(ChatTemplateFormat::ChatMl);
        let prompt = chat::render_chat(&messages, &options).expect("render chatml");
        assert_eq!(
            prompt,
            "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn loads_eos_ids_and_model_type_from_config_json() {
        let dir = std::env::temp_dir().join(format!("zstep_tok_cfg_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create temp dir");
        std::fs::write(
            dir.join("tokenizer.json"),
            r#"{
              "version": "1.0",
              "truncation": null,
              "padding": null,
              "added_tokens": [
                {"id": 151644, "content": "<|im_start|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                {"id": 151645, "content": "<|im_end|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
              ],
              "normalizer": null,
              "pre_tokenizer": {"type": "Whitespace"},
              "post_processor": null,
              "decoder": null,
              "model": {
                "type": "WordLevel",
                "vocab": {"[UNK]": 0, "hello": 1, "<|im_start|>": 151644, "<|im_end|>": 151645},
                "unk_token": "[UNK]"
              }
            }"#,
        )
        .expect("write tokenizer.json");
        std::fs::write(
            dir.join("config.json"),
            r#"{"model_type": "qwen3_5_moe", "eos_token_id": [151645, 151643]}"#,
        )
        .expect("write config.json");

        let tok = Tokenizer::from_model_dir(&dir).expect("load");
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(tok.model_type(), Some("qwen3_5_moe"));
        assert_eq!(tok.default_template_format(), ChatTemplateFormat::ChatMl);
        assert!(tok.is_eos(151645));
        assert!(tok.is_eos(151643));
    }

    #[test]
    fn golden_prompt_strings_match_reference_fixture() {
        let raw = std::fs::read_to_string(fixture_path()).expect("golden fixture");
        let cases: Vec<serde_json::Value> = serde_json::from_str(&raw).expect("valid json");
        assert!(cases.len() >= 4);
        for case in cases {
            let messages: Vec<ChatMessage> =
                serde_json::from_value(case["messages"].clone()).expect("messages");
            let mode = match case["thinking_mode"].as_str().expect("mode") {
                "thinking" => ThinkingMode::Thinking,
                _ => ThinkingMode::Chat,
            };
            let options = ChatTemplateOptions {
                thinking_mode: mode,
                reasoning_effort: case["reasoning_effort"].as_str().map(str::to_owned),
                ..ChatTemplateOptions::default()
            };
            let prompt = chat::render_chat(&messages, &options).expect("render");
            assert_eq!(
                prompt,
                case["prompt"].as_str().expect("prompt"),
                "case {} prompt mismatch",
                case["name"]
            );
        }
    }

    /// Token-ID parity against the HF reference. Requires the model directory:
    /// `ZSTEP_TOKENIZER_MODEL_DIR=/models/DeepSeek-V4.1-Flash cargo test -p zs-tokenizer`.
    #[test]
    fn golden_token_ids_match_reference_fixture() {
        let Some(dir) = model_dir() else {
            return;
        };
        let tokenizer = Tokenizer::from_model_dir(&dir).expect("load tokenizer");
        let raw = std::fs::read_to_string(fixture_path()).expect("golden fixture");
        let cases: Vec<serde_json::Value> = serde_json::from_str(&raw).expect("valid json");
        for case in cases {
            let messages: Vec<ChatMessage> =
                serde_json::from_value(case["messages"].clone()).expect("messages");
            let mode = match case["thinking_mode"].as_str().expect("mode") {
                "thinking" => ThinkingMode::Thinking,
                _ => ThinkingMode::Chat,
            };
            let options = ChatTemplateOptions {
                thinking_mode: mode,
                reasoning_effort: case["reasoning_effort"].as_str().map(str::to_owned),
                ..ChatTemplateOptions::default()
            };
            let encoded = tokenizer.encode_chat(&messages, &options).expect("encode");
            let expected: Vec<u32> = case["ids"]
                .as_array()
                .expect("ids")
                .iter()
                .map(|value| value.as_u64().expect("id") as u32)
                .collect();
            assert_eq!(
                encoded.token_ids, expected,
                "case {} token IDs mismatch",
                case["name"]
            );
        }
        assert!(tokenizer.is_eos(1), "EOS token id must be 1");
    }
}
