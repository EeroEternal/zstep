// Copyright (c) 2026 zstep Contributors
// SPDX-License-Identifier: MIT

//! Chat prompt rendering for DeepSeek-V4.1 (`encoding/encoding.py::encode_messages`)
//! and generic ChatML (`Qwen3.5-35B-A3B` / `Qwen2.5`).
//!
//! Tool calling, images, task tokens, and response-format schemas are not implemented
//! in P0 (`static_capability_zstep` reports `tool_calling = Unsupported`) and return
//! [`TokenizerError::Unsupported`] rather than silently degrading.

use serde::{Deserialize, Serialize};

use crate::{
    ASSISTANT_TOKEN, BOS_TOKEN, EOS_TOKEN, LATEST_REMINDER_TOKEN, Result, SYSTEM_TOKEN,
    THINK_END_TOKEN, THINK_START_TOKEN, TokenizerError, USER_TOKEN,
};

/// Prompt template dialect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChatTemplateFormat {
    #[default]
    DeepSeekV41,
    ChatMl,
}

/// Thinking mode (`chat` renders a closed `</think>` on DeepSeek-V4.1, `thinking` opens `<think>`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ThinkingMode {
    Chat,
    Thinking,
}

/// Content part inside an OpenAI array-valued `content` field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentPart {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub text: Option<String>,
}

/// OpenAI message `content` (`string` or `[{type: "text", text: "..."}]`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
    UnsupportedValue(serde_json::Value),
}

impl MessageContent {
    pub fn resolve_text(&self) -> Result<String> {
        match self {
            Self::Text(text) => Ok(text.clone()),
            Self::Parts(parts) => {
                let mut joined = String::new();
                for part in parts {
                    if part.kind != "text" {
                        return Err(TokenizerError::Unsupported(format!(
                            "multimodal content part type `{}` is not supported",
                            part.kind
                        )));
                    }
                    let Some(text) = &part.text else {
                        return Err(TokenizerError::Invalid(
                            "content part of type `text` is missing `text` field".to_owned(),
                        ));
                    };
                    joined.push_str(text);
                }
                Ok(joined)
            }
            Self::UnsupportedValue(_) => Err(TokenizerError::Invalid(
                "message `content` must be a string or an array of text parts".to_owned(),
            )),
        }
    }
}

/// One OpenAI chat message (text-only subset).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<MessageContent>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub function_call: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_blocks: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio: Option<serde_json::Value>,
    #[serde(default)]
    pub wo_eos: bool,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_owned(),
            content: Some(MessageContent::Text(content.into())),
            ..Self::default()
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_owned(),
            content: Some(MessageContent::Text(content.into())),
            ..Self::default()
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_owned(),
            content: Some(MessageContent::Text(content.into())),
            ..Self::default()
        }
    }

    fn is_assistant_or_reminder(&self) -> bool {
        matches!(self.role.as_str(), "assistant" | "latest_reminder")
    }

    fn resolved_content(&self) -> Result<String> {
        match &self.content {
            None => Ok(String::new()),
            Some(content) => content.resolve_text(),
        }
    }

    fn validate_supported(&self) -> Result<()> {
        if self.tools.is_some() {
            return Err(TokenizerError::Unsupported(
                "tools are not supported by zstep online serving (P0)".to_owned(),
            ));
        }
        if self.tool_calls.is_some() || self.tool_call_id.is_some() || self.function_call.is_some()
        {
            return Err(TokenizerError::Unsupported(
                "tool_calls rendering is not supported by zstep online serving (P0)".to_owned(),
            ));
        }
        if self.response_format.is_some() {
            return Err(TokenizerError::Unsupported(
                "response_format is not supported by zstep online serving (P0)".to_owned(),
            ));
        }
        if self.task.is_some() {
            return Err(TokenizerError::Unsupported(
                "internal task messages are not supported".to_owned(),
            ));
        }
        if self.content_blocks.is_some() || self.audio.is_some() {
            return Err(TokenizerError::Unsupported(
                "multimodal content blocks are not supported".to_owned(),
            ));
        }
        let _ = self.resolved_content()?;
        Ok(())
    }
}

/// Rendering options mirroring `encode_messages(...)` parameters plus dialect selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatTemplateOptions {
    #[serde(default)]
    pub format: ChatTemplateFormat,
    pub thinking_mode: ThinkingMode,
    /// `"low" | "high" | "max"` or a numeric string in `1..=100`.
    pub reasoning_effort: Option<String>,
    pub drop_thinking: bool,
    pub add_bos: bool,
}

impl Default for ChatTemplateOptions {
    fn default() -> Self {
        Self {
            format: ChatTemplateFormat::DeepSeekV41,
            thinking_mode: ThinkingMode::Chat,
            reasoning_effort: None,
            drop_thinking: true,
            add_bos: true,
        }
    }
}

impl ChatTemplateOptions {
    pub fn chat() -> Self {
        Self::default()
    }

    pub fn for_format(format: ChatTemplateFormat) -> Self {
        Self {
            format,
            ..Self::default()
        }
    }

    pub fn thinking(effort: Option<&str>) -> Self {
        Self {
            thinking_mode: ThinkingMode::Thinking,
            reasoning_effort: effort.map(str::to_owned),
            ..Self::default()
        }
    }
}

fn reasoning_budget(effort: Option<&str>) -> Result<u32> {
    let Some(raw) = effort else {
        return Ok(75); // DEFAULT_REASONING_EFFORT = "high"
    };
    match raw.trim().to_ascii_lowercase().as_str() {
        "low" => Ok(50),
        "high" => Ok(75),
        "max" => Ok(100),
        other => {
            let budget: u32 = other.parse().map_err(|_| {
                TokenizerError::Invalid(format!(
                    "invalid reasoning_effort `{raw}`: expected low/high/max or an integer in 1..=100"
                ))
            })?;
            if (1..=100).contains(&budget) {
                Ok(budget)
            } else {
                Err(TokenizerError::Invalid(format!(
                    "reasoning_effort {budget} is outside 1..=100"
                )))
            }
        }
    }
}

fn render_reasoning_effort(
    index: usize,
    mode: ThinkingMode,
    effort: Option<&str>,
) -> Result<String> {
    if index != 0 || mode != ThinkingMode::Thinking {
        return Ok(String::new());
    }
    let budget = reasoning_budget(effort)?;
    Ok(format!(
        "Reasoning Effort: {budget} (range 1-100, the higher the value, the more thorough the reasoning)\n\n"
    ))
}

fn find_last_user_index(messages: &[ChatMessage]) -> Option<usize> {
    messages
        .iter()
        .enumerate()
        .rev()
        .find(|(index, message)| message.role == "user" || (message.role == "system" && *index > 0))
        .map(|(index, _)| index)
}

/// Drop reasoning content / assistant history before the last user turn
/// (`_drop_thinking_messages` in the reference implementation).
fn drop_thinking_messages(messages: &[ChatMessage]) -> Vec<ChatMessage> {
    let last_user_index = find_last_user_index(messages);
    messages
        .iter()
        .enumerate()
        .filter_map(|(index, message)| {
            let keep_role = matches!(
                message.role.as_str(),
                "user" | "system" | "latest_reminder" | "direct_search_results"
            );
            let after_last_user = last_user_index.is_some_and(|last| index >= last);
            if keep_role || after_last_user {
                Some(message.clone())
            } else if message.role == "assistant" {
                let mut clone = message.clone();
                clone.reasoning_content = None;
                Some(clone)
            } else {
                None
            }
        })
        .collect()
}

fn render_deepseek_message(
    index: usize,
    messages: &[ChatMessage],
    options: &ChatTemplateOptions,
) -> Result<String> {
    let message = &messages[index];
    let last_user_index = find_last_user_index(messages);
    let thinking = options.thinking_mode == ThinkingMode::Thinking;
    let effort_prompt = render_reasoning_effort(
        index,
        options.thinking_mode,
        options.reasoning_effort.as_deref(),
    )?;

    let mut prompt = String::new();
    if index == 0 && (!effort_prompt.is_empty() || message.role == "system") {
        prompt.push_str(SYSTEM_TOKEN);
    }
    prompt.push_str(&effort_prompt);

    let content = message.resolved_content()?;
    match message.role.as_str() {
        "system" => {
            if index > 0 {
                prompt.push_str(SYSTEM_TOKEN);
            }
            prompt.push_str(&content);
        }
        "user" => {
            prompt.push_str(USER_TOKEN);
            prompt.push_str(&content);
        }
        "latest_reminder" => {
            prompt.push_str(LATEST_REMINDER_TOKEN);
            prompt.push_str(&content);
        }
        "tool" => {
            return Err(TokenizerError::Unsupported(
                "role=tool messages must be merged into user content before encoding".to_owned(),
            ));
        }
        "assistant" => {
            let thinking_part = if thinking && (last_user_index.is_some_and(|last| index > last)) {
                format!(
                    "{}{}",
                    message.reasoning_content.as_deref().unwrap_or(""),
                    THINK_END_TOKEN
                )
            } else {
                String::new()
            };
            prompt.push_str(&thinking_part);
            prompt.push_str(&content);
            if !message.wo_eos {
                prompt.push_str(EOS_TOKEN);
            }
        }
        other => {
            return Err(TokenizerError::Invalid(format!("unknown role `{other}`")));
        }
    }

    if index + 1 < messages.len() && !messages[index + 1].is_assistant_or_reminder() {
        return Ok(prompt);
    }

    if message.role == "user" || (message.role == "system" && index > 0) {
        prompt.push_str(ASSISTANT_TOKEN);
        let last_user = last_user_index.unwrap_or(usize::MAX);
        if thinking && (!options.drop_thinking || index >= last_user) {
            prompt.push_str(THINK_START_TOKEN);
        } else {
            prompt.push_str(THINK_END_TOKEN);
        }
    }
    Ok(prompt)
}

fn render_chatml(messages: &[ChatMessage], options: &ChatTemplateOptions) -> Result<String> {
    if let Some(effort) = options.reasoning_effort.as_deref() {
        let _ = reasoning_budget(Some(effort))?;
    }
    let mut prompt = String::new();
    for (index, message) in messages.iter().enumerate() {
        let content = message.resolved_content()?;
        match message.role.as_str() {
            "system" | "user" => {
                prompt.push_str("<|im_start|>");
                prompt.push_str(&message.role);
                prompt.push('\n');
                prompt.push_str(&content);
                prompt.push_str("<|im_end|>\n");
            }
            "assistant" => {
                prompt.push_str("<|im_start|>assistant\n");
                if let Some(reasoning) = message.reasoning_content.as_deref()
                    && !reasoning.is_empty()
                {
                    prompt.push_str("<think>\n");
                    prompt.push_str(reasoning);
                    prompt.push_str("\n</think>\n\n");
                }
                prompt.push_str(&content);
                let is_last = index + 1 == messages.len();
                if !(is_last && message.wo_eos) {
                    prompt.push_str("<|im_end|>\n");
                }
            }
            "tool" => {
                return Err(TokenizerError::Unsupported(
                    "role=tool messages are not supported by zstep online serving (P0)".to_owned(),
                ));
            }
            other => {
                return Err(TokenizerError::Invalid(format!("unknown role `{other}`")));
            }
        }
    }
    let needs_generation_prompt = messages
        .last()
        .is_some_and(|last| !(last.role == "assistant" && last.wo_eos));
    if needs_generation_prompt {
        prompt.push_str("<|im_start|>assistant\n");
        if options.thinking_mode == ThinkingMode::Thinking {
            prompt.push_str("<think>\n");
        }
    }
    Ok(prompt)
}

/// Render a prompt string from OpenAI-style messages according to `options.format`.
pub fn render_chat(messages: &[ChatMessage], options: &ChatTemplateOptions) -> Result<String> {
    if messages.is_empty() {
        return Err(TokenizerError::Invalid(
            "messages must contain at least one entry".to_owned(),
        ));
    }
    for message in messages {
        message.validate_supported()?;
    }

    if options.format == ChatTemplateFormat::ChatMl {
        return render_chatml(messages, options);
    }

    let effective = if options.thinking_mode == ThinkingMode::Thinking && options.drop_thinking {
        drop_thinking_messages(messages)
    } else {
        messages.to_vec()
    };
    if effective.is_empty() {
        return Err(TokenizerError::Invalid(
            "no messages left to render after dropping thinking history".to_owned(),
        ));
    }

    let mut prompt = String::new();
    if options.add_bos {
        prompt.push_str(BOS_TOKEN);
    }
    for index in 0..effective.len() {
        prompt.push_str(&render_deepseek_message(index, &effective, options)?);
    }
    Ok(prompt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_mode_appends_closed_thinking_tag() {
        let messages = vec![ChatMessage::user("hi")];
        let prompt = render_chat(&messages, &ChatTemplateOptions::chat()).expect("render");
        assert_eq!(
            prompt,
            "<｜begin▁of▁sentence｜><｜User｜>hi<｜Assistant｜></think>"
        );
    }

    #[test]
    fn thinking_mode_appends_reasoning_effort_and_open_tag() {
        let messages = vec![ChatMessage::user("1+1=?")];
        let prompt =
            render_chat(&messages, &ChatTemplateOptions::thinking(Some("low"))).expect("render");
        assert!(prompt.starts_with("<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 50 "));
        assert!(prompt.ends_with("<｜Assistant｜><think>"));
    }

    #[test]
    fn mid_conversation_system_gets_own_frame() {
        let messages = vec![
            ChatMessage::user("a"),
            ChatMessage::assistant("b"),
            ChatMessage::system("late"),
            ChatMessage::user("c"),
        ];
        let prompt = render_chat(&messages, &ChatTemplateOptions::chat()).expect("render");
        assert!(prompt.contains("<｜end▁of▁sentence｜><｜System｜>late<｜User｜>c"));
    }

    #[test]
    fn tools_and_multimodal_content_parts_are_rejected_while_text_parts_work() {
        let mut message = ChatMessage::user("hi");
        message.tools = Some(serde_json::json!([]));
        assert!(matches!(
            render_chat(&[message], &ChatTemplateOptions::chat()),
            Err(TokenizerError::Unsupported(_))
        ));

        let img_msg: ChatMessage = serde_json::from_value(serde_json::json!({
            "role": "user",
            "content": [
                {"type": "text", "text": "describe this"},
                {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}
            ]
        }))
        .expect("deserialize multimodal array");
        assert!(matches!(
            render_chat(&[img_msg], &ChatTemplateOptions::chat()),
            Err(TokenizerError::Unsupported(_))
        ));

        let text_parts_msg: ChatMessage = serde_json::from_value(serde_json::json!({
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello "},
                {"type": "text", "text": "world"}
            ]
        }))
        .expect("deserialize text parts");
        let rendered =
            render_chat(&[text_parts_msg], &ChatTemplateOptions::chat()).expect("render text");
        assert!(rendered.contains("Hello world"));
    }

    #[test]
    fn invalid_reasoning_effort_is_rejected() {
        let messages = vec![ChatMessage::user("hi")];
        assert!(matches!(
            render_chat(&messages, &ChatTemplateOptions::thinking(Some("medium"))),
            Err(TokenizerError::Invalid(_))
        ));
        assert!(matches!(
            render_chat(&messages, &ChatTemplateOptions::thinking(Some("101"))),
            Err(TokenizerError::Invalid(_))
        ));
    }

    #[test]
    fn drop_thinking_follows_reference_semantics() {
        let mut history = ChatMessage::assistant("history-answer");
        history.reasoning_content = Some("history-thoughts".to_owned());
        let with_history = vec![ChatMessage::user("u1"), history, ChatMessage::user("u2")];
        let dropped =
            render_chat(&with_history, &ChatTemplateOptions::thinking(None)).expect("render");
        assert!(
            !dropped.contains("history-thoughts"),
            "history reasoning must be dropped"
        );
        assert!(dropped.ends_with("<｜Assistant｜><think>"));

        let mut current = ChatMessage::assistant("answer");
        current.reasoning_content = Some("current-thoughts".to_owned());
        let with_current = vec![ChatMessage::user("u1"), current];
        let mut options = ChatTemplateOptions::thinking(None);
        options.drop_thinking = false;
        let kept = render_chat(&with_current, &options).expect("render");
        assert!(
            kept.contains("current-thoughts</think>"),
            "current-turn reasoning must render"
        );
    }
}
