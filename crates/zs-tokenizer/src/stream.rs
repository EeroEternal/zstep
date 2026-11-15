// Copyright (c) 2026 zstep Contributors
// SPDX-License-Identifier: MIT

//! Incremental UTF-8 detokenizer with cross-chunk `stop` string clipping.
//!
//! Generated tokens are appended one at a time; the decoder only hands the caller
//! the *stable* suffix while holding back:
//! 1. the trailing `max(stop.len()) - 1` bytes so a `stop` string split across
//!    token boundaries is detected before any byte of it is emitted;
//! 2. all trailing `U+FFFD` replacement characters produced when byte-level BPE
//!    tokens carry an incomplete multi-byte UTF-8 sequence.
//!
//! To avoid $O(N^2)$ full-sequence decoding on long completions, [`StreamDecoder::new`]
//! compacts the decoded token window once a verified splice anchor is reached.

use std::sync::Arc;

use crate::Tokenizer;

/// Decoder function used to turn a token slice into text.
pub type DecodeFn = Arc<dyn Fn(&[u32]) -> String + Send + Sync>;

/// Why a stream terminated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// `max_tokens` reached.
    Length,
    /// Model EOS / stop token emitted.
    StopToken,
    /// A caller supplied `stop` string matched.
    StopString,
    /// Client disconnected or a drain timeout cancelled the request.
    Cancelled,
    /// Worker fault / engine error.
    Failed,
}

/// Text delta handed to one SSE chunk.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StreamDelta {
    pub text: String,
    pub finished: Option<FinishReason>,
}

impl StreamDelta {
    pub fn is_terminal(&self) -> bool {
        self.finished.is_some()
    }
}

const WINDOW_COMPACT_THRESHOLD: usize = 48;
const WINDOW_OVERLAP_TOKENS: usize = 8;

/// Streaming detokenizer state for one request.
pub struct StreamDecoder {
    stops: Vec<String>,
    /// Maximum number of bytes any stop string contributes to the holdback window.
    holdback: usize,
    token_ids: Vec<u32>,
    /// Start index in `token_ids` for incremental window decoding (`0` when uncompacted).
    window_token_start: usize,
    /// Prefix bytes of `text` committed before `window_overlap_text_len`.
    committed_prefix: String,
    /// Decoded length of `token_ids[window_token_start..window_anchor_end]` inside the window.
    window_anchor_bytes: usize,
    allow_window_compaction: bool,
    text: String,
    emitted: usize,
    produced: String,
    finished: Option<FinishReason>,
    decoder: DecodeFn,
}

impl std::fmt::Debug for StreamDecoder {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("StreamDecoder")
            .field("stops", &self.stops)
            .field("tokens", &self.token_ids.len())
            .field("emitted", &self.emitted)
            .field("finished", &self.finished)
            .finish_non_exhaustive()
    }
}

impl StreamDecoder {
    /// Decoder bound to a loaded [`Tokenizer`]. `stops` are raw caller strings.
    pub fn new(tokenizer: &Tokenizer, stops: Vec<String>) -> Self {
        let tokenizer = tokenizer.clone();
        let mut dec = Self::with_decoder(move |ids: &[u32]| tokenizer.decode(ids), stops);
        dec.allow_window_compaction = true;
        dec
    }

    /// Test/alternative decoder injection (no tokenizer required).
    pub fn with_decoder(
        decoder: impl Fn(&[u32]) -> String + Send + Sync + 'static,
        stops: Vec<String>,
    ) -> Self {
        let holdback = stops
            .iter()
            .map(|stop| stop.len())
            .max()
            .map_or(0, |longest| longest.saturating_sub(1));
        Self {
            stops,
            holdback,
            token_ids: Vec::new(),
            window_token_start: 0,
            committed_prefix: String::new(),
            window_anchor_bytes: 0,
            allow_window_compaction: false,
            text: String::new(),
            emitted: 0,
            produced: String::new(),
            finished: None,
            decoder: Arc::new(decoder),
        }
    }

    pub fn stops(&self) -> &[String] {
        &self.stops
    }

    pub fn token_count(&self) -> usize {
        self.token_ids.len()
    }

    pub fn emitted_text(&self) -> &str {
        &self.produced
    }

    pub fn is_finished(&self) -> bool {
        self.finished.is_some()
    }

    /// Append one generated token ID and return the newly stable text.
    pub fn push(&mut self, token_id: u32) -> StreamDelta {
        if self.finished.is_some() {
            return StreamDelta::default();
        }
        self.token_ids.push(token_id);
        self.refresh_decoded_text();
        let delta = self.drain(false);
        if self.allow_window_compaction && self.finished.is_none() {
            self.maybe_compact_window();
        }
        delta
    }

    /// Flush everything that is still held back and mark the stream terminal.
    pub fn finish(&mut self, reason: FinishReason) -> StreamDelta {
        if self.finished.is_some() {
            return StreamDelta::default();
        }
        let mut delta = self.drain(true);
        let delta_reason = delta.finished.take();
        self.finished = Some(delta_reason.unwrap_or(reason));
        delta.finished = self.finished;
        delta
    }

    fn refresh_decoded_text(&mut self) {
        let window_decoded = (self.decoder)(&self.token_ids[self.window_token_start..]);
        if self.window_token_start == 0 {
            self.text = window_decoded;
        } else if window_decoded.len() >= self.window_anchor_bytes {
            let mut next_text =
                String::with_capacity(self.committed_prefix.len() + window_decoded.len());
            next_text.push_str(&self.committed_prefix);
            next_text.push_str(&window_decoded[self.window_anchor_bytes..]);
            self.text = next_text;
        } else {
            // Fallback to full-prefix decode if a custom/non-monotonic decoder altered anchor bytes.
            self.window_token_start = 0;
            self.committed_prefix.clear();
            self.window_anchor_bytes = 0;
            self.text = (self.decoder)(&self.token_ids);
        }
    }

    fn maybe_compact_window(&mut self) {
        let window_len = self.token_ids.len() - self.window_token_start;
        if window_len < WINDOW_COMPACT_THRESHOLD {
            return;
        }
        let new_start = self.token_ids.len() - WINDOW_OVERLAP_TOKENS;
        if new_start <= self.window_token_start {
            return;
        }
        // Choose a single-token anchor at `new_start` and verify exact splice equivalence.
        let overlap_decoded = (self.decoder)(&self.token_ids[new_start..]);
        if overlap_decoded.contains('\u{FFFD}') || overlap_decoded.is_empty() {
            return;
        }
        let anchor_one = (self.decoder)(&self.token_ids[new_start..new_start + 1]);
        if anchor_one.is_empty() || !overlap_decoded.starts_with(&anchor_one) {
            return;
        }
        let tail_after_anchor = &overlap_decoded[anchor_one.len()..];
        if !self.text.ends_with(tail_after_anchor) {
            return;
        }
        let prefix_end = self.text.len() - tail_after_anchor.len();
        if prefix_end > self.emitted || !self.text.is_char_boundary(prefix_end) {
            return;
        }
        self.committed_prefix = self.text[..prefix_end].to_owned();
        self.window_token_start = new_start;
        self.window_anchor_bytes = anchor_one.len();
    }

    fn find_stop(&self, pending: &str) -> Option<(usize, usize)> {
        self.stops
            .iter()
            .filter(|stop| !stop.is_empty())
            .filter_map(|stop| pending.find(stop.as_str()).map(|pos| (pos, stop.len())))
            .min_by_key(|(pos, _)| *pos)
    }

    fn drain(&mut self, flush: bool) -> StreamDelta {
        let pending = self.text[self.emitted..].to_owned();
        if let Some((position, stop_len)) = self.find_stop(&pending) {
            let text = pending[..position].to_owned();
            self.emitted += position + stop_len;
            self.finished = Some(FinishReason::StopString);
            self.produced.push_str(&text);
            return StreamDelta {
                text,
                finished: Some(FinishReason::StopString),
            };
        }
        if flush {
            let text = pending;
            self.emitted = self.text.len();
            self.produced.push_str(&text);
            return StreamDelta {
                text,
                finished: None,
            };
        }
        let mut safe_end = self.text.len().saturating_sub(self.holdback);
        while safe_end > self.emitted && !self.text.is_char_boundary(safe_end) {
            safe_end -= 1;
        }
        let trimmed_len = self.text.trim_end_matches('\u{FFFD}').len();
        if trimmed_len < safe_end {
            safe_end = trimmed_len.max(self.emitted);
        }
        if safe_end <= self.emitted {
            return StreamDelta::default();
        }
        let text = self.text[self.emitted..safe_end].to_owned();
        self.emitted = safe_end;
        self.produced.push_str(&text);
        StreamDelta {
            text,
            finished: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table_decoder(stops: &[&str], expected: &'static [(usize, &'static str)]) -> StreamDecoder {
        StreamDecoder::with_decoder(
            move |ids| {
                let len = ids.len();
                expected
                    .iter()
                    .find(|(n, _)| *n == len)
                    .map(|(_, text)| (*text).to_owned())
                    .unwrap_or_default()
            },
            stops.iter().map(|s| (*s).to_owned()).collect(),
        )
    }

    #[test]
    fn clamps_stop_string_split_across_chunks() {
        let mut stream = table_decoder(
            &["\n\n"],
            &[(1, "Hello"), (2, "Hello\n\n"), (3, "Hello\n\nUser")],
        );
        assert_eq!(stream.push(0).text, "Hell");
        let delta = stream.push(1);
        assert_eq!(delta.text, "o");
        assert_eq!(delta.finished, Some(FinishReason::StopString));
        assert!(stream.is_finished());
        assert_eq!(stream.emitted_text(), "Hello");
        assert_eq!(stream.push(2), StreamDelta::default());
    }

    #[test]
    fn flushes_remaining_text_on_finish() {
        let mut stream = table_decoder(&["\n\n"], &[(1, "abc"), (2, "abc\n")]);
        assert_eq!(stream.push(0).text, "ab");
        assert_eq!(stream.push(1).text, "c");
        let done = stream.finish(FinishReason::Length);
        assert_eq!(done.text, "\n");
        assert_eq!(done.finished, Some(FinishReason::Length));
    }

    #[test]
    fn utf8_multi_byte_split_holds_back_multiple_replacement_chars() {
        // Simulates a 4-byte emoji split across 4 single-byte BPE tokens producing 1, 2, 3 U+FFFD
        // characters before healing on the 4th byte.
        let mut stream = table_decoder(
            &[],
            &[
                (1, "\u{FFFD}"),
                (2, "\u{FFFD}\u{FFFD}"),
                (3, "\u{FFFD}\u{FFFD}\u{FFFD}"),
                (4, "🦀"),
                (5, "🦀!"),
            ],
        );
        assert_eq!(stream.push(0).text, "");
        assert_eq!(stream.push(1).text, "");
        assert_eq!(stream.push(2).text, "");
        assert_eq!(stream.push(3).text, "🦀");
        assert_eq!(stream.push(4).text, "!");
        assert_eq!(stream.emitted_text(), "🦀!");
    }

    #[test]
    fn sliding_window_compaction_preserves_long_stream_output() {
        let mut dec = StreamDecoder::with_decoder(
            |ids| ids.iter().map(|id| format!("w{id} ")).collect::<String>(),
            vec!["STOP".to_owned()],
        );
        dec.allow_window_compaction = true;
        let mut out = String::new();
        for tok in 0..120u32 {
            out.push_str(&dec.push(tok).text);
        }
        out.push_str(&dec.finish(FinishReason::Length).text);
        let expected: String = (0..120u32).map(|id| format!("w{id} ")).collect();
        assert_eq!(out, expected);
        assert!(dec.window_token_start > 0, "window should have compacted");
    }
}
