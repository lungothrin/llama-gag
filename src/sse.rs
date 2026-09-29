//! Minimal Server-Sent Events (SSE) parser for OpenAI-compatible streams.
//!
//! The parser is byte-oriented and reassembles lines and multi-byte UTF-8
//! sequences that are split across network chunks. It emits complete events
//! (the joined `data:` payload) and recognises the `data: [DONE]` terminator.
//!
//! The proxy forwards the raw bytes to the client verbatim; this parser is used
//! only to extract `delta.content` for loop detection, so it must never alter
//! what is forwarded.

use serde_json::Value;

/// A parsed SSE event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SseEvent {
    /// The joined `data:` payload of one event (without the `data:` prefix).
    Data(String),
    /// The stream terminator (`data: [DONE]`).
    Done,
}

impl SseEvent {
    /// The `data:` payload as a string (`"[DONE]"` for the terminator).
    pub fn payload(&self) -> &str {
        match self {
            SseEvent::Data(s) => s,
            SseEvent::Done => "[DONE]",
        }
    }
}

/// Incremental SSE parser. Feed raw bytes as they arrive; complete events are
/// returned as they are delimited by blank lines.
#[derive(Debug, Default)]
pub struct SseParser {
    /// Bytes of the current, not-yet-terminated line.
    line_buf: Vec<u8>,
    /// Joined `data:` payloads of the event being accumulated.
    data_buf: String,
}

impl SseParser {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed a chunk of raw bytes; return any events completed by this chunk.
    pub fn feed(&mut self, chunk: &[u8]) -> Vec<SseEvent> {
        let mut out = Vec::new();
        // Prepend any partial line buffered from a previous chunk.
        let mut input: Vec<u8> = std::mem::take(&mut self.line_buf);
        input.extend_from_slice(chunk);

        let mut pos = 0;
        while let Some(rel) = input[pos..].iter().position(|&b| b == b'\n') {
            let line_end = pos + rel;
            let line = &input[pos..line_end];
            pos = line_end + 1;
            let line = if line.last() == Some(&b'\r') {
                &line[..line.len() - 1]
            } else {
                line
            };
            if let Some(event) = self.process_line(line) {
                out.push(event);
            }
        }
        // Buffer the trailing partial line (no terminating newline yet).
        self.line_buf = input[pos..].to_vec();
        out
    }

    /// Process one complete line; return an event if the line terminates one.
    fn process_line(&mut self, line: &[u8]) -> Option<SseEvent> {
        // Blank line: event boundary.
        if line.is_empty() {
            let event = if self.data_buf.is_empty() {
                return None;
            } else if self.data_buf == "[DONE]" {
                SseEvent::Done
            } else {
                SseEvent::Data(std::mem::take(&mut self.data_buf))
            };
            return Some(event);
        }

        // Comment line (starts with ':') is ignored.
        if line.first() == Some(&b':') {
            return None;
        }

        // `data:` field (the only field the proxy cares about).
        if let Some(payload) = line.strip_prefix(b"data:") {
            let payload = if payload.first() == Some(&b' ') {
                &payload[1..]
            } else {
                payload
            };
            // UTF-8 is valid here: the line is complete (terminated by '\n').
            let payload = std::str::from_utf8(payload).unwrap_or("");
            if !self.data_buf.is_empty() {
                self.data_buf.push('\n');
            }
            self.data_buf.push_str(payload);
        }
        // Other fields (event:, id:, retry:) are ignored.
        None
    }
}

/// All the fields the proxy extracts from a single `data:` payload, computed
/// in one JSON parse (the hot path parses each chunk exactly once).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChunkFields {
    /// `choices[0].delta.content`, if present.
    pub content: Option<String>,
    /// `choices[0].delta.reasoning_content`, if present.
    pub reasoning: Option<String>,
    /// Whether the chunk carries a non-null `choices[0].finish_reason`.
    pub finish_reason: bool,
    /// Whether the chunk carries a non-null `usage` object.
    pub usage: bool,
}

/// Parse a `data:` payload once, extracting every field the proxy needs.
///
/// Malformed JSON yields an all-`None`/`false` result (the chunk is then
/// treated as carrying no content, no finish reason, and no usage).
pub fn parse_chunk(data: &str) -> ChunkFields {
    let v: Value = match serde_json::from_str(data) {
        Ok(v) => v,
        Err(_) => return ChunkFields::default(),
    };
    let delta = &v["choices"][0]["delta"];
    ChunkFields {
        content: delta["content"].as_str().map(str::to_string),
        reasoning: delta["reasoning_content"].as_str().map(str::to_string),
        finish_reason: v["choices"][0]
            .get("finish_reason")
            .map(|f| !f.is_null())
            .unwrap_or(false),
        usage: v.get("usage").map(|u| !u.is_null()).unwrap_or(false),
    }
}

/// Extract `choices[0].delta.content` from a `data:` payload, if present.
///
/// Returns `None` for non-content chunks (role-only, tool_calls, usage-only,
/// malformed JSON). The proxy feeds the returned string to the loop detector.
pub fn delta_content(data: &str) -> Option<String> {
    parse_chunk(data).content
}

/// Extract `choices[0].delta.reasoning_content` from a `data:` payload, if
/// present.
///
/// llama-server (with `--reasoning on`) emits the model's thinking in a
/// separate `reasoning_content` field. The proxy feeds it to its own loop
/// detector (separate from `content`) and carries it back in the continuation
/// request so the upstream template renders it correctly.
pub fn delta_reasoning(data: &str) -> Option<String> {
    parse_chunk(data).reasoning
}

/// Whether a `data:` payload carries a non-null `usage` object.
///
/// llama-server (and OpenAI-compatible servers) emit a final chunk containing
/// `usage` (often together with `[DONE]`). The proxy forwards only the final
/// response's usage, so usage chunks belonging to discarded (looping)
/// responses must be stripped.
pub fn is_usage(data: &str) -> bool {
    parse_chunk(data).usage
}

/// Whether a `data:` payload carries a non-null `choices[0].finish_reason`
/// (the chunk that signals the model finished generating).
pub fn has_finish_reason(data: &str) -> bool {
    parse_chunk(data).finish_reason
}

#[cfg(test)]
mod tests {
    use super::*;

    // B1: normal content chunk.
    #[test]
    fn b1_normal_chunk() {
        let mut p = SseParser::new();
        // A single data line without a trailing blank line is not emitted yet.
        let out = p.feed(r#"data: {"choices":[{"delta":{"content":"hi"}}]}"#.as_bytes());
        assert!(out.is_empty());
        let out = p.feed(b"\n\n");
        assert_eq!(
            out,
            vec![SseEvent::Data(r#"{"choices":[{"delta":{"content":"hi"}}]}"#.into())]
        );
        assert_eq!(delta_content(out[0].payload()), Some("hi".to_string()));
    }

    // B2: empty delta -> no content.
    #[test]
    fn b2_empty_delta() {
        let mut p = SseParser::new();
        let out = p.feed(b"data: {\"choices\":[{\"delta\":{}}]}\n\n");
        assert_eq!(out.len(), 1);
        assert_eq!(delta_content(out[0].payload()), None);
    }

    // B3: [DONE] terminator.
    #[test]
    fn b3_done() {
        let mut p = SseParser::new();
        let out = p.feed(b"data: [DONE]\n\n");
        assert_eq!(out, vec![SseEvent::Done]);
    }

    // B4: multi-line data (one event split across two data: lines).
    #[test]
    fn b4_multiline_data() {
        let mut p = SseParser::new();
        let out = p.feed(b"data: part1\ndata: part2\n\n");
        assert_eq!(out, vec![SseEvent::Data("part1\npart2".into())]);
    }

    // B5: comment / keep-alive line is ignored.
    #[test]
    fn b5_comment() {
        let mut p = SseParser::new();
        let out = p.feed(b": ping\n\n");
        assert!(out.is_empty());
    }

    // B6: blank line is an event separator (no data -> no event).
    #[test]
    fn b6_blank_line() {
        let mut p = SseParser::new();
        let out = p.feed(b"\n\n");
        assert!(out.is_empty());
    }

    // B7: UTF-8 split across chunks reassembles correctly.
    #[test]
    fn b7_utf8_split() {
        let full = "data: {\"choices\":[{\"delta\":{\"content\":\"你\"}}]}\n\n";
        let bytes = full.as_bytes();
        // Split in the middle of the 3-byte UTF-8 sequence for '你'.
        let split = bytes.iter().position(|&b| b == 0xE4).unwrap() + 1;
        let mut p = SseParser::new();
        let mut out = p.feed(&bytes[..split]);
        out.extend(p.feed(&bytes[split..]));
        assert_eq!(out.len(), 1);
        assert_eq!(delta_content(out[0].payload()), Some("你".to_string()));
    }

    // B8: role-only delta -> no content.
    #[test]
    fn b8_role_only() {
        let mut p = SseParser::new();
        let out = p.feed(b"data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n");
        assert_eq!(out.len(), 1);
        assert_eq!(delta_content(out[0].payload()), None);
    }

    // B9: tool_calls delta -> no content.
    #[test]
    fn b9_tool_calls() {
        let mut p = SseParser::new();
        let out = p.feed(b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0}]}}]}\n\n");
        assert_eq!(out.len(), 1);
        assert_eq!(delta_content(out[0].payload()), None);
    }

    // B10: multiple events in one chunk.
    #[test]
    fn b10_multiple_events() {
        let mut p = SseParser::new();
        let out = p.feed(b"data: a\n\ndata: b\n\ndata: [DONE]\n\n");
        assert_eq!(out, vec![
            SseEvent::Data("a".into()),
            SseEvent::Data("b".into()),
            SseEvent::Done,
        ]);
    }

    // B11: CRLF line endings.
    #[test]
    fn b11_crlf() {
        let mut p = SseParser::new();
        let out = p.feed(b"data: hi\r\n\r\n");
        assert_eq!(out, vec![SseEvent::Data("hi".into())]);
    }

    // M4: array-form delta.content (multimodal content parts) -> no content.
    #[test]
    fn m4_array_content() {
        let payload = r#"{"id":"x","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":[{"type":"text","text":"看图："},{"type":"image_url","image_url":{"url":"data:image/png;base64,AAA="}}]},"finish_reason":null}]}"#;
        let fields = parse_chunk(payload);
        assert_eq!(fields.content, None, "array content must not be extracted");
        assert_eq!(fields.reasoning, None);
        assert!(!fields.finish_reason);
        assert!(!fields.usage);
    }
}
