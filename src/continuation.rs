//! Continuation helpers: deduplicate the looped text and build the follow-up
//! request that asks the model to continue from the deduplicated point.

use serde_json::{json, Value};

/// Return the first `n` characters of `s` (multilingual-safe).
pub fn take_chars(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

/// Deduplicate the accumulated text after a loop is detected.
///
/// `hit.kept_chars` is the number of leading characters to keep (prefix + one
/// block). The remaining repeated copies are dropped so the model continues
/// from a clean point instead of re-looping.
pub fn dedup(full_text: &str, kept_chars: usize) -> String {
    take_chars(full_text, kept_chars)
}

/// Build the continuation message list: the original messages plus an
/// assistant turn containing the deduplicated text and a user turn with the
/// continuation prompt.
///
/// `messages` is the `messages` array from the original request (a JSON array).
/// If it is not a valid array, an empty array is used.
pub fn build_continuation_messages(messages: &Value, kept_text: &str, prompt: &str) -> Value {
    let base = if messages.is_array() {
        messages.clone()
    } else {
        json!([])
    };
    let mut out = base.as_array().unwrap().clone();
    out.push(json!({ "role": "assistant", "content": kept_text }));
    out.push(json!({ "role": "user", "content": prompt }));
    Value::Array(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    // take_chars is multilingual-safe.
    #[test]
    fn take_chars_basic() {
        assert_eq!(take_chars("ABCDABCD", 4), "ABCD");
    }

    #[test]
    fn take_chars_cjk() {
        assert_eq!(take_chars("你好世界你好世界", 4), "你好世界");
    }

    #[test]
    fn take_chars_emoji() {
        assert_eq!(take_chars("🦀🐙🦀🐙", 2), "🦀🐙");
    }

    #[test]
    fn take_chars_beyond_len() {
        assert_eq!(take_chars("abc", 10), "abc");
    }

    // dedup keeps prefix + one block.
    #[test]
    fn dedup_prefix_plus_block() {
        assert_eq!(dedup("Hello, world!world!world!", 13), "Hello, world!");
    }

    #[test]
    fn dedup_whole_text_is_loop() {
        assert_eq!(dedup("ABCDABCDABCD", 4), "ABCD");
    }

    // build_continuation_messages appends assistant + user turns.
    #[test]
    fn build_messages_appends() {
        let msgs = json!([
            { "role": "system", "content": "be brief" },
            { "role": "user", "content": "write a poem" }
        ]);
        let out = build_continuation_messages(&msgs, "line1\nline2", "Continue...");
        let arr = out.as_array().unwrap();
        assert_eq!(arr.len(), 4);
        assert_eq!(arr[0]["role"], "system");
        assert_eq!(arr[1]["role"], "user");
        assert_eq!(arr[2]["role"], "assistant");
        assert_eq!(arr[2]["content"], "line1\nline2");
        assert_eq!(arr[3]["role"], "user");
        assert_eq!(arr[3]["content"], "Continue...");
    }

    // A non-array messages value is treated as empty.
    #[test]
    fn build_messages_non_array() {
        let out = build_continuation_messages(&json!("oops"), "kept", "go");
        let arr = out.as_array().unwrap();
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0]["role"], "assistant");
        assert_eq!(arr[1]["role"], "user");
    }

    // The original messages array is not mutated.
    #[test]
    fn build_messages_does_not_mutate_input() {
        let msgs = json!([{ "role": "user", "content": "hi" }]);
        let _ = build_continuation_messages(&msgs, "kept", "go");
        assert_eq!(msgs.as_array().unwrap().len(), 1);
    }
}
