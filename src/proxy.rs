//! Core proxy handler for `POST /v1/chat/completions` (streaming).
//!
//! The handler forwards the request to the upstream llama-server and streams the
//! response back to the client **verbatim** (pass-through, zero added latency).
//! As bytes arrive they are parsed for `delta.content` and fed to an incremental
//! [`LoopDetector`]. When a loop is detected the current upstream response is
//! dropped and a continuation request is issued (original messages + an
//! assistant turn with the deduplicated text + a user turn with the
//! continuation prompt). The client sees one continuous stream; intermediate
//! `[DONE]` markers are suppressed and a single `[DONE]` is emitted at the end.

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::task::{Context, Poll};

use axum::body::Body;
use axum::extract::State;
use axum::http::{header, HeaderMap, Method, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use bytes::Bytes;
use futures_util::stream::{Stream, StreamExt};
use futures_util::FutureExt;
use reqwest::Client;
use serde_json::{json, Value};

use crate::config::Config;
use crate::continuation::{build_continuation_messages, dedup};
use crate::detector::{LoopHit, TieredDetector};
use crate::passthrough;
use crate::server::AppState;
use crate::sse::{delta_content, delta_reasoning, has_finish_reason, is_usage, SseEvent, SseParser};

const DONE: &[u8] = b"data: [DONE]\n\n";

type ByteStream = Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>;
type PendingReq = Pin<Box<dyn Future<Output = Result<reqwest::Response, reqwest::Error>> + Send>>;

/// Global counter used to assign a short, unique id to every request so that
/// concurrent requests can be told apart in the logs.
static REQ_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Allocate the next request id (formatted as `req-<n>`).
fn next_req_id() -> String {
    format!("req-{}", REQ_COUNTER.fetch_add(1, Ordering::Relaxed) + 1)
}

/// Handle `POST /v1/chat/completions`.
///
/// Non-streaming requests and requests with detection disabled are passed
/// through untouched. Streaming requests with detection enabled are handled by
/// [`ProxyStream`].
pub async fn handle_chat_completions(
    State(app): State<AppState>,
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let client = app.client.clone();
    let cfg = app.cfg.clone();
    let req_id = next_req_id();
    // Parse the request body to inspect `stream`.
    let parsed: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(_) => {
            // Unparseable body: pass through and let the upstream report the error.
            tracing::debug!(%req_id, "unparseable request body, passing through");
            return passthrough::passthrough(&client, &cfg.upstream, method, uri, headers, body)
                .await
                .into_response();
        }
    };
    let is_stream = parsed.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let model = parsed.get("model").and_then(Value::as_str).unwrap_or("<none>");
    let n_messages = parsed
        .get("messages")
        .and_then(Value::as_array)
        .map(|m| m.len())
        .unwrap_or(0);
    tracing::debug!(
        %req_id,
        %model,
        is_stream,
        detection_enabled = cfg.detection_enabled,
        messages = n_messages,
        "chat completion request received"
    );

    if !is_stream || !cfg.detection_enabled {
        tracing::debug!(
            %req_id,
            is_stream,
            detection_enabled = cfg.detection_enabled,
            "passthrough (not streaming or detection disabled)"
        );
        return passthrough::passthrough(&client, &cfg.upstream, method, uri, headers, body)
            .await
            .into_response();
    }

    // Streaming + detection: issue the first request and inspect its status.
    // Forward the client's headers (hop-by-hop filtered) so auth (e.g.
    // `Authorization`) reaches the upstream.
    let headers = passthrough::filter_headers(&headers);
    let url = passthrough::build_upstream_url(&cfg.upstream, &uri);
    let mut req = client.post(&url);
    for (name, value) in headers.iter() {
        req = req.header(name, value);
    }
    match req.json(&parsed).send().await {
        Ok(resp) if resp.status().is_success() => {
            let stream = ProxyStream::new(
                client,
                cfg,
                parsed,
                Some(Box::pin(resp.bytes_stream())),
                req_id,
                headers,
            );
            Response::builder()
                .status(StatusCode::OK)
                .header(header::CONTENT_TYPE, "text/event-stream")
                .header(header::CACHE_CONTROL, "no-cache")
                .body(Body::from_stream(stream))
                .unwrap()
        }
        Ok(resp) => {
            // Non-2xx: forward the upstream status and body.
            let status = resp.status();
            tracing::warn!(%req_id, %status, "first upstream request non-2xx, forwarding");
            let b = resp.bytes().await.unwrap_or_default();
            Response::builder().status(status).body(Body::from(b)).unwrap()
        }
        Err(e) => {
            tracing::warn!(%req_id, "upstream request failed: {e}");
            (StatusCode::BAD_GATEWAY, format!("upstream error: {e}")).into_response()
        }
    }
}

/// A stream that drives the upstream, performs pass-through + loop detection,
/// and transparently issues continuation requests.
pub struct ProxyStream {
    client: Client,
    cfg: Config,
    /// Short unique id for this request (for log correlation).
    req_id: String,
    /// Client request headers (hop-by-hop filtered), forwarded to upstream.
    headers: HeaderMap,
    /// The original parsed request body (all params except `messages`).
    original_body: Value,
    /// The original `messages` array (never mutated).
    original_messages: Value,
    /// Current `messages` array (original + latest continuation turns).
    messages: Value,
    /// Loop detector for the current response's `content` (reset per response
    /// and whenever a new content segment begins).
    detector: Option<TieredDetector>,
    /// Loop detector for the current response's `reasoning_content` (reset per
    /// response and whenever a new reasoning segment begins).
    reasoning_detector: Option<TieredDetector>,
    /// Accumulated `delta.content` for the current response.
    full_text: String,
    /// Accumulated `delta.reasoning_content` for the current response.
    reasoning_text: String,
    /// The last non-empty segment (`true` = reasoning, `false` = content),
    /// used to detect segment switches so the opposite detector can be reset.
    last_was_reasoning: Option<bool>,
    /// SSE parser for the current response (reset per response).
    sse: SseParser,
    /// Byte stream of the current upstream response, if any.
    resp: Option<ByteStream>,
    /// In-flight continuation request, if any.
    pending: Option<PendingReq>,
    /// Number of continuation requests started so far.
    continuations: usize,
    /// Whether the stream has ended.
    done: bool,
    /// Whether any upstream chunk carried a non-null `finish_reason`
    /// (diagnostic: clients such as VS Code expect one before `[DONE]`).
    finish_reason_seen: bool,
}

impl ProxyStream {
    /// Create a stream. `first_resp` is the byte stream of the already-issued
    /// first request (always `Some` from the handler).
    pub fn new(
        client: Client,
        cfg: Config,
        original_body: Value,
        first_resp: Option<ByteStream>,
        req_id: String,
        headers: HeaderMap,
    ) -> Self {
        let original_messages = original_body
            .get("messages")
            .cloned()
            .unwrap_or_else(|| json!([]));
        let detector = Some(TieredDetector::new(
            cfg.short_min_block,
            cfg.short_max_block,
            cfg.short_repeats,
            cfg.long_max_block,
        ));
        let reasoning_detector = Some(TieredDetector::new(
            cfg.short_min_block,
            cfg.short_max_block,
            cfg.short_repeats,
            cfg.long_max_block,
        ));
        let messages = original_messages.clone();
        ProxyStream {
            client,
            cfg,
            req_id,
            headers,
            original_body,
            original_messages,
            messages,
            detector,
            reasoning_detector,
            full_text: String::new(),
            reasoning_text: String::new(),
            last_was_reasoning: None,
            sse: SseParser::new(),
            resp: first_resp,
            pending: None,
            continuations: 0,
            done: false,
            finish_reason_seen: false,
        }
    }

    /// Build the request body for the current `messages`.
    fn build_request_body(&self) -> Value {
        let mut body = self.original_body.clone();
        if let Some(obj) = body.as_object_mut() {
            obj.insert("messages".to_string(), self.messages.clone());
        }
        body
    }
}

impl Stream for ProxyStream {
    type Item = Result<Bytes, std::io::Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            if self.done {
                return Poll::Ready(None);
            }

            // 1) Read from the current upstream response, if any.
            if let Some(resp) = self.resp.as_mut() {
                match resp.poll_next_unpin(cx) {
                    Poll::Ready(Some(Ok(bytes))) => {
                        return Poll::Ready(Some(Ok(self.process_chunk(bytes))));
                    }
                    Poll::Ready(Some(Err(e))) => {
                        tracing::warn!(%self.req_id, "upstream stream error: {e}");
                        self.log_stream_end("upstream stream error");
                        self.done = true;
                        return Poll::Ready(Some(Ok(self.terminal_done())));
                    }
                    Poll::Ready(None) => {
                        // Upstream ended without [DONE]: emit one to avoid a client hang.
                        self.log_stream_end("upstream ended without [DONE]");
                        self.done = true;
                        return Poll::Ready(Some(Ok(self.terminal_done())));
                    }
                    Poll::Pending => return Poll::Pending,
                }
            }

            // 2) No response stream: start a continuation request if allowed.
            if self.continuations >= self.cfg.max_retries {
                // Retries exhausted: end the stream.
                self.log_stream_end("max_retries exhausted");
                self.done = true;
                return Poll::Ready(Some(Ok(self.terminal_done())));
            }

            if self.pending.is_none() {
                let body = self.build_request_body();
                let url = format!(
                    "{}/v1/chat/completions",
                    self.cfg.upstream.trim_end_matches('/')
                );
                let n_messages = body
                    .get("messages")
                    .and_then(Value::as_array)
                    .map(|m| m.len())
                    .unwrap_or(0);
                tracing::debug!(
                    %self.req_id,
                    attempt = self.continuations + 1,
                    messages = n_messages,
                    "issuing continuation request"
                );
                let client = self.client.clone();
                let headers = self.headers.clone();
                self.pending = Some(Box::pin(async move {
                    let mut req = client.post(url);
                    for (name, value) in headers.iter() {
                        req = req.header(name, value);
                    }
                    req.json(&body).send().await
                }));
            }

            if let Some(p) = self.pending.as_mut() {
                match p.poll_unpin(cx) {
                    Poll::Ready(Ok(resp)) => {
                        self.pending = None;
                        if resp.status().is_success() {
                            self.continuations += 1;
                            self.resp = Some(Box::pin(resp.bytes_stream()));
                            // Loop back to read from the new response.
                        } else {
                            tracing::warn!(
                                %self.req_id,
                                status = %resp.status(),
                                "continuation upstream returned non-2xx"
                            );
                            self.log_stream_end("continuation non-2xx");
                            self.done = true;
                            return Poll::Ready(Some(Ok(self.terminal_done())));
                        }
                    }
                    Poll::Ready(Err(e)) => {
                        tracing::warn!(%self.req_id, "continuation request failed: {e}");
                        self.log_stream_end("continuation request failed");
                        self.pending = None;
                        self.done = true;
                        return Poll::Ready(Some(Ok(self.terminal_done())));
                    }
                    Poll::Pending => return Poll::Pending,
                }
            }
        }
    }
}

impl ProxyStream {
    /// Process one chunk of upstream bytes: feed the detector, decide whether to
    /// continue, and return the bytes to forward to the client.
    fn process_chunk(&mut self, bytes: Bytes) -> Bytes {
        let events = self.sse.feed(&bytes);
        let mut got_done = false;
        // `(is_reasoning, hit)`: which field's detector fired.
        let mut hit: Option<(bool, LoopHit)> = None;

        for ev in &events {
            match ev {
                SseEvent::Data(payload) => {
                    if has_finish_reason(payload) {
                        self.finish_reason_seen = true;
                    }
                    let reasoning = delta_reasoning(payload);
                    let content = delta_content(payload);
                    // Current segment: content takes precedence when both are
                    // present in one delta (content follows reasoning).
                    let cur = if content.is_some() {
                        Some(false)
                    } else if reasoning.is_some() {
                        Some(true)
                    } else {
                        None
                    };
                    // On a segment switch, reset the detector of the field we
                    // are leaving so repetition is never counted across
                    // separate thinking/content blocks.
                    if let (Some(cur), Some(last)) = (cur, self.last_was_reasoning) {
                        if cur != last {
                            if cur {
                                // Switching to reasoning: reset content detector.
                                self.detector = Some(TieredDetector::new(
                                    self.cfg.short_min_block,
                                    self.cfg.short_max_block,
                                    self.cfg.short_repeats,
                                    self.cfg.long_max_block,
                                ));
                            } else {
                                // Switching to content: reset reasoning detector.
                                self.reasoning_detector = Some(TieredDetector::new(
                                    self.cfg.short_min_block,
                                    self.cfg.short_max_block,
                                    self.cfg.short_repeats,
                                    self.cfg.long_max_block,
                                ));
                            }
                        }
                    }
                    // Feed reasoning first, then content (model output order).
                    // Each field has its own detector; a hit in either triggers
                    // a continuation.
                    if let Some(r) = &reasoning {
                        self.reasoning_text.push_str(r);
                        if hit.is_none() {
                            if let Some(det) = self.reasoning_detector.as_mut() {
                                for c in r.chars() {
                                    if let Some(h) = det.feed(c) {
                                        hit = Some((true, h));
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    if let Some(c) = &content {
                        self.full_text.push_str(c);
                        if hit.is_none() {
                            if let Some(det) = self.detector.as_mut() {
                                for ch in c.chars() {
                                    if let Some(h) = det.feed(ch) {
                                        hit = Some((false, h));
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    if cur.is_some() {
                        self.last_was_reasoning = cur;
                    }
                }
                SseEvent::Done => got_done = true,
            }
        }

        // A loop takes priority over [DONE]: if the model loops and sends
        // [DONE] in the same chunk, we still continue (the [DONE] belongs to
        // the discarded looping response). This keeps behavior independent of
        // how the upstream chunks its bytes.
        if let Some((is_reasoning, h)) = hit {
            // Deduplicate the field that looped. When the loop is in content,
            // the (complete) reasoning is preserved in full; when it is in
            // reasoning, no content has been produced yet.
            let (kept_reasoning, kept_content) = if is_reasoning {
                (dedup(&self.reasoning_text, h.kept_chars), String::new())
            } else {
                (self.reasoning_text.clone(), dedup(&self.full_text, h.kept_chars))
            };
            tracing::info!(
                %self.req_id,
                attempt = self.continuations + 1,
                field = if is_reasoning { "reasoning_content" } else { "content" },
                block_len = h.block_len,
                run = h.run,
                kept_chars = h.kept_chars,
                kept_len = kept_reasoning.chars().count() + kept_content.chars().count(),
                "loop detected, continuing"
            );
            // Always build from the ORIGINAL messages so continuations do not
            // accumulate across retries.
            self.messages = build_continuation_messages(
                &self.original_messages,
                &kept_reasoning,
                &kept_content,
                &self.cfg.continuation_prompt,
            );
            // Reset per-response state for the continuation.
            self.resp = None;
            self.sse = SseParser::new();
            self.full_text.clear();
            self.reasoning_text.clear();
            self.last_was_reasoning = None;
            self.finish_reason_seen = false;
            self.detector = Some(TieredDetector::new(
                self.cfg.short_min_block,
                self.cfg.short_max_block,
                self.cfg.short_repeats,
                self.cfg.long_max_block,
            ));
            self.reasoning_detector = Some(TieredDetector::new(
                self.cfg.short_min_block,
                self.cfg.short_max_block,
                self.cfg.short_repeats,
                self.cfg.long_max_block,
            ));
            if self.continuations >= self.cfg.max_retries {
                // Retries exhausted: end the stream.
                self.log_stream_end("max_retries exhausted");
                self.done = true;
                let mut out = Vec::with_capacity(bytes.len() + DONE.len());
                out.extend_from_slice(&bytes);
                if !got_done {
                    out.extend_from_slice(DONE);
                }
                return Bytes::from(out);
            }
            // Continue: forward the content, stripping the [DONE] marker and
            // any usage chunk (both belong to the discarded looping response).
            return Self::serialize_events(&events);
        }

        if got_done {
            // The model finished this response: forward [DONE] and end.
            self.log_stream_end("completed");
            self.done = true;
            return bytes;
        }

        bytes
    }

    /// Log a single `info` line summarizing how this request's stream ended,
    /// including the total number of continuation requests issued.
    fn log_stream_end(&self, reason: &str) {
        tracing::info!(
            %self.req_id,
            continuations = self.continuations,
            finish_reason_seen = self.finish_reason_seen,
            reason,
            "stream ended"
        );
    }

    /// Re-serialize SSE events, omitting the [DONE] marker and any `usage`
    /// chunks. Used when a loop is detected in a chunk that also contains
    /// [DONE] and we are continuing: the [DONE] and the usage belong to the
    /// discarded looping response, so only the final response's usage reaches
    /// the client.
    fn serialize_events(events: &[SseEvent]) -> Bytes {
        let mut out = Vec::new();
        for ev in events {
            if let SseEvent::Data(payload) = ev {
                if is_usage(payload) {
                    continue;
                }
                out.extend_from_slice(b"data: ");
                out.extend_from_slice(payload.as_bytes());
                out.extend_from_slice(b"\n\n");
            }
        }
        Bytes::from(out)
    }

    /// Build the terminal `[DONE]` bytes.
    fn terminal_done(&self) -> Bytes {
        Bytes::copy_from_slice(DONE)
    }
}
