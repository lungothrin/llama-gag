//! Integration tests (D1-D11) for the llama-gag proxy.
//!
//! Each test spins up a **mock upstream** (a real axum server on an ephemeral
//! port) and the **proxy under test** (another real axum server on an ephemeral
//! port), then drives the proxy with a real HTTP client. This exercises the full
//! stack: routing, passthrough, SSE parsing, incremental loop detection, and
//! continuation orchestration over real TCP.

use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, Method, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::Router;
use bytes::Bytes;
use futures_util::stream::{Stream, StreamExt};
use std::marker::Unpin;
use reqwest::Client;
use serde_json::{json, Value};

use llama_gag::config::Config;
use llama_gag::server::router;
use llama_gag::sse::{delta_content, SseEvent, SseParser};

// ---------------------------------------------------------------------------
// Mock upstream
// ---------------------------------------------------------------------------

/// Handler for `POST /v1/chat/completions` on the mock upstream. Receives the
/// parsed request body and returns the response to send.
type ChatHandler = Arc<dyn Fn(&Value) -> Response + Send + Sync>;

#[derive(Clone)]
struct MockState {
    chat: ChatHandler,
    chat_requests: Arc<Mutex<Vec<Value>>>,
    /// Raw (unparsed) body bytes of every chat request, so tests can assert the
    /// proxy forwards the client's body verbatim.
    chat_raw_bodies: Arc<Mutex<Vec<Bytes>>>,
    /// Request headers of every chat request, so tests can assert which headers
    /// the proxy forwards.
    chat_headers: Arc<Mutex<Vec<HeaderMap>>>,
    other_requests: Arc<Mutex<Vec<(String, String)>>>,
}

struct MockUpstream {
    addr: String,
    chat_requests: Arc<Mutex<Vec<Value>>>,
    chat_raw_bodies: Arc<Mutex<Vec<Bytes>>>,
    chat_headers: Arc<Mutex<Vec<HeaderMap>>>,
    other_requests: Arc<Mutex<Vec<(String, String)>>>,
}

impl MockUpstream {
    async fn start(chat: ChatHandler) -> Self {
        let chat_requests: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(Vec::new()));
        let chat_raw_bodies: Arc<Mutex<Vec<Bytes>>> = Arc::new(Mutex::new(Vec::new()));
        let chat_headers: Arc<Mutex<Vec<HeaderMap>>> = Arc::new(Mutex::new(Vec::new()));
        let other_requests: Arc<Mutex<Vec<(String, String)>>> = Arc::new(Mutex::new(Vec::new()));
        let state = MockState {
            chat,
            chat_requests: chat_requests.clone(),
            chat_raw_bodies: chat_raw_bodies.clone(),
            chat_headers: chat_headers.clone(),
            other_requests: other_requests.clone(),
        };

        let app = Router::new()
            .route(
                "/v1/chat/completions",
                post(
                    |State(s): State<MockState>, headers: HeaderMap, body: Bytes| async move {
                        let parsed: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
                        s.chat_requests.lock().unwrap().push(parsed.clone());
                        s.chat_raw_bodies.lock().unwrap().push(body);
                        s.chat_headers.lock().unwrap().push(headers);
                        (s.chat)(&parsed)
                    },
                ),
            )
            .fallback(|State(s): State<MockState>, method: Method, uri: Uri| async move {
                let path = uri.path().to_string();
                s.other_requests
                    .lock()
                    .unwrap()
                    .push((method.to_string(), path.clone()));
                let v = json!({ "path": path, "method": method.to_string() });
                (StatusCode::OK, serde_json::to_vec(&v).unwrap()).into_response()
            })
            .with_state(state);

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        MockUpstream {
            addr: format!("http://{addr}"),
            chat_requests,
            chat_raw_bodies,
            chat_headers,
            other_requests,
        }
    }

    fn chat_request_bodies(&self) -> Vec<Value> {
        self.chat_requests.lock().unwrap().clone()
    }

    /// The exact bytes of every chat request, in arrival order.
    fn chat_request_raw_bodies(&self) -> Vec<Bytes> {
        self.chat_raw_bodies.lock().unwrap().clone()
    }

    /// The request headers of every chat request, in arrival order.
    fn chat_request_headers(&self) -> Vec<HeaderMap> {
        self.chat_headers.lock().unwrap().clone()
    }

    fn other_requests(&self) -> Vec<(String, String)> {
        self.other_requests.lock().unwrap().clone()
    }
}

// ---------------------------------------------------------------------------
// Proxy under test
// ---------------------------------------------------------------------------

struct Proxy {
    addr: String,
    client: Client,
}

async fn start_proxy(cfg: Config) -> Proxy {
    let client = Client::new();
    let app = router(client, cfg);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    Proxy {
        addr: format!("http://{addr}"),
        client: Client::new(),
    }
}

/// A test config pointing at the given upstream, with a small
/// `short_min_block` so short test blocks can trigger the detector.
fn test_config(upstream: &str) -> Config {
    test_config_detection(upstream, true)
}

fn test_config_detection(upstream: &str, detection_enabled: bool) -> Config {
    Config {
        listen: "127.0.0.1:0".to_string(),
        upstream: upstream.to_string(),
        max_body_size: 64 * 1024 * 1024,
        detection_enabled,
        short_min_block: 4,
        short_max_block: 64,
        short_repeats: 2,
        long_max_block: 1024,
        continuation_prompt: "CONTINUE".to_string(),
        max_retries: 3,
        log_level: "info".to_string(),
        log_file: None,
    }
}

// ---------------------------------------------------------------------------
// Response-building helpers (mock upstream)
// ---------------------------------------------------------------------------

/// A non-streaming JSON response.
fn json_response(status: StatusCode, value: &Value) -> Response {
    let body = serde_json::to_vec(value).unwrap();
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Body::from(body))
        .unwrap()
}

/// A streaming SSE response from raw chunks (each chunk is raw SSE text).
fn sse_response(chunks: Vec<String>) -> Response {
    let stream = futures_util::stream::iter(
        chunks
            .into_iter()
            .map(|c| Ok::<Bytes, std::io::Error>(Bytes::from(c))),
    );
    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream")
        .body(Body::from_stream(stream))
        .unwrap()
}

/// Build a single SSE `data:` chunk carrying `delta.content`.
fn content_chunk(content: &str) -> String {
    let v = json!({
        "id": "x",
        "object": "chat.completion.chunk",
        "choices": [{ "index": 0, "delta": { "content": content }, "finish_reason": null }]
    });
    format!("data: {v}\n\n")
}

/// Build a single SSE `data:` chunk carrying `delta.reasoning_content`.
fn reasoning_chunk(reasoning: &str) -> String {
    let v = json!({
        "id": "x",
        "object": "chat.completion.chunk",
        "choices": [{ "index": 0, "delta": { "reasoning_content": reasoning }, "finish_reason": null }]
    });
    format!("data: {v}\n\n")
}

/// Build a single SSE `data:` chunk carrying an arbitrary `delta` value.
fn delta_chunk(delta: &Value) -> String {
    let v = json!({
        "id": "x",
        "object": "chat.completion.chunk",
        "choices": [{ "index": 0, "delta": delta, "finish_reason": null }]
    });
    format!("data: {v}\n\n")
}

const DONE_CHUNK: &str = "data: [DONE]\n\n";

/// A stream wrapper that records when it is dropped. Used to verify the proxy
/// releases the upstream connection on client cancel.
struct DropGuardStream<S> {
    inner: S,
    dropped: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl<S: Stream + Unpin> Stream for DropGuardStream<S> {
    type Item = S::Item;
    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<S::Item>> {
        std::pin::Pin::new(&mut self.inner).poll_next(cx)
    }
}

impl<S> Drop for DropGuardStream<S> {
    fn drop(&mut self) {
        self.dropped
            .store(true, std::sync::atomic::Ordering::SeqCst);
    }
}

/// A streaming SSE response that yields the given chunks, then blocks forever,
/// and records when it is dropped.
fn sse_response_with_drop_guard(
    chunks: Vec<String>,
    dropped: std::sync::Arc<std::sync::atomic::AtomicBool>,
) -> Response {
    let base = futures_util::stream::iter(
        chunks
            .into_iter()
            .map(|c| Ok::<Bytes, std::io::Error>(Bytes::from(c))),
    )
    .chain(futures_util::stream::pending::<Result<Bytes, std::io::Error>>());
    let stream = DropGuardStream {
        inner: base,
        dropped,
    };
    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream")
        .body(Body::from_stream(stream))
        .unwrap()
}

// ---------------------------------------------------------------------------
// Client-side reading helpers
// ---------------------------------------------------------------------------

/// Read the entire response body into bytes.
async fn read_all(resp: reqwest::Response) -> Vec<u8> {
    let mut out = Vec::new();
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        out.extend_from_slice(&chunk.unwrap());
    }
    out
}

/// Parse all SSE events from a byte buffer.
fn parse_events(bytes: &[u8]) -> Vec<SseEvent> {
    let mut parser = SseParser::new();
    parser.feed(bytes)
}

/// Concatenate all `delta.content` across the SSE events.
fn extract_content(bytes: &[u8]) -> String {
    let mut content = String::new();
    for ev in parse_events(bytes) {
        if let SseEvent::Data(payload) = ev {
            if let Some(c) = delta_content(&payload) {
                content.push_str(&c);
            }
        }
    }
    content
}

/// Count the `[DONE]` terminators in the SSE byte buffer.
fn count_done(bytes: &[u8]) -> usize {
    parse_events(bytes)
        .iter()
        .filter(|e| matches!(e, SseEvent::Done))
        .count()
}

/// Build a single SSE `data:` chunk carrying a `usage` object (as llama-server
/// emits in its final chunk).
fn usage_chunk(prompt_tokens: u32, completion_tokens: u32) -> String {
    let v = json!({
        "id": "x",
        "object": "chat.completion.chunk",
        "choices": [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
    });
    format!("data: {v}\n\n")
}

/// Count the `data:` events carrying a non-null `usage` object.
fn count_usage(bytes: &[u8]) -> usize {
    parse_events(bytes)
        .iter()
        .filter(|e| matches!(e, SseEvent::Data(p) if llama_gag::sse::is_usage(p)))
        .count()
}

/// Return the `completion_tokens` of the (single) usage event, if any.
fn usage_completion_tokens(bytes: &[u8]) -> Option<u32> {
    for ev in parse_events(bytes) {
        if let SseEvent::Data(payload) = ev {
            if let Ok(v) = serde_json::from_str::<Value>(&payload) {
                if let Some(u) = v.get("usage") {
                    if !u.is_null() {
                        return u["completion_tokens"].as_u64().map(|n| n as u32);
                    }
                }
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// D1: non-streaming passthrough -> byte-identical, no detection
// ---------------------------------------------------------------------------

#[tokio::test]
async fn d1_non_streaming_passthrough() {
    let completion = json!({
        "id": "cmpl-1",
        "object": "chat.completion",
        "model": "m",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": "hello world" },
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8 }
    });

    let value = completion.clone();
    let mock = MockUpstream::start(Arc::new(move |_body| {
        json_response(StatusCode::OK, &value)
    }))
    .await;
    let proxy = start_proxy(test_config(&mock.addr)).await;

    let req_body = json!({
        "model": "m",
        "stream": false,
        "messages": [{ "role": "user", "content": "hi" }]
    });

    let resp = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.addr))
        .json(&req_body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = read_all(resp).await;

    // Response body is byte-identical to the upstream's.
    let expected = serde_json::to_vec(&completion).unwrap();
    assert_eq!(bytes, expected, "non-streaming body must be byte-identical");

    // The request was forwarded verbatim (parsed value equals the original).
    let bodies = mock.chat_request_bodies();
    assert_eq!(bodies.len(), 1, "exactly one upstream request");
    assert_eq!(bodies[0], req_body, "request body forwarded verbatim");
}

// ---------------------------------------------------------------------------
// D2: route passthrough -> /v1/models, /input_tokens, /health forwarded
// ---------------------------------------------------------------------------

#[tokio::test]
async fn d2_route_passthrough() {
    let mock = MockUpstream::start(Arc::new(|_body| {
        json_response(StatusCode::OK, &json!({ "unexpected": true }))
    }))
    .await;
    let proxy = start_proxy(test_config(&mock.addr)).await;

    // GET /v1/models
    let resp = proxy
        .client
        .get(format!("{}/v1/models", proxy.addr))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["path"], "/v1/models");

    // POST /v1/chat/completions/input_tokens (must NOT hit the chat endpoint)
    let resp = proxy
        .client
        .post(format!("{}/v1/chat/completions/input_tokens", proxy.addr))
        .json(&json!({ "model": "m", "messages": [] }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["path"], "/v1/chat/completions/input_tokens");

    // GET /health
    let resp = proxy
        .client
        .get(format!("{}/health", proxy.addr))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["path"], "/health");

    // None of these hit the chat endpoint.
    assert!(
        mock.chat_request_bodies().is_empty(),
        "route passthrough must not touch the chat endpoint"
    );

    // The mock recorded all three with the correct method + path.
    let others = mock.other_requests();
    assert_eq!(others.len(), 3);
    assert_eq!(others[0], ("GET".to_string(), "/v1/models".to_string()));
    assert_eq!(
        others[1],
        ("POST".to_string(), "/v1/chat/completions/input_tokens".to_string())
    );
    assert_eq!(others[2], ("GET".to_string(), "/health".to_string()));
}

// ---------------------------------------------------------------------------
// D3: stream + loop -> client sees prefix + 2 blocks + continuation + SINGLE [DONE]
// ---------------------------------------------------------------------------

#[tokio::test]
async fn d3_stream_loop_continuation() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let counter = Arc::new(AtomicUsize::new(0));
    let handler = Arc::new(move |_body: &Value| {
        let n = counter.fetch_add(1, Ordering::SeqCst);
        if n == 0 {
            // First request: the model loops. "Hello " + "ABCD" x2 triggers the
            // detector (short_min_block=4, short_repeats=2) at the 2nd block.
            sse_response(vec![
                content_chunk("Hello "),
                content_chunk("ABCDABCD"),
                DONE_CHUNK.to_string(),
            ])
        } else {
            // Continuation: a normal completion that ends the stream.
            sse_response(vec![
                content_chunk(" (continued) The end."),
                DONE_CHUNK.to_string(),
            ])
        }
    });

    let mock = MockUpstream::start(handler).await;
    let proxy = start_proxy(test_config(&mock.addr)).await;

    let req_body = json!({
        "model": "m",
        "stream": true,
        "messages": [{ "role": "user", "content": "hi" }]
    });

    let resp = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.addr))
        .json(&req_body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = read_all(resp).await;

    // The client sees: prefix + 2 blocks (pass-through) + continuation content.
    let content = extract_content(&bytes);
    assert_eq!(content, "Hello ABCDABCD (continued) The end.");

    // Exactly ONE [DONE] for the whole merged stream.
    assert_eq!(count_done(&bytes), 1, "merged stream must have a single [DONE]");

    // Two upstream requests: the original + one continuation.
    let bodies = mock.chat_request_bodies();
    assert_eq!(bodies.len(), 2, "original + one continuation");

    // The continuation request carries the deduplicated assistant turn
    // (prefix + one block) and the continuation prompt.
    let cont_msgs = bodies[1].get("messages").and_then(Value::as_array).cloned().unwrap();
    assert_eq!(cont_msgs.len(), 3);
    assert_eq!(cont_msgs[0], json!({ "role": "user", "content": "hi" }));
    assert_eq!(
        cont_msgs[1],
        json!({ "role": "assistant", "content": "Hello ABCD" })
    );
    assert_eq!(
        cont_msgs[2],
        json!({ "role": "user", "content": "CONTINUE" })
    );
}

// ---------------------------------------------------------------------------
// D14: reasoning_content loop -> detected in the reasoning field (separate
// detector), and the continuation request carries the deduplicated
// `reasoning_content` with an empty `content`.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn d14_reasoning_loop_continuation() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let counter = Arc::new(AtomicUsize::new(0));
    let handler = Arc::new(move |_body: &Value| {
        let n = counter.fetch_add(1, Ordering::SeqCst);
        if n == 0 {
            // First request: the model loops in its reasoning. "Let me " +
            // "ABCD" x2 triggers the reasoning detector (short_min_block=4,
            // short_repeats=2) at the 2nd block.
            sse_response(vec![
                reasoning_chunk("Let me "),
                reasoning_chunk("ABCDABCD"),
                DONE_CHUNK.to_string(),
            ])
        } else {
            // Continuation: a normal completion that ends the stream.
            sse_response(vec![
                content_chunk("The answer is 42."),
                DONE_CHUNK.to_string(),
            ])
        }
    });

    let mock = MockUpstream::start(handler).await;
    let proxy = start_proxy(test_config(&mock.addr)).await;

    let req_body = json!({
        "model": "m",
        "stream": true,
        "messages": [{ "role": "user", "content": "hi" }]
    });

    let resp = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.addr))
        .json(&req_body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = read_all(resp).await;

    // The client sees only the continuation content (the reasoning was
    // pass-through in `reasoning_content`, not `content`).
    let content = extract_content(&bytes);
    assert_eq!(content, "The answer is 42.");

    // Exactly ONE [DONE] for the whole merged stream.
    assert_eq!(count_done(&bytes), 1, "merged stream must have a single [DONE]");

    // Two upstream requests: the original + one continuation.
    let bodies = mock.chat_request_bodies();
    assert_eq!(bodies.len(), 2, "original + one continuation");

    // The continuation request carries the deduplicated reasoning in
    // `reasoning_content` (prefix + one block) and an empty `content`.
    let cont_msgs = bodies[1].get("messages").and_then(Value::as_array).cloned().unwrap();
    assert_eq!(cont_msgs.len(), 3);
    assert_eq!(cont_msgs[0], json!({ "role": "user", "content": "hi" }));
    assert_eq!(
        cont_msgs[1],
        json!({ "role": "assistant", "reasoning_content": "Let me ABCD", "content": "" })
    );
    assert_eq!(
        cont_msgs[2],
        json!({ "role": "user", "content": "CONTINUE" })
    );
}

// ---------------------------------------------------------------------------
// D4: [DONE] suppression -> intermediate [DONE] not forwarded.
//
// This specifically exercises the case where the loop trigger and the
// intermediate [DONE] arrive in the SAME chunk: the proxy must strip that
// [DONE] (it belongs to the discarded looping response) and continue, so the
// client sees only the final [DONE].
// ---------------------------------------------------------------------------

#[tokio::test]
async fn d4_done_suppression_same_chunk() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let counter = Arc::new(AtomicUsize::new(0));
    let handler = Arc::new(move |_body: &Value| {
        let n = counter.fetch_add(1, Ordering::SeqCst);
        if n == 0 {
            // Loop content AND [DONE] in a single chunk. The detector triggers
            // on "ABCDABCD" and the [DONE] is in the same raw chunk.
            sse_response(vec![
                content_chunk("Hello "),
                format!("{}{}", content_chunk("ABCDABCD"), DONE_CHUNK),
            ])
        } else {
            sse_response(vec![
                content_chunk(" (continued)"),
                DONE_CHUNK.to_string(),
            ])
        }
    });

    let mock = MockUpstream::start(handler).await;
    let proxy = start_proxy(test_config(&mock.addr)).await;

    let req_body = json!({
        "model": "m",
        "stream": true,
        "messages": [{ "role": "user", "content": "hi" }]
    });

    let resp = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.addr))
        .json(&req_body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = read_all(resp).await;

    // Content: prefix + 2 blocks (pass-through) + continuation.
    let content = extract_content(&bytes);
    assert_eq!(content, "Hello ABCDABCD (continued)");

    // The intermediate [DONE] was stripped; only the final [DONE] remains.
    assert_eq!(
        count_done(&bytes),
        1,
        "intermediate [DONE] must be suppressed, only the final one forwarded"
    );

    // A continuation was issued.
    assert_eq!(mock.chat_request_bodies().len(), 2);
}

// ---------------------------------------------------------------------------
// D5: max_retries exhausted -> upstream always loops -> 3 continuations then end
// ---------------------------------------------------------------------------

#[tokio::test]
async fn d5_max_retries_exhausted() {
    // The upstream ALWAYS loops, so every response triggers a continuation.
    let handler = Arc::new(|_body: &Value| {
        sse_response(vec![content_chunk("ABCDABCD"), DONE_CHUNK.to_string()])
    });

    let mock = MockUpstream::start(handler).await;
    let proxy = start_proxy(test_config(&mock.addr)).await;

    let req_body = json!({
        "model": "m",
        "stream": true,
        "messages": [{ "role": "user", "content": "hi" }]
    });

    let resp = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.addr))
        .json(&req_body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = read_all(resp).await;

    // Original + 3 continuations = 4 upstream requests.
    assert_eq!(
        mock.chat_request_bodies().len(),
        4,
        "original + max_retries(3) continuations"
    );

    // The client sees 4 copies of the loop content (one per response).
    let content = extract_content(&bytes);
    assert_eq!(content, "ABCDABCD".repeat(4));

    // The stream ends with exactly one [DONE].
    assert_eq!(count_done(&bytes), 1, "exactly one [DONE] after retries exhausted");
}

// ---------------------------------------------------------------------------
// D6: client cancel -> client disconnects -> drop upstream, stop continuation
// ---------------------------------------------------------------------------

#[tokio::test]
async fn d6_client_cancel_drops_upstream() {
    use std::sync::atomic::{AtomicBool, Ordering};

    let dropped = Arc::new(AtomicBool::new(false));
    // The upstream yields a little content, then blocks forever (a long-lived
    // stream). When the client cancels, the proxy must drop this stream.
    let dropped_for_handler = dropped.clone();
    let handler = Arc::new(move |_body: &Value| {
        sse_response_with_drop_guard(vec![content_chunk("Hello ")], dropped_for_handler.clone())
    });

    let mock = MockUpstream::start(handler).await;
    let proxy = start_proxy(test_config(&mock.addr)).await;

    let req_body = json!({
        "model": "m",
        "stream": true,
        "messages": [{ "role": "user", "content": "hi" }]
    });

    // Start the request, read a little, then drop the response (client cancel).
    let resp = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.addr))
        .json(&req_body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let mut stream = resp.bytes_stream();
    // Read at least one chunk so the proxy has started forwarding.
    let _first = stream.next().await;
    // Client cancel: drop the response stream.
    drop(stream);

    // The proxy should release the upstream connection, dropping the stream.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while !dropped.load(Ordering::SeqCst) && std::time::Instant::now() < deadline {
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    assert!(
        dropped.load(Ordering::SeqCst),
        "upstream stream must be dropped when the client cancels"
    );
}

// D7: upstream disconnects mid-stream (no [DONE]) -> proxy emits [DONE].
#[tokio::test]
async fn d7_upstream_disconnect_emits_done() {
    let upstream = MockUpstream::start(Arc::new(|_body: &Value| {
        // Stream ends without [DONE] (upstream died mid-stream).
        sse_response(vec![content_chunk("partial answer")])
    }))
    .await;
    let proxy = start_proxy(test_config(&upstream.addr)).await;

    let completion = json!({
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": true,
    });
    let resp = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.addr))
        .json(&completion)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = read_all(resp).await;

    // Content is forwarded, and the proxy terminates the stream with [DONE].
    assert_eq!(extract_content(&bytes), "partial answer");
    assert_eq!(count_done(&bytes), 1);
    // No continuation was issued (no loop).
    assert_eq!(upstream.chat_request_bodies().len(), 1);
}

// D8: concurrency isolation — 10 concurrent requests, independent detector state.
#[tokio::test]
async fn d8_concurrency_isolation() {
    let upstream = MockUpstream::start(Arc::new(|body: &Value| {
        let msgs = body["messages"].as_array().cloned().unwrap_or_default();
        let is_continuation = msgs.len() > 1;
        // First user message carries the original marker (preserved in continuations).
        let user_msg = msgs
            .iter()
            .find(|m| m["role"] == "user")
            .and_then(|m| m["content"].as_str())
            .unwrap_or("")
            .to_string();
        let idx: usize = user_msg
            .rsplit('-')
            .next()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        if is_continuation {
            // Continuation: unique, non-looping ending.
            sse_response(vec![
                content_chunk(&format!(" [cont {}]", idx)),
                DONE_CHUNK.to_string(),
            ])
        } else if user_msg.starts_with("LOOP") {
            // Original looping request: a unique 4-char block repeated twice.
            let block = format!("B{:03}", idx);
            sse_response(vec![
                content_chunk(&format!("{}{}", block, block)),
                DONE_CHUNK.to_string(),
            ])
        } else {
            // Non-looping request.
            sse_response(vec![
                content_chunk(&format!("unique-{}", idx)),
                DONE_CHUNK.to_string(),
            ])
        }
    }))
    .await;
    let proxy = start_proxy(test_config(&upstream.addr)).await;

    let mut handles = Vec::new();
    for i in 0..10 {
        let client = proxy.client.clone();
        let addr = proxy.addr.clone();
        let looping = i % 2 == 0;
        let marker = format!("{}-{}", if looping { "LOOP" } else { "PLAIN" }, i);
        handles.push(tokio::spawn(async move {
            let completion = json!({
                "model": "m",
                "messages": [{"role": "user", "content": marker}],
                "stream": true,
            });
            let resp = client
                .post(format!("{}/v1/chat/completions", addr))
                .json(&completion)
                .send()
                .await
                .unwrap();
            assert_eq!(resp.status(), StatusCode::OK);
            let bytes = read_all(resp).await;
            (looping, i, bytes)
        }));
    }

    for h in handles {
        let (looping, i, bytes) = h.await.unwrap();
        assert_eq!(count_done(&bytes), 1, "request {} must end with one [DONE]", i);
        let content = extract_content(&bytes);
        if looping {
            let block = format!("B{:03}", i);
            assert_eq!(
                content,
                format!("{}{} [cont {}]", block, block, i),
                "looping request {} content mismatch",
                i
            );
        } else {
            assert_eq!(content, format!("unique-{}", i), "plain request {} content mismatch", i);
        }
    }

    // 10 originals + 5 continuations (the 5 looping requests).
    assert_eq!(upstream.chat_request_bodies().len(), 15);
}

// D9: usage forwarding — only the final response's usage reaches the client.
#[tokio::test]
async fn d9_usage_forwarding_only_last() {
    let upstream = MockUpstream::start(Arc::new(|body: &Value| {
        let msgs = body["messages"].as_array().cloned().unwrap_or_default();
        let is_continuation = msgs.len() > 1;
        if is_continuation {
            // Continuation: unique ending + its own usage (completion_tokens=200).
            sse_response(vec![
                content_chunk(" (continued)"),
                usage_chunk(50, 200),
                DONE_CHUNK.to_string(),
            ])
        } else {
            // Original: loops, then emits its own usage (completion_tokens=100)
            // and [DONE]. Both must be stripped (the response is discarded).
            sse_response(vec![
                content_chunk("ABCDABCD"),
                usage_chunk(50, 100),
                DONE_CHUNK.to_string(),
            ])
        }
    }))
    .await;
    let proxy = start_proxy(test_config(&upstream.addr)).await;

    let completion = json!({
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": true,
    });
    let resp = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.addr))
        .json(&completion)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = read_all(resp).await;

    // Content is the merged loop + continuation.
    assert_eq!(extract_content(&bytes), "ABCDABCD (continued)");
    assert_eq!(count_done(&bytes), 1);
    // Two upstream requests (original + continuation).
    assert_eq!(upstream.chat_request_bodies().len(), 2);
    // Only ONE usage chunk reaches the client, and it is the continuation's.
    assert_eq!(count_usage(&bytes), 1, "exactly one usage chunk must be forwarded");
    assert_eq!(
        usage_completion_tokens(&bytes),
        Some(200),
        "the forwarded usage must be the final response's"
    );
}

// D10: detection disabled -> pure passthrough, no continuation.
#[tokio::test]
async fn d10_detection_disabled_passthrough() {
    let upstream = MockUpstream::start(Arc::new(|_body: &Value| {
        // A looping stream that WOULD trigger detection if it were enabled.
        sse_response(vec![
            content_chunk("ABCDABCD"),
            DONE_CHUNK.to_string(),
        ])
    }))
    .await;
    let proxy = start_proxy(test_config_detection(&upstream.addr, false)).await;

    let completion = json!({
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": true,
    });
    let resp = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.addr))
        .json(&completion)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = read_all(resp).await;

    // The looping content passes through verbatim, with the upstream's [DONE].
    assert_eq!(extract_content(&bytes), "ABCDABCD");
    assert_eq!(count_done(&bytes), 1);
    // No continuation was issued: exactly one upstream request.
    assert_eq!(upstream.chat_request_bodies().len(), 1);
}

// D11: upstream non-200 -> status code + body passthrough.
#[tokio::test]
async fn d11_upstream_non_200_passthrough() {
    let upstream = MockUpstream::start(Arc::new(|_body: &Value| {
        // Upstream rejects the request (e.g. bad model name), with a
        // retry-after header (as a rate-limited upstream would send).
        let body = serde_json::to_vec(&json!({
            "error": {"message": "model not found", "type": "invalid_request_error"}
        }))
        .unwrap();
        Response::builder()
            .status(StatusCode::BAD_REQUEST)
            .header("content-type", "application/json")
            .header("retry-after", "12")
            .header("x-request-id", "test-123")
            .body(Body::from(body))
            .unwrap()
    }))
    .await;
    let proxy = start_proxy(test_config(&upstream.addr)).await;

    let completion = json!({
        "model": "nope",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": true,
    });
    let resp = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.addr))
        .json(&completion)
        .send()
        .await
        .unwrap();
    // The proxy forwards the upstream status code.
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    // And the upstream content-type / retry-after headers.
    assert_eq!(
        resp.headers().get("content-type").and_then(|v| v.to_str().ok()),
        Some("application/json")
    );
    assert_eq!(
        resp.headers().get("retry-after").and_then(|v| v.to_str().ok()),
        Some("12")
    );
    assert_eq!(
        resp.headers().get("x-request-id").and_then(|v| v.to_str().ok()),
        Some("test-123")
    );
    let bytes = read_all(resp).await;
    // And the upstream error body verbatim.
    let v: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["error"]["message"], "model not found");
    assert_eq!(v["error"]["type"], "invalid_request_error");
    // Exactly one upstream request (no continuation on error).
    assert_eq!(upstream.chat_request_bodies().len(), 1);
}

// ---------------------------------------------------------------------------
// M1: non-streaming multimodal passthrough (request + response)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn m1_non_streaming_multimodal_passthrough() {
    // Upstream response: non-streaming JSON with multimodal content array.
    let completion = json!({
        "id": "cmpl-mm",
        "object": "chat.completion",
        "model": "m",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": [
                    { "type": "text", "text": "图中是一个卡通男孩。" },
                    { "type": "image_url", "image_url": { "url": "data:image/png;base64,RESP=" } }
                ]
            },
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18 }
    });

    let value = completion.clone();
    let mock = MockUpstream::start(Arc::new(move |_body| {
        json_response(StatusCode::OK, &value)
    }))
    .await;
    let proxy = start_proxy(test_config(&mock.addr)).await;

    // Request: user message with multimodal content array (text + image_url).
    let req_body = json!({
        "model": "m",
        "stream": false,
        "messages": [{
            "role": "user",
            "content": [
                { "type": "text", "text": "描述这张图" },
                { "type": "image_url", "image_url": { "url": "data:image/png;base64,REQ=" } }
            ]
        }]
    });

    let resp = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.addr))
        .json(&req_body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = read_all(resp).await;

    // Response body is byte-identical to the upstream's (multimodal intact).
    let expected = serde_json::to_vec(&completion).unwrap();
    assert_eq!(bytes, expected, "non-streaming multimodal body must be byte-identical");

    // The request was forwarded verbatim (multimodal content array intact).
    let bodies = mock.chat_request_bodies();
    assert_eq!(bodies.len(), 1, "exactly one upstream request");
    assert_eq!(bodies[0], req_body, "multimodal request body forwarded verbatim");
}

// ---------------------------------------------------------------------------
// M2: streaming + multimodal request + loop -> continuation preserves
// multimodal messages
// ---------------------------------------------------------------------------

#[tokio::test]
async fn m2_stream_multimodal_request_continuation() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let counter = Arc::new(AtomicUsize::new(0));
    let handler = Arc::new(move |_body: &Value| {
        let n = counter.fetch_add(1, Ordering::SeqCst);
        if n == 0 {
            // First request: the model loops.
            sse_response(vec![
                content_chunk("Hello "),
                content_chunk("ABCDABCD"),
                DONE_CHUNK.to_string(),
            ])
        } else {
            // Continuation: a normal completion.
            sse_response(vec![
                content_chunk(" (continued)"),
                DONE_CHUNK.to_string(),
            ])
        }
    });

    let mock = MockUpstream::start(handler).await;
    let proxy = start_proxy(test_config(&mock.addr)).await;

    // Request: user message with text + image_url + input_audio parts.
    let req_body = json!({
        "model": "m",
        "stream": true,
        "messages": [{
            "role": "user",
            "content": [
                { "type": "text", "text": "看图并听音频" },
                { "type": "image_url", "image_url": { "url": "data:image/png;base64,IMG=" } },
                { "type": "input_audio", "input_audio": { "data": "AUDIO_B64", "format": "wav" } }
            ]
        }]
    });

    let resp = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.addr))
        .json(&req_body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = read_all(resp).await;

    // Merged stream: loop prefix + continuation.
    assert_eq!(extract_content(&bytes), "Hello ABCDABCD (continued)");
    assert_eq!(count_done(&bytes), 1, "merged stream must have a single [DONE]");

    // Two upstream requests: original + one continuation.
    let bodies = mock.chat_request_bodies();
    assert_eq!(bodies.len(), 2, "original + one continuation");

    // The continuation request preserves the original multimodal message.
    let cont_msgs = bodies[1].get("messages").and_then(Value::as_array).cloned().unwrap();
    assert_eq!(cont_msgs.len(), 3);
    assert_eq!(
        cont_msgs[0], req_body["messages"][0],
        "original multimodal message must be preserved in continuation"
    );
    assert_eq!(
        cont_msgs[1],
        json!({ "role": "assistant", "content": "Hello ABCD" })
    );
    assert_eq!(
        cont_msgs[2],
        json!({ "role": "user", "content": "CONTINUE" })
    );
}

// ---------------------------------------------------------------------------
// M3: streaming response with multimodal delta (content parts array) ->
// byte-identical passthrough, no detection
// ---------------------------------------------------------------------------

#[tokio::test]
async fn m3_stream_multimodal_response_passthrough() {
    // Upstream emits a delta whose `content` is an array (content parts),
    // a non-standard multimodal format. The proxy must forward it verbatim
    // and NOT feed it to the loop detector.
    let mm_delta = json!([
        { "type": "text", "text": "看图：" },
        { "type": "image_url", "image_url": { "url": "data:image/png;base64,OUT=" } }
    ]);

    let chunks = vec![
        delta_chunk(&json!({ "role": "assistant" })),
        delta_chunk(&mm_delta),
        DONE_CHUNK.to_string(),
    ];
    let expected: Vec<u8> = chunks.iter().flat_map(|c| c.as_bytes().to_vec()).collect();

    let mock = MockUpstream::start(Arc::new(move |_body: &Value| {
        sse_response(chunks.clone())
    }))
    .await;
    let proxy = start_proxy(test_config(&mock.addr)).await;

    let req_body = json!({
        "model": "m",
        "stream": true,
        "messages": [{ "role": "user", "content": "hi" }]
    });

    let resp = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.addr))
        .json(&req_body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = read_all(resp).await;

    // The response is byte-identical to the upstream's (multimodal delta intact).
    assert_eq!(bytes, expected, "multimodal delta must be forwarded byte-identically");

    // No continuation was issued: exactly one upstream request.
    assert_eq!(mock.chat_request_bodies().len(), 1, "no continuation for multimodal delta");

    // Single [DONE].
    assert_eq!(count_done(&bytes), 1);
}

// ---------------------------------------------------------------------------
// Audit regression tests (BUGs 1-7)
// ---------------------------------------------------------------------------

/// Build a single SSE `data:` chunk carrying `delta.content` **and** a non-null
/// `finish_reason` (the chunk a server sends when the turn completes).
fn finish_chunk(content: &str, reason: &str) -> String {
    let v = json!({
        "id": "x",
        "object": "chat.completion.chunk",
        "choices": [{ "index": 0, "delta": { "content": content }, "finish_reason": reason }]
    });
    format!("data: {v}\n\n")
}

/// Count the `data:` events carrying a non-null `choices[0].finish_reason`.
fn count_finish_reason(bytes: &[u8]) -> usize {
    parse_events(bytes)
        .iter()
        .filter(|e| matches!(e, SseEvent::Data(p) if llama_gag::sse::has_finish_reason(p)))
        .count()
}

/// Concatenate all `delta.reasoning_content` across the SSE events.
fn extract_reasoning(bytes: &[u8]) -> String {
    let mut out = String::new();
    for ev in parse_events(bytes) {
        if let SseEvent::Data(payload) = ev {
            if let Some(r) = llama_gag::sse::delta_reasoning(&payload) {
                out.push_str(&r);
            }
        }
    }
    out
}

// R1 (BUG 3): llama-server emits its `usage` chunk AFTER `[DONE]`. The parser
// must not join it onto `[DONE]`, so the merged stream keeps a single [DONE]
// and the usage chunk is still recognized (and stripped).
#[tokio::test]
async fn r1_usage_after_done_is_separate_event() {
    let counter = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let handler = Arc::new(move |_body: &Value| {
        let n = counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if n == 0 {
            // Loop, then usage, then [DONE], then a second usage AFTER [DONE]
            // (llama-server's late usage chunk), all in one chunk.
            sse_response(vec![format!(
                "{}{}{}{}",
                content_chunk("ABCDABCD"),
                usage_chunk(10, 50),
                DONE_CHUNK,
                usage_chunk(10, 60)
            )])
        } else {
            sse_response(vec![
                content_chunk(" (continued)"),
                usage_chunk(10, 200),
                DONE_CHUNK.to_string(),
            ])
        }
    });
    let mock = MockUpstream::start(handler).await;
    let proxy = start_proxy(test_config(&mock.addr)).await;

    let resp = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.addr))
        .json(&json!({ "model": "m", "stream": true, "messages": [{ "role": "user", "content": "hi" }] }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = read_all(resp).await;

    // Single [DONE]; the pre-[DONE] usage of the discarded response stripped.
    assert_eq!(count_done(&bytes), 1, "exactly one [DONE]");
    assert_eq!(count_usage(&bytes), 1, "only the final response's usage forwarded");
    assert_eq!(
        usage_completion_tokens(&bytes),
        Some(200),
        "the late post-[DONE] usage must not be glued onto [DONE] and forwarded"
    );
    assert_eq!(extract_content(&bytes), "ABCDABCD (continued)");
    assert_eq!(mock.chat_request_bodies().len(), 2);
}

// R2 (BUG 2): a content loop in the THIRD segment (content -> reasoning ->
// content). `kept_chars` is relative to the detector that fired (the third
// segment), so the assistant turn must keep segment 1 in full + the deduped
// segment 3, not slice the whole accumulated content.
#[tokio::test]
async fn r2_content_loop_after_segment_switch() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let first = "IMPORTANT FIRST SEGMENT. "; // 25 chars, no loop
    let third = "THINK ABOUT IT.THINK ABOUT IT."; // 15-char block x2 -> kept 15
    let counter = Arc::new(AtomicUsize::new(0));
    let handler = Arc::new(move |_body: &Value| {
        let n = counter.fetch_add(1, Ordering::SeqCst);
        if n == 0 {
            sse_response(vec![
                content_chunk(first),
                reasoning_chunk("hmm "),
                content_chunk(third),
                DONE_CHUNK.to_string(),
            ])
        } else {
            sse_response(vec![content_chunk(" FINAL"), DONE_CHUNK.to_string()])
        }
    });
    let mock = MockUpstream::start(handler).await;
    let proxy = start_proxy(test_config(&mock.addr)).await;

    let resp = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.addr))
        .json(&json!({ "model": "m", "stream": true, "messages": [{ "role": "user", "content": "hi" }] }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = read_all(resp).await;

    // The client sees everything forwarded from the discarded response plus the
    // continuation (pass-through is unaffected by the fix).
    assert_eq!(
        extract_content(&bytes),
        format!("{first}{third} FINAL"),
        "client content must be the merged stream"
    );
    assert_eq!(extract_reasoning(&bytes), "hmm ");
    assert_eq!(count_done(&bytes), 1);

    let bodies = mock.chat_request_bodies();
    assert_eq!(bodies.len(), 2, "original + one continuation");
    let cont_msgs = bodies[1]["messages"].as_array().cloned().unwrap();
    assert_eq!(cont_msgs.len(), 3);
    // The assistant turn: reasoning segment in full; content = segment 1 (kept
    // in full) + deduped segment 3. Before the fix this sliced the whole
    // accumulated content from char 0 ("IMPO") and lost segment 1.
    assert_eq!(
        cont_msgs[1],
        json!({
            "role": "assistant",
            "reasoning_content": "hmm ",
            "content": format!("{first}THINK ABOUT IT."),
        }),
        "kept content must be segment 1 + deduped segment 3"
    );
    assert_eq!(cont_msgs[2], json!({ "role": "user", "content": "CONTINUE" }));
}

// R2b (BUG 2): a reasoning loop in the SECOND segment, after a content segment.
// The content produced so far must be kept in full (not replaced by ""), and
// the kept reasoning must be the deduped reasoning segment.
#[tokio::test]
async fn r2b_reasoning_loop_after_content() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let content1 = "answer-part-1 "; // 14 chars, no loop
    // "REASONING-BLOCK-ONE-should-be-kept." (35) x2 + "THIN" -> loops at char 70, kept 35.
    let reasoning1 = "REASONING-BLOCK-ONE-should-be-kept.REASONING-BLOCK-ONE-should-be-kept.THIN";
    let counter = Arc::new(AtomicUsize::new(0));
    let handler = Arc::new(move |_body: &Value| {
        let n = counter.fetch_add(1, Ordering::SeqCst);
        if n == 0 {
            sse_response(vec![
                content_chunk(content1),
                reasoning_chunk(reasoning1),
                DONE_CHUNK.to_string(),
            ])
        } else {
            sse_response(vec![content_chunk(" FINAL"), DONE_CHUNK.to_string()])
        }
    });
    let mock = MockUpstream::start(handler).await;
    let proxy = start_proxy(test_config(&mock.addr)).await;

    let resp = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.addr))
        .json(&json!({ "model": "m", "stream": true, "messages": [{ "role": "user", "content": "hi" }] }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = read_all(resp).await;

    assert_eq!(extract_content(&bytes), format!("{content1} FINAL"));
    assert_eq!(extract_reasoning(&bytes), reasoning1);
    assert_eq!(count_done(&bytes), 1);

    let bodies = mock.chat_request_bodies();
    assert_eq!(bodies.len(), 2);
    let cont_msgs = bodies[1]["messages"].as_array().cloned().unwrap();
    // Before the fix: content was hardcoded "" and reasoning was sliced from
    // char 0 ("REAS").
    assert_eq!(
        cont_msgs[1],
        json!({
            "role": "assistant",
            "reasoning_content": "REASONING-BLOCK-ONE-should-be-kept.",
            "content": content1,
        }),
        "reasoning loop must keep all content so far and dedupe only the reasoning segment"
    );
}

// R3 (BUG 1): the discarded looping response's `finish_reason` chunk must be
// stripped, so the client sees exactly one finish_reason (the final one).
#[tokio::test]
async fn r3_finish_reason_of_discarded_response_stripped() {
    let counter = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let handler = Arc::new(move |_body: &Value| {
        let n = counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if n == 0 {
            // The looping response ends with a finish_reason chunk + [DONE].
            sse_response(vec![
                content_chunk("ABCDABCD"),
                finish_chunk("tail", "stop"),
                DONE_CHUNK.to_string(),
            ])
        } else {
            sse_response(vec![
                content_chunk(" FINAL"),
                finish_chunk("", "stop"),
                DONE_CHUNK.to_string(),
            ])
        }
    });
    let mock = MockUpstream::start(handler).await;
    let proxy = start_proxy(test_config(&mock.addr)).await;

    let resp = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.addr))
        .json(&json!({ "model": "m", "stream": true, "messages": [{ "role": "user", "content": "hi" }] }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = read_all(resp).await;

    assert_eq!(extract_content(&bytes), "ABCDABCD FINAL");
    assert_eq!(count_done(&bytes), 1);
    // Before the fix the discarded response's finish_reason was forwarded, so
    // clients (e.g. VS Code) saw a mid-stream "stop" before the continuation.
    assert_eq!(
        count_finish_reason(&bytes),
        1,
        "only the final response's finish_reason may reach the client"
    );
    assert_eq!(mock.chat_request_bodies().len(), 2);
}

// R4 (BUG 4): upstream ends mid-event (partial `data:` line, no [DONE]). The
// proxy's terminal [DONE] must be preceded by an event boundary, so the client
// receives a well-formed terminator.
#[tokio::test]
async fn r4_truncated_mid_event_emits_well_formed_done() {
    let mock = MockUpstream::start(Arc::new(|_body: &Value| {
        // Ends in the middle of a data line (no newline, no blank line).
        sse_response(vec![r#"data: {"choices":[{"index":0,"delta":{"content":"partial answer"},"finish_reason":null}]}"#.to_string()])
    }))
    .await;
    let proxy = start_proxy(test_config(&mock.addr)).await;

    let resp = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.addr))
        .json(&json!({ "model": "m", "stream": true, "messages": [{ "role": "user", "content": "hi" }] }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = read_all(resp).await;

    // The partial event is closed by the boundary and the [DONE] stands alone.
    assert_eq!(
        count_done(&bytes),
        1,
        "the terminal [DONE] must not be glued onto the partial line"
    );
    assert_eq!(extract_content(&bytes), "partial answer");
    assert_eq!(mock.chat_request_bodies().len(), 1, "no continuation (no loop)");
}

// R4b (BUG 4): max_retries exhausted with the last response ending mid-event.
// The [DONE] appended in the exhausted branch must also be boundary-separated.
#[tokio::test]
async fn r4b_max_retries_exhausted_mid_event() {
    // Every response loops (complete event) AND ends mid-event: a truncated
    // `data:` line with no newline, so the parser has a partial line buffered
    // when the proxy appends its terminal [DONE].
    let looping = content_chunk("ABCDABCD");
    let partial = r#"data: {"choices":[{"index":0,"delta":{"content":"tail""#;
    let mock = MockUpstream::start(Arc::new(move |_body: &Value| {
        sse_response(vec![looping.clone(), partial.to_string()])
    }))
    .await;
    let proxy = start_proxy(test_config(&mock.addr)).await;

    let resp = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.addr))
        .json(&json!({ "model": "m", "stream": true, "messages": [{ "role": "user", "content": "hi" }] }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = read_all(resp).await;

    // Original + 3 continuations, all looping.
    assert_eq!(mock.chat_request_bodies().len(), 4);
    assert_eq!(extract_content(&bytes), "ABCDABCD".repeat(4));
    assert_eq!(
        count_done(&bytes),
        1,
        "exactly one well-formed [DONE] after retries exhausted"
    );
}

// R5 (BUG 5): `n > 1` must be passed through untouched — no detection, no
// continuation, byte-identical response.
#[tokio::test]
async fn r5_multi_choice_n_gt_1_passthrough() {
    let counter = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    // A payload that WOULD trigger detection if choices[0] were monitored: both
    // choices repeat a 4-char block twice.
    let loop_choices = |n: usize| -> Vec<Value> {
        (0..n)
            .map(|i| {
                json!({
                    "index": i,
                    "delta": { "content": format!("C{i}XC{i}X") },
                    "finish_reason": null
                })
            })
            .collect()
    };
    let handler = Arc::new(move |body: &Value| {
        let n = counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let n_choices = body.get("n").and_then(Value::as_u64).unwrap_or(1) as usize;
        if n == 0 {
            sse_response(vec![
                delta_chunk(&json!({ "content": "prefix " })),
                delta_chunk(&json!({ "content": "ABCDABCD" })),
                delta_chunk(&json!({ "multi": loop_choices(n_choices) })),
                DONE_CHUNK.to_string(),
            ])
        } else {
            sse_response(vec![content_chunk(" FINAL"), DONE_CHUNK.to_string()])
        }
    });
    let mock = MockUpstream::start(handler).await;
    let proxy = start_proxy(test_config(&mock.addr)).await;

    let raw = r#"{"model":"m","stream":true,"n":2,"messages":[{"role":"user","content":"hi"}]}"#;
    let resp = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.addr))
        .header("content-type", "application/json")
        .body(raw.to_string())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = read_all(resp).await;

    // Byte-identical passthrough of the upstream's stream.
    let expected: Vec<u8> = [
        delta_chunk(&json!({ "content": "prefix " })),
        delta_chunk(&json!({ "content": "ABCDABCD" })),
        delta_chunk(&json!({ "multi": loop_choices(2) })),
        DONE_CHUNK.to_string(),
    ]
    .iter()
    .flat_map(|c| c.as_bytes().to_vec())
    .collect();
    assert_eq!(bytes, expected, "n > 1 stream must be forwarded byte-identically");

    // Exactly one upstream request: no detection, no continuation.
    let bodies = mock.chat_request_bodies();
    assert_eq!(bodies.len(), 1, "n > 1 must never trigger a continuation");
    assert_eq!(bodies[0]["n"], json!(2));
    // The body is forwarded verbatim (passthrough path).
    assert_eq!(
        mock.chat_request_raw_bodies()[0],
        Bytes::from(raw.to_string()),
        "n > 1 body must be forwarded verbatim"
    );
    assert_eq!(count_done(&bytes), 1);
}

// R6 (BUGs 6 + 7): the detection path must not advertise `accept-encoding`
// (the proxy cannot decompress), must keep it on the passthrough path, and
// must forward the client's body bytes verbatim (no JSON re-serialization).
#[tokio::test]
async fn r6_detection_strips_accept_encoding_and_forwards_body_verbatim() {
    let counter = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let handler = Arc::new(move |_body: &Value| {
        let n = counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if n == 0 {
            sse_response(vec![
                content_chunk("ABCDABCD"),
                DONE_CHUNK.to_string(),
            ])
        } else {
            sse_response(vec![content_chunk(" FINAL"), DONE_CHUNK.to_string()])
        }
    });
    let mock = MockUpstream::start(handler).await;
    let proxy = start_proxy(test_config(&mock.addr)).await;

    // Exact bytes, with a key order and number formatting that serde_json
    // re-serialization would not preserve.
    let raw = r#"{"temperature":0.30,"frequency_penalty":0.0,"messages":[{"role":"user","content":"hi"}],"model":"m","stream":true}"#;
    let resp = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.addr))
        .header("content-type", "application/json")
        .header("accept-encoding", "gzip, br, zstd")
        .body(raw.to_string())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = read_all(resp).await;
    assert_eq!(extract_content(&bytes), "ABCDABCD FINAL");
    assert_eq!(count_done(&bytes), 1);

    // BUG 7: the first upstream request carries the client's bytes verbatim.
    let raws = mock.chat_request_raw_bodies();
    assert_eq!(raws.len(), 2, "original + one continuation");
    assert_eq!(
        raws[0],
        Bytes::from(raw.to_string()),
        "detection path must forward the body verbatim (no re-serialization)"
    );

    // BUG 6: neither the first request nor the continuation asks for encoding.
    let hdrs = mock.chat_request_headers();
    assert_eq!(hdrs.len(), 2);
    for (i, h) in hdrs.iter().enumerate() {
        assert!(
            h.get("accept-encoding").is_none(),
            "detection path request {i} must not forward accept-encoding"
        );
    }

    // The passthrough path (non-streaming) keeps accept-encoding: it forwards
    // the upstream response headers verbatim, so content-encoding stays
    // consistent with the body.
    let resp = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.addr))
        .header("content-type", "application/json")
        .header("accept-encoding", "gzip")
        .body(r#"{"model":"m","stream":false,"messages":[]}"#.to_string())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let _ = read_all(resp).await;
    let hdrs = mock.chat_request_headers();
    let last = hdrs.last().unwrap();
    assert_eq!(
        last.get("accept-encoding"),
        Some(&axum::http::HeaderValue::from_static("gzip")),
        "passthrough path must keep accept-encoding"
    );
}


