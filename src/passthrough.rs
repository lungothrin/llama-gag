//! Generic request/response passthrough.
//!
//! Used for every route the proxy does not special-case (e.g. `/v1/models`,
//! `/v1/chat/completions/input_tokens`, `/health`) and for non-streaming
//! `/v1/chat/completions` requests. The request is forwarded to the upstream
//! llama-server and the response is streamed back verbatim.

use axum::body::Body;
use axum::http::{HeaderMap, Method, StatusCode, Uri};
use axum::response::Response;
use bytes::Bytes;
use reqwest::Client;

/// Hop-by-hop and connection-specific headers that must not be forwarded.
const HOP_BY_HOP: &[&str] = &[
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
];

/// True if the header should not be forwarded to the upstream.
pub fn is_hop_by_hop(name: &str) -> bool {
    HOP_BY_HOP.contains(&name.to_ascii_lowercase().as_str())
}

/// Build the upstream URL by joining the base endpoint with the request path
/// and query string.
pub fn build_upstream_url(base: &str, uri: &Uri) -> String {
    let base = base.trim_end_matches('/');
    let path_and_query = uri.path_and_query().map(|pq| pq.as_str()).unwrap_or("");
    format!("{base}{path_and_query}")
}

/// Return a copy of `headers` with hop-by-hop headers removed.
pub fn filter_headers(headers: &HeaderMap) -> HeaderMap {
    let mut out = HeaderMap::new();
    for (name, value) in headers {
        if is_hop_by_hop(name.as_str()) {
            continue;
        }
        out.append(name, value.clone());
    }
    out
}

/// Forward the request to the upstream and stream the response back verbatim.
///
/// Returns `StatusCode::BAD_GATEWAY` if the upstream cannot be reached.
pub async fn passthrough(
    client: &Client,
    base: &str,
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, StatusCode> {
    let url = build_upstream_url(base, &uri);
    let mut req = client.request(method, url);
    let filtered = filter_headers(&headers);
    for (name, value) in filtered.iter() {
        req = req.header(name, value);
    }
    if !body.is_empty() {
        req = req.body(body);
    }

    let resp = req.send().await.map_err(|_| StatusCode::BAD_GATEWAY)?;
    let status = StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    // Forward every upstream response header except hop-by-hop ones
    // (content-length / transfer-encoding are in the list, so hyper
    // re-frames the body).
    let upstream_headers = filter_headers(resp.headers());

    let stream = resp.bytes_stream();
    let body = Body::from_stream(stream);
    let mut response = Response::builder().status(status).body(body).unwrap();
    for (name, value) in upstream_headers.iter() {
        response.headers_mut().append(name, value.clone());
    }
    Ok(response)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::header::{AUTHORIZATION, CONTENT_TYPE, HOST};
    use axum::http::HeaderValue;

    // is_hop_by_hop.
    #[test]
    fn hop_by_hop_detection() {
        assert!(is_hop_by_hop("Host"));
        assert!(is_hop_by_hop("host"));
        assert!(is_hop_by_hop("Connection"));
        assert!(is_hop_by_hop("Content-Length"));
        assert!(!is_hop_by_hop("Authorization"));
        assert!(!is_hop_by_hop("Content-Type"));
    }

    // build_upstream_url.
    #[test]
    fn build_url_with_query() {
        let uri: Uri = "/v1/models?foo=bar".parse().unwrap();
        assert_eq!(build_upstream_url("http://127.0.0.1:8080", &uri), "http://127.0.0.1:8080/v1/models?foo=bar");
    }

    #[test]
    fn build_url_trims_trailing_slash() {
        let uri: Uri = "/health".parse().unwrap();
        assert_eq!(build_upstream_url("http://127.0.0.1:8080/", &uri), "http://127.0.0.1:8080/health");
    }

    #[test]
    fn build_url_no_query() {
        let uri: Uri = "/v1/models".parse().unwrap();
        assert_eq!(build_upstream_url("http://x:1", &uri), "http://x:1/v1/models");
    }

    // filter_headers drops hop-by-hop, keeps the rest.
    #[test]
    fn filter_drops_hop_by_hop() {
        let mut headers = HeaderMap::new();
        headers.insert(HOST, HeaderValue::from_static("example.com"));
        headers.insert(AUTHORIZATION, HeaderValue::from_static("Bearer tok"));
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        let filtered = filter_headers(&headers);
        assert!(filtered.get(HOST).is_none());
        assert!(filtered.get(AUTHORIZATION).is_some());
        assert!(filtered.get(CONTENT_TYPE).is_some());
    }
}
