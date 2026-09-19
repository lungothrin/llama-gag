//! Route dispatch.
//!
//! - `POST /v1/chat/completions` (exact match) -> [`handle_chat_completions`]
//!   (streaming detection + continuation, or pure passthrough for
//!   non-streaming / detection-disabled requests).
//! - Everything else (any method, any other path, including
//!   `/v1/chat/completions/input_tokens`) -> pure passthrough.
//!
//! Axum routes are exact matches, so `/v1/chat/completions/input_tokens`
//! does NOT match the completions route and falls through to the fallback.

use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, Method, Uri};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::Router;
use reqwest::Client;

use crate::config::Config;
use crate::passthrough;
use crate::proxy::handle_chat_completions;

/// Shared application state, passed to every handler via `State<AppState>`.
#[derive(Clone)]
pub struct AppState {
    pub client: Client,
    pub cfg: Config,
}

/// Build the application router.
pub fn router(client: Client, cfg: Config) -> Router {
    let state = AppState { client, cfg };
    Router::new()
        .route("/v1/chat/completions", post(handle_chat_completions))
        .fallback(fallback)
        .with_state(state)
}

/// Pure passthrough for every route that is not the completions endpoint.
async fn fallback(
    State(app): State<AppState>,
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    tracing::debug!(
        method = %method,
        path = %uri.path(),
        "passthrough (non-completions route)"
    );
    passthrough::passthrough(&app.client, &app.cfg.upstream, method, uri, headers, body)
        .await
        .into_response()
}
