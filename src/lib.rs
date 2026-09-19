//! llama-gag: loop-detection proxy for llama-server.
//!
//! The library exposes the proxy's building blocks so integration tests can
//! drive them directly with a mock upstream.

pub mod config;
pub mod continuation;
pub mod detector;
pub mod passthrough;
pub mod proxy;
pub mod server;
pub mod sse;
