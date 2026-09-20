//! llama-gag binary entry point.
//!
//! Parses CLI args, loads configuration (defaults <- TOML <- CLI), validates it,
//! and starts the axum proxy server.

use std::net::SocketAddr;

use clap::Parser;
use reqwest::Client;
use tracing_subscriber::EnvFilter;

use llama_gag::config::{self, Cli};
use llama_gag::server::router;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let cfg = config::load(&cli)?;
    if let Err(e) = cfg.validate() {
        eprintln!("configuration error: {e}");
        std::process::exit(2);
    }

    // Set up logging from config: a level filter plus an optional log file.
    // When a log file is configured, logs go ONLY to the file (not the console).
    // The non-blocking worker guard must stay alive for the process lifetime so
    // buffered lines are flushed on exit.
    let filter = EnvFilter::new(&cfg.log_level);
    let _file_guard = match &cfg.log_file {
        Some(path) => {
            let file = std::fs::File::create(path)?;
            let (non_blocking, guard) = tracing_appender::non_blocking(file);
            tracing_subscriber::fmt()
                .with_env_filter(filter)
                .with_ansi(false)
                .with_writer(non_blocking)
                .init();
            Some(guard)
        }
        None => {
            tracing_subscriber::fmt().with_env_filter(filter).init();
            None
        }
    };

    tracing::info!(
        listen = %cfg.listen,
        upstream = %cfg.upstream,
        max_body_size = cfg.max_body_size,
        detection = cfg.detection_enabled,
        min_block = cfg.min_block,
        max_block = cfg.max_block,
        min_repeats = cfg.min_repeats,
        max_retries = cfg.max_retries,
        auto_close = cfg.auto_close_enabled,
        auto_close_blocks = cfg.auto_close_blocks.len(),
        log_level = %cfg.log_level,
        log_file = ?cfg.log_file,
        "starting llama-gag"
    );

    let client = Client::builder()
        // No global timeout: streaming responses are long-lived. Per-request
        // timeouts are not appropriate for an LLM proxy.
        .build()?;

    let app = router(client, cfg.clone());

    let addr: SocketAddr = cfg
        .listen
        .parse()
        .map_err(|e| format!("invalid listen address {:?}: {e}", cfg.listen))?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(%addr, "listening");

    axum::serve(listener, app).await?;
    Ok(())
}
