//! Configuration: built-in defaults <- TOML file (if present) <- CLI arguments.
//!
//! Precedence (highest wins): CLI > TOML file > built-in default.

use std::path::Path;

use clap::Parser;
use serde::Deserialize;

/// A block-marker pair that the proxy auto-closes when a loop is detected
/// inside an unclosed block. `open` is the opening marker, `close` the closing
/// marker appended to balance it.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct BlockMarker {
    pub open: String,
    pub close: String,
}

/// Command-line arguments. Any value provided here overrides the config file.
#[derive(Debug, Clone, Parser)]
#[command(name = "llama-gag", version, about = "Loop-detection proxy for llama-server")]
pub struct Cli {
    /// Path to the TOML configuration file.
    #[arg(long, default_value = "config.toml")]
    pub config: String,

    /// Listen address (host:port).
    #[arg(long)]
    pub listen: Option<String>,

    /// Upstream llama-server endpoint (base URL).
    #[arg(long)]
    pub upstream: Option<String>,

    /// Maximum request body size in bytes (overrides file).
    #[arg(long)]
    pub max_body_size: Option<usize>,

    /// Enable loop detection (overrides file).
    #[arg(long)]
    pub detection: bool,

    /// Disable loop detection (overrides file).
    #[arg(long)]
    pub no_detection: bool,

    /// Minimum repeated-block length in characters.
    #[arg(long)]
    pub min_block: Option<usize>,

    /// Maximum repeated-block length in characters.
    #[arg(long)]
    pub max_block: Option<usize>,

    /// Minimum number of consecutive repeats to trigger.
    #[arg(long)]
    pub min_repeats: Option<usize>,

    /// User message injected to request continuation.
    #[arg(long)]
    pub prompt: Option<String>,

    /// Maximum number of continuation retries.
    #[arg(long)]
    pub max_retries: Option<usize>,

    /// Enable auto-closing of unclosed block markers (overrides file).
    #[arg(long)]
    pub auto_close: bool,

    /// Disable auto-closing of unclosed block markers (overrides file).
    #[arg(long)]
    pub no_auto_close: bool,

    /// Log level: trace, debug, info, warn, or error (overrides file).
    #[arg(long)]
    pub log_level: Option<String>,

    /// Write logs to this file instead of the console (overrides file).
    #[arg(long)]
    pub log_file: Option<String>,
}

/// Fully-resolved runtime configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub listen: String,
    pub upstream: String,
    /// Maximum request body size, in bytes. Requests larger than this are
    /// rejected with HTTP 413 before reaching a handler.
    pub max_body_size: usize,
    pub detection_enabled: bool,
    pub min_block: usize,
    pub max_block: usize,
    pub min_repeats: usize,
    pub continuation_prompt: String,
    pub max_retries: usize,
    /// Whether to auto-close unclosed block markers on continuation.
    pub auto_close_enabled: bool,
    /// The block-marker pairs to auto-close (open -> close).
    pub auto_close_blocks: Vec<BlockMarker>,
    /// Log level (trace/debug/info/warn/error).
    pub log_level: String,
    /// Optional log file; when set, logs go to the file (not the console).
    pub log_file: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            listen: "127.0.0.1:8090".to_string(),
            upstream: "http://127.0.0.1:8080".to_string(),
            max_body_size: 64 * 1024 * 1024,
            detection_enabled: true,
            min_block: 10,
            max_block: 2048,
            min_repeats: 3,
            continuation_prompt: "Continue from where you left off. Do not repeat any content already generated."
                .to_string(),
            max_retries: 3,
            auto_close_enabled: true,
            // No block markers by default: auto-close is a no-op until the user
            // configures `auto_close.blocks` (e.g. the real think-block markers).
            auto_close_blocks: Vec::new(),
            log_level: "info".to_string(),
            log_file: None,
        }
    }
}

impl Config {
    /// Validate invariants; return an error message if violated.
    pub fn validate(&self) -> Result<(), String> {
        if self.min_block == 0 {
            return Err("detection.min_block must be >= 1".into());
        }
        if self.max_block < self.min_block {
            return Err("detection.max_block must be >= detection.min_block".into());
        }
        if self.min_repeats == 0 {
            return Err("detection.min_repeats must be >= 1".into());
        }
        if self.max_retries == 0 {
            return Err("continuation.max_retries must be >= 1".into());
        }
        for (i, b) in self.auto_close_blocks.iter().enumerate() {
            if b.open.is_empty() || b.close.is_empty() {
                return Err(format!("auto_close.blocks[{i}] open and close must be non-empty"));
            }
        }
        Ok(())
    }
}

// ---- File (TOML) schema. All fields optional so partial files are valid. ----

#[derive(Debug, Default, Deserialize)]
struct FileConfig {
    #[serde(default)]
    server: ServerSection,
    #[serde(default)]
    upstream: UpstreamSection,
    #[serde(default)]
    detection: DetectionSection,
    #[serde(default)]
    continuation: ContinuationSection,
    #[serde(default)]
    auto_close: AutoCloseSection,
    #[serde(default)]
    logging: LoggingSection,
}

#[derive(Debug, Default, Deserialize)]
struct ServerSection {
    listen: Option<String>,
    max_body_size: Option<usize>,
}

#[derive(Debug, Default, Deserialize)]
struct UpstreamSection {
    endpoint: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct DetectionSection {
    enabled: Option<bool>,
    min_block: Option<usize>,
    max_block: Option<usize>,
    min_repeats: Option<usize>,
}

#[derive(Debug, Default, Deserialize)]
struct ContinuationSection {
    prompt: Option<String>,
    max_retries: Option<usize>,
}

#[derive(Debug, Default, Deserialize)]
struct AutoCloseSection {
    enabled: Option<bool>,
    blocks: Option<Vec<BlockMarker>>,
}

#[derive(Debug, Default, Deserialize)]
struct LoggingSection {
    level: Option<String>,
    file: Option<String>,
}

/// Load configuration: built-in defaults <- TOML file (if present) <- CLI args.
pub fn load(cli: &Cli) -> Result<Config, String> {
    let mut cfg = Config::default();

    // 1) TOML file (optional; a missing file is not an error).
    let path = Path::new(&cli.config);
    if path.exists() {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("failed to read config file {}: {e}", path.display()))?;
        let file: FileConfig = toml::from_str(&content)
            .map_err(|e| format!("failed to parse config file {}: {e}", path.display()))?;
        if let Some(v) = file.server.listen {
            cfg.listen = v;
        }
        if let Some(v) = file.server.max_body_size {
            cfg.max_body_size = v;
        }
        if let Some(v) = file.upstream.endpoint {
            cfg.upstream = v;
        }
        if let Some(v) = file.detection.enabled {
            cfg.detection_enabled = v;
        }
        if let Some(v) = file.detection.min_block {
            cfg.min_block = v;
        }
        if let Some(v) = file.detection.max_block {
            cfg.max_block = v;
        }
        if let Some(v) = file.detection.min_repeats {
            cfg.min_repeats = v;
        }
        if let Some(v) = file.continuation.prompt {
            cfg.continuation_prompt = v;
        }
        if let Some(v) = file.continuation.max_retries {
            cfg.max_retries = v;
        }
        if let Some(v) = file.auto_close.enabled {
            cfg.auto_close_enabled = v;
        }
        if let Some(v) = file.auto_close.blocks {
            cfg.auto_close_blocks = v;
        }
        if let Some(v) = file.logging.level {
            cfg.log_level = v;
        }
        if let Some(v) = file.logging.file {
            cfg.log_file = Some(v);
        }
    }

    // 2) CLI overrides (highest precedence).
    if let Some(v) = &cli.listen {
        cfg.listen = v.clone();
    }
    if let Some(v) = &cli.upstream {
        cfg.upstream = v.clone();
    }
    if let Some(v) = cli.max_body_size {
        cfg.max_body_size = v;
    }
    if cli.no_detection {
        cfg.detection_enabled = false;
    } else if cli.detection {
        cfg.detection_enabled = true;
    }
    if let Some(v) = cli.min_block {
        cfg.min_block = v;
    }
    if let Some(v) = cli.max_block {
        cfg.max_block = v;
    }
    if let Some(v) = cli.min_repeats {
        cfg.min_repeats = v;
    }
    if let Some(v) = &cli.prompt {
        cfg.continuation_prompt = v.clone();
    }
    if let Some(v) = cli.max_retries {
        cfg.max_retries = v;
    }
    if cli.no_auto_close {
        cfg.auto_close_enabled = false;
    } else if cli.auto_close {
        cfg.auto_close_enabled = true;
    }
    if let Some(v) = &cli.log_level {
        cfg.log_level = v.clone();
    }
    if let Some(v) = &cli.log_file {
        cfg.log_file = Some(v.clone());
    }

    // 3) Validate.
    cfg.validate()?;
    Ok(cfg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static COUNTER: AtomicUsize = AtomicUsize::new(0);

    fn cli_with_config(path: &str) -> Cli {
        Cli {
            config: path.to_string(),
            listen: None,
            upstream: None,
            max_body_size: None,
            detection: false,
            no_detection: false,
            min_block: None,
            max_block: None,
            min_repeats: None,
            prompt: None,
            max_retries: None,
            auto_close: false,
            no_auto_close: false,
            log_level: None,
            log_file: None,
        }
    }

    /// CLI pointing at a path that does not exist -> pure built-in defaults.
    fn default_cli() -> Cli {
        cli_with_config(&format!("/nonexistent-llama-gag-{}.toml", std::process::id()))
    }

    fn write_temp_toml(content: &str) -> String {
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let p = std::env::temp_dir().join(format!("llama-gag-cfg-{}-{n}.toml", std::process::id()));
        std::fs::write(&p, content).unwrap();
        p.to_string_lossy().to_string()
    }

    // C1: all defaults (no file, no CLI).
    #[test]
    fn c1_all_defaults() {
        let cfg = load(&default_cli()).unwrap();
        assert_eq!(cfg.listen, "127.0.0.1:8090");
        assert_eq!(cfg.upstream, "http://127.0.0.1:8080");
        assert!(cfg.detection_enabled);
        assert_eq!(cfg.min_block, 10);
        assert_eq!(cfg.max_block, 2048);
        assert_eq!(cfg.min_repeats, 3);
        assert_eq!(cfg.max_retries, 3);
        assert!(!cfg.continuation_prompt.is_empty());
    }

    // C2: TOML only.
    #[test]
    fn c2_toml_only() {
        let p = write_temp_toml(
            r#"
[server]
listen = "0.0.0.0:9999"
[upstream]
endpoint = "http://10.0.0.5:1234"
[detection]
enabled = false
min_block = 5
max_block = 500
min_repeats = 3
[continuation]
prompt = "go on"
max_retries = 7
"#,
        );
        let cfg = load(&cli_with_config(&p)).unwrap();
        assert_eq!(cfg.listen, "0.0.0.0:9999");
        assert_eq!(cfg.upstream, "http://10.0.0.5:1234");
        assert!(!cfg.detection_enabled);
        assert_eq!(cfg.min_block, 5);
        assert_eq!(cfg.max_block, 500);
        assert_eq!(cfg.min_repeats, 3);
        assert_eq!(cfg.continuation_prompt, "go on");
        assert_eq!(cfg.max_retries, 7);
    }

    // C3: CLI only.
    #[test]
    fn c3_cli_only() {
        let mut cli = default_cli();
        cli.listen = Some("1.2.3.4:1".into());
        cli.upstream = Some("http://x:2".into());
        cli.min_block = Some(3);
        cli.max_block = Some(99);
        cli.min_repeats = Some(4);
        cli.max_retries = Some(9);
        cli.prompt = Some("keep going".into());
        let cfg = load(&cli).unwrap();
        assert_eq!(cfg.listen, "1.2.3.4:1");
        assert_eq!(cfg.upstream, "http://x:2");
        assert_eq!(cfg.min_block, 3);
        assert_eq!(cfg.max_block, 99);
        assert_eq!(cfg.min_repeats, 4);
        assert_eq!(cfg.max_retries, 9);
        assert_eq!(cfg.continuation_prompt, "keep going");
    }

    // C4: CLI wins over TOML.
    #[test]
    fn c4_cli_wins_over_toml() {
        let p = write_temp_toml(
            r#"
[server]
listen = "0.0.0.0:9999"
[detection]
min_block = 5
max_block = 500
"#,
        );
        let mut cli = cli_with_config(&p);
        cli.listen = Some("9.9.9.9:8".into());
        cli.min_block = Some(42);
        let cfg = load(&cli).unwrap();
        assert_eq!(cfg.listen, "9.9.9.9:8"); // CLI wins
        assert_eq!(cfg.min_block, 42); // CLI wins
        assert_eq!(cfg.max_block, 500); // from TOML (no CLI override)
    }

    // C5: --no-detection overrides file's enabled=true.
    #[test]
    fn c5_no_detection() {
        let p = write_temp_toml("[detection]\nenabled = true\n");
        let mut cli = cli_with_config(&p);
        cli.no_detection = true;
        let cfg = load(&cli).unwrap();
        assert!(!cfg.detection_enabled);
    }

    // C6: missing config file -> defaults, no error.
    #[test]
    fn c6_missing_file_defaults() {
        let cfg = load(&cli_with_config("/no/such/file.toml")).unwrap();
        assert_eq!(cfg, Config::default());
    }

    // C7: TOML syntax error -> clear error message.
    #[test]
    fn c7_toml_syntax_error() {
        let p = write_temp_toml("listen = \n");
        let err = load(&cli_with_config(&p)).unwrap_err();
        assert!(err.contains("failed to parse config file"), "got: {err}");
    }

    // C8: auto_close defaults (enabled, no block markers).
    #[test]
    fn c8_auto_close_defaults() {
        let cfg = load(&default_cli()).unwrap();
        assert!(cfg.auto_close_enabled);
        assert!(cfg.auto_close_blocks.is_empty());
    }

    // C9: TOML auto_close section (enabled + custom blocks).
    #[test]
    fn c9_auto_close_toml() {
        let p = write_temp_toml(
            r#"
[auto_close]
enabled = false
blocks = [
    { open = "A", close = "B" },
    { open = "C", close = "D" },
]
"#,
        );
        let cfg = load(&cli_with_config(&p)).unwrap();
        assert!(!cfg.auto_close_enabled);
        assert_eq!(cfg.auto_close_blocks.len(), 2);
        assert_eq!(cfg.auto_close_blocks[0].open, "A");
        assert_eq!(cfg.auto_close_blocks[0].close, "B");
        assert_eq!(cfg.auto_close_blocks[1].open, "C");
        assert_eq!(cfg.auto_close_blocks[1].close, "D");
    }

    // C10: --no-auto-close overrides file's enabled=true.
    #[test]
    fn c10_no_auto_close_cli() {
        let p = write_temp_toml("[auto_close]\nenabled = true\n");
        let mut cli = cli_with_config(&p);
        cli.no_auto_close = true;
        let cfg = load(&cli).unwrap();
        assert!(!cfg.auto_close_enabled);
    }

    // C11: empty open/close marker -> validation error.
    #[test]
    fn c11_auto_close_empty_marker_invalid() {
        let p = write_temp_toml(
            r#"
[auto_close]
blocks = [ { open = "", close = "B" } ]
"#,
        );
        let err = load(&cli_with_config(&p)).unwrap_err();
        assert!(err.contains("auto_close.blocks"), "got: {err}");
    }

    // C12: logging defaults + TOML + CLI override.
    #[test]
    fn c12_logging_config() {
        // Defaults: info level, no file.
        let cfg = load(&default_cli()).unwrap();
        assert_eq!(cfg.log_level, "info");
        assert!(cfg.log_file.is_none());

        // TOML [logging] section.
        let p = write_temp_toml(
            r#"
[logging]
level = "debug"
file = "/tmp/llama-gag.log"
"#,
        );
        let cfg = load(&cli_with_config(&p)).unwrap();
        assert_eq!(cfg.log_level, "debug");
        assert_eq!(cfg.log_file.as_deref(), Some("/tmp/llama-gag.log"));

        // CLI overrides TOML.
        let mut cli = cli_with_config(&p);
        cli.log_level = Some("trace".to_string());
        cli.log_file = Some("/tmp/other.log".to_string());
        let cfg = load(&cli).unwrap();
        assert_eq!(cfg.log_level, "trace");
        assert_eq!(cfg.log_file.as_deref(), Some("/tmp/other.log"));
    }

    // C13: max_body_size default, TOML, and CLI override.
    #[test]
    fn c13_max_body_size() {
        // Default: 64 MiB.
        let cfg = load(&default_cli()).unwrap();
        assert_eq!(cfg.max_body_size, 64 * 1024 * 1024);

        // TOML [server] section.
        let p = write_temp_toml("[server]\nmax_body_size = 123456\n");
        let cfg = load(&cli_with_config(&p)).unwrap();
        assert_eq!(cfg.max_body_size, 123456);

        // CLI overrides TOML.
        let mut cli = cli_with_config(&p);
        cli.max_body_size = Some(999);
        let cfg = load(&cli).unwrap();
        assert_eq!(cfg.max_body_size, 999);
    }
}
