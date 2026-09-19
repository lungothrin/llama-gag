# llama-gag

A loop-detection proxy for [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server`. It intercepts streaming `POST /v1/chat/completions` requests, detects repeated-content loops **incrementally** as tokens stream in, transparently issues continuation requests, and merges everything into one clean SSE stream.

[English](README.md) | [中文](README.zh-CN.md)

## Problems it solves

Local LLMs can get stuck in **repetition loops**: the same sentence or code block over and over, or a `think` block that never closes. Existing mitigations fall short:

- `repetition_penalty` is only a soft logit bias — it reduces loop frequency but **cannot guarantee** recovery.
- Otherwise the user has to cancel the request and retry manually.

**llama-gag** fixes this at the proxy layer:

- **Incremental loop detection** with zero added latency — content is passed through to the client as it arrives.
- **Transparent continuation** — when a loop is detected, the looping response is dropped and a continuation request is issued automatically (deduplicated context + a continuation prompt). The client sees one continuous stream.
- **Auto-closing of unclosed block markers** (e.g. `think`) — thinking content is preserved; only the closing marker is appended.
- **Fully transparent to VS Code** — point the built-in Custom Endpoint at the proxy. No plugin required.

Non-streaming requests and all other routes (`/v1/models`, `/health`, ...) are pure pass-through.

## How to use

### 1. Build

Requires a Rust toolchain (cargo).

```bash
cargo build --release
```

### 2. Run

Assuming `llama-server` is running on port 8080:

```bash
./target/release/llama-gag
# listens on 127.0.0.1:8090, proxies to http://127.0.0.1:8080 by default
```

The proxy is ready when the log shows `listening addr=127.0.0.1:8090`.

### 3. Point VS Code at the proxy

In VS Code's Custom Endpoint settings:

- **URL**: `http://127.0.0.1:8090/v1/chat/completions`
- **Streaming**: enabled
- Token limits (`maxInputTokens`, `maxOutputTokens`, ...): set to your model's context window.

Everything else works exactly as with a direct llama-server connection.

### 4. Configuration

Precedence (highest wins): **CLI > `config.toml` > built-in defaults**.

A full example is provided in [`config.toml`](config.toml):

```toml
[server]
listen = "127.0.0.1:8090"

[upstream]
endpoint = "http://127.0.0.1:8080"

[detection]
enabled = true
min_block = 10      # minimum repeated-block length, in characters
max_block = 2048    # maximum repeated-block length, in characters
min_repeats = 2     # consecutive repeats needed to trigger

[continuation]
prompt = "Continue from where you left off. Do not repeat any content already generated."
max_retries = 3     # max continuation requests before giving up

[auto_close]
enabled = true
blocks = [
    { open = "think", close = "/think" },
]

[logging]
level = "info"      # trace | debug | info | warn | error
# file = "/var/log/llama-gag.log"   # when set, logs go to the file only (no console)
```

### 5. CLI options

| Option | Description | Default |
| --- | --- | --- |
| `--config <PATH>` | Path to the TOML config file | `config.toml` |
| `--listen <ADDR>` | Listen address (host:port) | `127.0.0.1:8090` |
| `--upstream <URL>` | Upstream llama-server endpoint | `http://127.0.0.1:8080` |
| `--detection` / `--no-detection` | Enable / disable loop detection | enabled |
| `--min-block <N>` | Minimum repeated-block length (chars) | `10` |
| `--max-block <N>` | Maximum repeated-block length (chars) | `2048` |
| `--min-repeats <N>` | Consecutive repeats needed to trigger | `2` |
| `--prompt <TEXT>` | Continuation prompt | see `config.toml` |
| `--max-retries <N>` | Max continuation retries | `3` |
| `--auto-close` / `--no-auto-close` | Enable / disable auto-closing block markers | enabled |
| `--log-level <LEVEL>` | `trace` \| `debug` \| `info` \| `warn` \| `error` | `info` |
| `--log-file <PATH>` | Write logs to file (suppresses console) | console |

Run `llama-gag --help` for the full list.

### 6. Verify

```bash
curl http://127.0.0.1:8090/health   # passed through to the upstream
```

### 7. Logs

Every request gets a short id (`req-1`, `req-2`, ...) that appears in all of its log lines, so concurrent requests can be correlated. Key events:

- `info` — request received, loop detected (block length, repeats, kept chars), stream ended (reason + continuation count)
- `debug` — pass-through decisions, continuation requests, auto-close injections
- `warn` — upstream errors, non-2xx responses

To troubleshoot, run with `--log-level debug` (optionally `--log-file /tmp/llama-gag.log`).

## Tests

```bash
cargo test   # 69 tests: 56 unit + 13 integration
```
