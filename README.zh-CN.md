# llama-gag

面向 [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server` 的循环检测代理。拦截流式 `POST /v1/chat/completions` 请求，在 token 流式返回的过程中**增量检测**重复内容（死循环），透明地发起续写请求，并把所有内容合并成一条干净的 SSE 流。

[English](README.md) | [中文](README.zh-CN.md)

## 主要解决的问题

本地 LLM 可能陷入**重复循环**：反复输出同一句话或同一段代码，甚至卡在未闭合的 `think` 块里出不来。现有手段都不够：

- `repetition_penalty` 只是软性 logit 偏置——能降低循环频率，但**无法保证**恢复；
- 否则只能由用户手动取消请求、重新提问。

**llama-gag** 在代理层解决这个问题：

- **增量循环检测**，零额外延迟——内容到达即透传给客户端；
- **透明续写**——检测到循环后丢弃当前响应，自动发起续写请求（去重后的上下文 + 续写提示），客户端看到的是一条连续的流；
- **自动闭合未闭合的块标记**（如 `think`）——保留思考内容，只补闭合标记；
- **对 VS Code 完全透明**——把内置 Custom Endpoint 指向代理即可，无需插件。

非流式请求和其他路由（`/v1/models`、`/health` 等）均为纯透传。

## 如何使用

### 1. 构建

需要 Rust 工具链（cargo）。

```bash
cargo build --release
```

### 2. 启动

假设 `llama-server` 运行在 8080 端口：

```bash
./target/release/llama-gag
# 默认监听 127.0.0.1:8090，上游为 http://127.0.0.1:8080
```

日志出现 `listening addr=127.0.0.1:8090` 即启动成功。

### 3. 在 VS Code 中指向代理

在 VS Code 的 Custom Endpoint 设置中：

- **URL**：`http://127.0.0.1:8090/v1/chat/completions`
- **Streaming**：开启
- Token 上限（`maxInputTokens`、`maxOutputTokens` 等）：按模型上下文窗口设置。

其余行为与直连 llama-server 完全一致。

### 4. 配置

优先级（高者胜）：**CLI > `config.toml` > 内置默认值**。

完整示例见 [`config.toml`](config.toml)：

```toml
[server]
listen = "127.0.0.1:8090"

[upstream]
endpoint = "http://127.0.0.1:8080"

[detection]
enabled = true
min_block = 10      # 最小循环块长度（字符）
max_block = 2048    # 最大循环块长度（字符）
min_repeats = 2     # 触发所需的连续重复次数

[continuation]
prompt = "Continue from where you left off. Do not repeat any content already generated."
max_retries = 3     # 最大续写次数

[auto_close]
enabled = true
blocks = [
    { open = "think", close = "/think" },
]

[logging]
level = "info"      # trace | debug | info | warn | error
# file = "/var/log/llama-gag.log"   # 设置后日志只写文件（不输出控制台）
```

### 5. CLI 参数

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--config <PATH>` | TOML 配置文件路径 | `config.toml` |
| `--listen <ADDR>` | 监听地址（host:port） | `127.0.0.1:8090` |
| `--upstream <URL>` | 上游 llama-server 地址 | `http://127.0.0.1:8080` |
| `--detection` / `--no-detection` | 开启 / 关闭循环检测 | 开启 |
| `--min-block <N>` | 最小循环块长度（字符） | `10` |
| `--max-block <N>` | 最大循环块长度（字符） | `2048` |
| `--min-repeats <N>` | 触发所需的连续重复次数 | `2` |
| `--prompt <TEXT>` | 续写提示词 | 见 `config.toml` |
| `--max-retries <N>` | 最大续写次数 | `3` |
| `--auto-close` / `--no-auto-close` | 开启 / 关闭块标记自动闭合 | 开启 |
| `--log-level <LEVEL>` | `trace` \| `debug` \| `info` \| `warn` \| `error` | `info` |
| `--log-file <PATH>` | 日志写入文件（不再输出控制台） | 控制台 |

完整列表见 `llama-gag --help`。

### 6. 验证

```bash
curl http://127.0.0.1:8090/health   # 透传到上游
```

### 7. 日志

每个请求都有一个短 ID（`req-1`、`req-2`、...），出现在该请求的所有日志行中，便于关联并发请求。关键事件：

- `info` — 请求接收、循环检测命中（块长度、重复次数、保留字符数）、流结束（原因 + 续写次数）
- `debug` — 透传决策、续写请求、自动闭合注入
- `warn` — 上游错误、非 2xx 响应

排查问题时可用 `--log-level debug`（可配合 `--log-file /tmp/llama-gag.log`）。

## 测试

```bash
cargo test   # 69 个测试：56 单元 + 13 集成
```
