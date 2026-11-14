# 09 — 在线 OpenAI HTTP 服务演进与 Nebula 跨仓联调开发计划（对应 Issue `#7`）

> **状态**：Slice Z1–Z4 已实现（`zs-tokenizer`、`zs-scheduler::serve_loop`、`zs-server::serve`，支持 `zstep serve` / `zs-server serve`）  
> **需求基线**：`EeroEternal/zstep#7`（从 Benchmark CLI 演进为可长期运行的 OpenAI HTTP 服务）  
> **对端契约**：`EeroEternal/nebula#92` 及 `docs/dev/zstep.md` / `docs/dev/zstep-plan.md`（`engine_type = "zstep"`）

---

## 1. 核心架构红线与防冲突隔离规则（必读）

1. **与算子/内核性能优化线物理隔离（零合并冲突）**：
   - 在线服务能力必须写在**新增模块或新 crate** 中：
     - 分词与流式停止器：新增 `crates/zs-tokenizer`（或 `crates/zs-server/src/tokenizer.rs`）
     - 常驻在线调度循环：新增 `crates/zs-scheduler/src/serve_loop.rs`
     - HTTP/SSE 与运维探针：新增 `crates/zs-server/src/serve.rs`（子命令 `zs-server serve`）
   - **严禁重写或扰动** [`worker/zstep_worker/kernels_sm120/`](../worker/zstep_worker/kernels_sm120/)、[`deepseek_v41.py`](../worker/zstep_worker/models/deepseek_v41.py)、[`qwen35_moe.py`](../worker/zstep_worker/models/qwen35_moe.py) 的热路径算子代码（P2 阶段扩展 GPU 采样前保持只读），也**严禁破坏** [`serving.rs`](../crates/zs-scheduler/src/serving.rs) 与 [`main.rs`](../crates/zs-server/src/main.rs) 现有的 `--bench-tiers` / `--verify-greedy-json` 离线评测路径。
2. **Zone A/B vs Zone C 职责边界**：
   - `zstep` 只做单节点完整 TP 引擎黑盒（HTTP/SSE $\rightarrow$ Tokenizer/状态机/有界队列 $\rightarrow$ `ServeLoop` $\rightarrow$ `SHM + eventfd` $\rightarrow$ 单机 GPU Workers），不在 `zstep` 内造多机网关或跨实例路由。
   - 多副本/多机路由、鉴权、配额、扩缩容与故障摘流全部由上层 **Nebula** 负责。

---

## 2. 必须严格遵守的 Nebula 对接契约（4 条防踩坑硬约束）

### 2.1 `zs-server serve` CLI 参数契约

| CLI 参数 | 要求与语义 |
| :--- | :--- |
| `--model-dir <PATH>` | 模型权重与 `tokenizer.json` 目录 |
| `--served-model-name <ID>` | `/v1/models` 返回的 `data[0].id` 及推理请求匹配的模型名（缺省取目录名） |
| `--host <IP>` / `--port <PORT>` | 默认支持 `--host 0.0.0.0`（容器/Pod 必传）与指定监听端口 |
| `--world-size <N>` | GPU TP Worker 数（`1..=8`） |
| `--gpu-memory-utilization <F>` | `(0.0, 1.0]`，启动期自动换算物理 `--num-kv-pages`（显式传 `--num-kv-pages` 时以显式值为准） |
| `--max-model-len <N>` | 最大上下文总长（换算 `--max-pages-per-seq` 并在准入期拦截超长请求） |
| `--max-seqs <N>` | 最大并发槽位数（`1..=128`，对齐 [`MAX_DECODE_GRAPH_SEQS`](../crates/zs-wire/src/lib.rs)） |
| `--max-step-tokens <N>` | 单步最大 Token 容量（`ShmGeometry.max_step_tokens`） |
| `--max-prefill-chunk <N>` | Chunked Prefill 块上限（默认 `2048`） |
| `--page-block-tokens <N>` | 物理 KV 页大小（默认 `16`） |
| `--tokenizer-mode <MODE>` | `auto` / `deepseek_v41`（强制走专用 `encode_messages`） |
| `--max-waiting-requests <N>` | HTTP/调度有界等待队列上限（满载返 `429`） |
| `--drain-timeout-secs <N>` | 收到 `SIGTERM` 后等待在途请求完成的最大秒数 |

### 2.2 四态探针 HTTP 状态码矩阵（与 Nebula `wait_engine_ready` + `heartbeat` 对齐）

> **陷阱警告**：Nebula `wait_engine_ready` 只要看到 `GET /health` **或** `GET /v1/models` 返回 `200` 就立即判定引擎就绪；且 Nebula `heartbeat` **仅在 `GET /health == 200` 时才抓取 `/metrics` 更新 `pending_requests`**（若 `Draining` 期 `/health` 返 `503`，会导致排空死等超时并触发误重启）。

| 状态 | 含义 | `GET /livez` | `GET /health` | `GET /ready` | `GET /v1/models` | `POST /v1/chat/completions` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`Starting`** | 加载权重/`b12x`、拉起 Worker、Warmup & Graph Capture 未完成 | `200` | **`503`** | `503` | **`503`** | `503` (`engine_starting`) |
| **`Ready`** | Warmup 完成且 Worker 全部健康 | `200` | **`200`** | **`200`** | **`200`** | 正常服务（满载返 **`429`**） |
| **`Draining`** | 收到 `SIGTERM` 或 `POST /admin/drain`，排空存量请求 | `200` | **`200`** `{"status":"draining"}` | **`503`** | **`503`** | **`503`** (`engine_draining`) |
| **`Faulted`** | 任一 Worker 退出、`WORKER_STATUS_ERROR`、心跳超时或 SHM abort | `200` | **`503`** `{"status":"faulted"}` | `503` | `503` | **`503`** (`engine_faulted`) |

### 2.3 错误码、请求 ID 与断连取消契约

1. **过载返 `429`，绝不返 `503`**：Nebula Gateway 对所有上游 `5xx` 累加端点熔断计数并消耗 `5xx` 错误预算。因此等待队列满、`SlotPool` 满或 `BlockPool` 不足时**必须返回 `429 Too Many Requests`**（`{"error":{"type":"rate_limit_error","code":"engine_overloaded",...}}`）；未声明支持的参数显式返 `400`（禁止静默忽略）。
2. **请求 ID 提取顺序**：优先读取 `x-nebula-request-id` $\rightarrow$ 次选 `x-request-id` $\rightarrow$ 缺省生成 `chatcmpl-<uuid>`，回写响应 `id` 并打入结构化日志。
3. **安全 Step 边界取消**：客户端断开 SSE 连接或慢客户端输出通道满超时后，标记请求 `Cancelled`；`ServeLoop` 在当前 in-flight Step 收敛后移除该序列，释放 `SlotLease` + `BlockLease`，重置 `shadow_page_table` 对应行回 `dummy_page` 并置位 `dirty_pt_slots[0..2]`，`data: [DONE]\n\n` 至多发一次。

### 2.4 Prometheus `GET /metrics` 字段契约

必须暴露以下精确指标名供 Nebula `parse_zstep_metrics_text` 抓取：
- `zstep:num_requests_waiting`（Gauge：等待队列请求数）
- `zstep:num_requests_running`（Gauge：`Prefilling + Decoding` 活跃序列数）
- `zstep:kv_cache_usage_perc`（Gauge：`[0.0, 1.0]` 物理页占用比）
- `zstep:num_preemptions_total`（Counter：抢占次数，无抢占时为 `0`）
- `zstep:time_to_first_token_seconds` / `zstep:time_per_output_token_seconds`（Histogram）
- `zstep:request_outcomes_total{outcome="finished|cancelled|rejected_overloaded|failed"}`（Counter）
- `zstep:triton_recompilations_total`（Gauge/Counter：稳态必须为 `0`）
- **三态规则**：P0/P1 未在线接入 [`radix.rs`](../crates/zs-kv/src/radix.rs) 前，**禁止输出** `zstep:prefix_cache_hits_total` / `zstep:prefix_cache_queries_total`（P2 接入后再暴露）。

---

## 3. 分阶段 PR 切片路线图

1. **PR-Z1（P0-1 Tokenizer & 流式停止器）**：实现 `zs-tokenizer`，覆盖 HF `tokenizer.json` 加载、`deepseek_v41` 专用 `encode_messages()`、UTF-8 增量解码与跨 chunk `stop` 字符串裁剪；通过固定 Token-ID Trace 黄金对齐单测。
2. **PR-Z2（P0-2 常驻 `ServeLoop`）**：在 [`crates/zs-scheduler`](../crates/zs-scheduler/) 新增 `serve_loop.rs`，实现动态连续批处理（保留稳态 Decode `in_flight_depth = 1` 与混合步）、安全 Step 边界取消与 Worker 故障传染熔断；单测验证 50% 随机取消下 `SlotPool` 与 `BlockPool` 零泄漏。
3. **PR-Z3（P0-3 `zs-server serve` HTTP/SSE & 四态探针）**：在 [`crates/zs-server`](../crates/zs-server/) 新增 `serve.rs`，打通 `/v1/models`、`/v1/chat/completions`（流式/非流式）、四态探针、`429` 背压与 `x-nebula-request-id` 追踪。
4. **PR-Z4（P1 Metrics & `SIGTERM` Drain）**：补齐 `GET /metrics`、`SIGTERM` 优雅排空退出（`/dev/shm` 零残留）及与离线引擎的同 Trace 性能回归核验，随后与 Nebula `Slice N1/N2` 进行多副本真机合龙。
5. **PR-ZN3（P2 采样 + 在线 Radix + Nebula KV 事件）**：扩展 GPU 侧 `temperature/top_p/top_k/seed` 采样；将 [`KvCacheManager`](../crates/zs-kv/src/radix.rs) 接入 `ServeLoop`；按 `NEBULA_KV_HASH_SEED` 实现 16-token `sha256_cbor` 块哈希与 ZMQ `BlockStored`/`BlockRemoved` 事件发布，接通 `nebula-indexd` 跨副本前缀亲和。
