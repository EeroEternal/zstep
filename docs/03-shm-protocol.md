# 03 — 共享内存 IPC 与同步协议设计 (`zs-wire`)

本文档定义 `zstep` **单主控共享内存架构（1 个纯 Rust 主控进程 + 8 个 GPU Worker 进程）** 之间的二进制内存布局、同步原语、双缓冲重叠时序与故障安全机制。对应源码位于 `crates/zs-wire` 与 `worker/zstep_worker/shm_bridge.py`。

---

## 一、 设计目标与核心约束

1. **零序列化、零动态分配**：主控与 Worker 之间绝不使用 JSON / Protobuf / MsgPack 编解码，也不通过 Socket 拷贝张量元数据；所有字段均为 `#[repr(C)]` 定长对齐标量或扁平定界数组（Pod / Plain Old Data）。
2. **Cacheline 严格隔离（消除伪共享 False Sharing）**：主控只写区、Worker 只读区、以及各 Rank 状态回报区严格按 **64 字节（x86_64 Cacheline）** 对齐隔离。
3. **原生适配 `in_flight_depth = 1` 双缓冲**：提供 `slot[0]` 与 `slot[1]` 两个完全独立的 `StepPlanSlot`，当 8 个 Worker 正在消费 `slot[N & 1]` 启动 H2D 拷贝时，Rust 主控可同时无锁填充 `slot[(N + 1) & 1]`。
4. **跨语言未来兼容**：任何语言（Python `mmap` + `numpy.frombuffer`/`torch.frombuffer`、C++、CUDA Host Register、Zig）只要包含同一份内存偏移契约即可接入。

---

## 二、 进程间句柄与全局共享内存布局

### 1. OS 内核原语组合
* **数据通道**：单个 POSIX 共享内存对象 `/zstep_ipc_<instance_id>`，由 Rust 主控启动时 `shm_open(..., O_CREAT | O_EXCL | O_RDWR, 0600)` 创建并 `ftruncate` 至固定大小（默认 **64 MiB**，足够容纳大并发页表与重放索引），Worker 启动时以只读（Plan 区）+ 读写（Status/Result 区）映射，并调用 **`cudaHostRegister(..., cudaHostRegisterMapped)`** 将整块共享内存锁页注册为 GPU 可直接异步 DMA (`cudaMemcpyAsync`) 的 Pinned Memory！
* **唤醒通道（避免 CPU 空转轮询）**：
  * `plan_efd[0..8]`：8 个 Linux `eventfd`，主控写完当前 StepPlan 并执行 `Release` 内存屏障后，向 8 个 `plan_efd` 各写 `1u64` 同时唤醒 8 个 Worker。
  * `done_efd`：1 个汇总 `eventfd`（或每 Worker 1 个 `worker_efd[0..8]` 挂载在主控 `epoll` 上）：日常 Step 由 **Rank 0** 在完成采样结果写回后触发唤醒主控；初始化握手、权重检查与异常退出则由 8 个 Worker 各自上报。

### 2. `/zstep_ipc_<id>` 64 MiB 总体内存映射表

```text
Offset        Size        Alignment   Region Name & Ownership
───────────────────────────────────────────────────────────────────────────────────────
0x0000_0000   4 KiB       4096 B      [Region 0] ShmSuperblock (魔数、版本号、全局几何配置，主控初始化只写)
0x0000_1000   4 KiB       64 B/rank   [Region 1] WorkerSlotState[8] (8 个 Worker 各自独立占 512B：心跳、就绪态、错误码)
0x0000_2000   24 MiB      4096 B      [Region 2] StepPlanSlot[0] (偶数步 step_id % 2 == 0 元数据平面，主控写 / Worker 读)
0x0180_2000   24 MiB      4096 B      [Region 3] StepPlanSlot[1] (奇数步 step_id % 2 == 1 元数据平面，主控写 / Worker 读)
0x0300_2000   2 MiB       4096 B      [Region 4] StepResultSlot[2] (双缓冲采样/DSpark 验证回传区，Rank 0 写 / 主控读)
0x0320_2000   ~14 MiB     4096 B      [Region 5] ControlPayloadArena (低频控制指令载荷：热重载路径、Profile 输出路径等)
```

---

## 三、 核心 `#[repr(C)]` 数据结构规范

### 1. 全局超级块 `ShmSuperblock`
启动握手阶段校验，防止 Rust 主控与 Python Worker 版本不一致导致内存错位：
* `magic: u64` = `0x5A53_5445_505F_5631` (`"ZSTEP_V1"`)
* `protocol_version: u32` = `3`
* `world_size: u32` = `8`（对应 `8 × RTX 6000D`）
* `max_seqs: u32`（例如 `64`）
* `max_step_tokens: u32`（例如 `16384`，单步最大 Chunked Prefill + Decode/Verify 行数）
* `max_pages_per_seq: u32`（例如 `2048`，64 tokens/page 下支持 128K 上下文）
* `spec_verify_width: u32`（DSpark 下为 `6`：1 个上步确定 Token + 5 个草稿 Token）
* `master_pid: u32`（Worker 监控父进程存活）

### 2. 单步执行计划 `StepPlanSlot`（专为 `DeepSeek-V4.1-Flash` 定制）
每个 `StepPlanSlot` 头部是一个 128 字节对齐的 `StepHeader`，紧随其后是定界紧凑数组（Worker 只按实际有效长度 `cudaMemcpyAsync` H2D，绝不拷贝空白尾部）：

```rust
#[repr(C, align(64))]
pub struct StepHeader {
    /// 单调递增步号（0 表示空置）；Worker 校验 published_step_id == expected_step_id
    pub step_id: u64,
    /// 命令类型：1 = NormalStep, 2 = WarmupCaptureGraphs, 3 = ReloadForward, 99 = Shutdown
    pub opcode: u32,
    /// 当前调度批次请求数 B (0 <= B <= max_seqs)
    pub num_seqs: u32,

    // ── CED 四阶段行数与 CUDA Graph 桶指示（主控统一决策，8 卡严格一致） ──
    /// Stage 1 (Layers 0..19) 实际有效输入 Token 总行数 T_enc
    pub num_encoder_tokens: u32,
    /// Stage 1 匹配的 CUDA Graph Padding 桶大小（0 表示走 Eager）
    pub encoder_graph_bucket: u32,
    /// Stage 3 (Layers 21..39) 经第 20 层窄化后的实际有效行数 T_dec
    pub num_decoder_rows: u32,
    /// Stage 3 匹配的 CUDA Graph Padding 桶大小（纯 Decode/Verify 时启用整图模式）
    pub decoder_graph_bucket: u32,

    /// 需要采样/验证输出的槽位数
    pub num_sample_rows: u32,
    /// 是否启用 DSpark 6-token 投机验证路径 (0 = 普通单步, 1 = DSpark Verify+Draft)
    pub is_dspark_verify: u32,
    pub flags: u32,
    pub _reserved: u32,
}
```

**紧随 `StepHeader` 的扁平数组区（均按 256 字节对齐，适配 GPU Vectorized Load）：**
1. **输入 Token 与位置区（长度 `num_encoder_tokens`）**：
   * `token_ids: [i32; max_step_tokens]` — 输入词表 ID（若前一步尚在 GPU 飞行中，Decode/Verify 槽填 `-1` 标记由 GPU 直接从上一步设备端输出缓冲区自举回填，**实现真正的零停顿 GPU 自举 Overlap**！）。
   * `positions: [i32; max_step_tokens]` — 绝对 RoPE 位置索引。
   * `req_slot_ids: [i32; max_step_tokens]` — 每行归属的请求槽位号 `0..B`。
   * `replay_mask: [u8; max_step_tokens]` — **SWA 有界重放掩码**：`1` 表示该行是 Prefix Hit 重放的最后 128 Token，只写私有 `SwaWindow` / `CompressorTail`，跳过写共享 `GlobalKV` / `IndexKey`。
2. **Engram 3-gram 预计算输入区（长度 `num_encoder_tokens`）**：
   * `engram_prev_window: [[i32; 3]; max_step_tokens]` — 主控维护的归一化前 3 个历史 Token ID（边界或非文本填 `-1`），或直接传递预哈希列索引与 `engram_mask: [u8; max_step_tokens]`。
3. **CED 第 20 层窄化映射区**：
   * `narrow_keep_indices: [i32; max_decoder_rows]` — 从 `0..T_enc` 中挑出进入第 20~39 层的行索引。
   * `sample_row_indices: [i32; max_sample_rows]` — 从 `0..T_dec` 中挑出进入 LM Head 采样的行索引。
4. **FlatKV 四组页表与序列长度区（长度 `B = num_seqs`）**：
   * `seq_lens: [i32; max_seqs]` — 各请求当前可见总上下文长度。
   * `query_start_loc: [i32; max_seqs + 1]` — Ragged 前缀和偏移（`cu_seqlens_q`）。
   * `global_kv_page_table: [[i32; max_pages_per_seq]; max_seqs]` — Group 0 & Group 1 共用的物理页号表（它们 1:1 同步分配，节省一半页表 H2D 带宽！）。
   * `swa_page_table: [[i32; 4]; max_seqs]` — Group 2 环形窗口物理页号（每请求仅需 4 个 `i32`！）。
   * `compressor_slot_ids: [i32; max_seqs]` — Group 3 压缩尾槽物理索引。
5. **采样参数区（长度 `B`）**：
   * `temperatures: [f32; max_seqs]`、`top_p: [f32; max_seqs]`、`top_k: [i32; max_seqs]`、`rng_seeds: [u64; max_seqs]`。

### 3. 结果回传区 `StepResultSlot`（Rank 0 写，主控读）
```rust
#[repr(C, align(64))]
pub struct StepResultHeader {
    /// 已完成的步号
    pub completed_step_id: u64,
    /// GPU 端纯执行耗时（微秒，通过 CUDA Event 测得，供主控 Prometheus 指标采集）
    pub gpu_exec_us: u32,
    /// 状态码：0 = OK, 非 0 = Worker 运行时异常
    pub status_code: u32,
    /// 实际写回的请求结果条数
    pub num_seq_results: u32,
    pub _pad: [u32; 11],
}

#[repr(C)]
pub struct SeqStepOutput {
    pub req_slot: u32,
    /// 本步最终接受的真实 Token 数量（普通 Decode 为 1；DSpark Verify 为 1..=6）
    pub num_accepted: u32,
    /// 接受的真实 Token ID 序列（按顺序吐给流式客户端，并推进主控 KV 长度与 Engram 历史）
    pub accepted_tokens: [i32; 8],
    /// DSpark 为下一步预生成的 5 个草稿候选 Token（用于主控校验或日志诊断）
    pub next_draft_tokens: [i32; 8],
    /// 可选选中 Token 的 logprob 诊断值
    pub accepted_logprobs: [f32; 8],
}
```

---

## 四、 内存屏障与崩溃防护契约

1. **发布-订阅内存序（Acquire / Release Ordering）**：
   * Rust 主控写完所有扁平数组后，最后一步使用 `AtomicU64::store(step_id, Ordering::Release)` 写入 `StepHeader.step_id`，随后 `write(plan_efd)`。
   * Python/C Worker 从 `plan_efd` 醒来后，通过 C 扩展或原子读取执行 `Acquire` 栅栏读取 `step_id`，确保所有数组字节对当前 CPU 核及随后的 DMA 引擎完全可见。
2. **零孤儿进程（`PR_SET_PDEATHSIG`）、即时 `shm_unlink` 与广播中止（`SUPERBLOCK_ABORT_OFFSET`）**：
   * 8 个 Worker 子进程启动时设置 `prctl(PR_SET_PDEATHSIG, SIGTERM)` 并安装 `SIGTERM`/`SIGINT` 信号处理器；Rust 主控在全部 8 个 Worker 完成 `mmap` 挂载握手后**立即执行 `SharedMemoryArena::unlink_now()`**（数据面 `shm_v5` AllReduce/AllGather 与 `HostMappedShmArgmax` 同样在 `dist.barrier` 后立即 `os.unlink`），因此即使遭遇 `SIGKILL` 极端断电/强杀，`/dev/shm` 也保持 **0 文件残留**。
   * `ShmSuperblock` 显式携带 `result_slot_stride`（offset `96`）、`layout_hash`（offset `104`，64-bit FNV-1a 摘要，同时覆盖 `ZSTEP_PROTOCOL_VERSION`、全部 POD 关键字段偏移、`STEP_FLAG_DECODE_ONLY` / `STEP_FLAG_DEVICE_TOKENS` / `STEP_FLAG_NEXT_PREPUBLISHED` 与 `StepPlanLayout` 切片偏移）与 `max_decode_graph_seqs = 128`（offset `112`，紧邻 `_pad_u32` offset `116`），Python `ShmBridge` 在挂载时逐项强校验；`SUPERBLOCK_ABORT_OFFSET = 128` 提供无锁全局中止位，Worker 通过 `poll(plan_efd, 250ms)` 与后台心跳线程感知异常并立即中断 GPU 集体通信自旋。
   * `WorkerSlotState` 独立字段 `triton_recompilations`（offset `44`）与 `watched_kernels`（offset `48`）按 Rank 汇报 Triton JIT 编译增量，不再复用 `StepResultHeader` 状态位。

---

## 五、 当前已启用字段 vs 后续演进预留字段对照表

为避免前瞻架构设计与当前 Phase 2 / Phase 3.6 实现能力混淆，特明确各协议区段状态：

| 协议字段 / 机制 | 状态 | 当前实现说明 |
| :--- | :---: | :--- |
| `ShmSuperblock` (`layout_hash`, `result_slot_stride`, `max_decode_graph_seqs`, `SUPERBLOCK_ABORT_OFFSET`) | **已启用** | Rust/Python 双端强校验 + Golden JSON 跨语言单测锁定 |
| `StepPlanSlot`：`token_ids`, `positions`, `req_slot_ids`, `sample_row_indices`, `seq_lens`, `query_start_loc`, `global_kv_page_table`（脏行增量同步）, `engram_prev_window` | **已启用** | Phase 2 (`Qwen3.5-35B-A3B`) 与 Phase 3.6 (`DeepSeek-V4.1-Flash` + `b12x`) 生产路径使用 |
| `SeqStepOutput`：Top-1 / Top-2 token 与原始 BF16 logit 回传 | **已启用** | `accepted_tokens[0]` + `next_draft_tokens[0]` 承载确定性全局 Top-2 贪心审计 |
| `StepOpcode::WarmupCaptureGraphs` / `ReloadForward`、`ControlPayloadArena` | **预留** | Warmup 与 CUDA Graph 捕获目前在 Worker 初始化阶段直接完成 |
| `is_dspark_verify`、`narrow_keep_indices`、`replay_mask`、`swa_page_table`、`compressor_slot_ids`、`temperatures`（随机采样） | **预留** | 面向后续 DSpark 投机验证、CED 层间窄化、SWA 前缀重放与非贪心采样预留定长偏移（SWA/Compressor 页映射当前由 `DeepSeekV41StepRunner` 从 `global_kv_page_table` 与 `req_slot_ids` 无损派生） |
