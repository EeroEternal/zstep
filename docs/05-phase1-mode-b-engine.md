# 05 — Phase 1：Mode B（纯 Rust 主控 + 8 × `sm_120` Worker 共享内存引擎）实测报告

Phase 1 已将 `zstep` 的 **Mode B 控制面/数据面物理分离架构** 完整落地并在 8 卡 `sm_120` 测试机上完成端到端实测：1 个零 Python/GIL 依赖的纯 Rust 主进程（`zstep --mode-b`）通过 POSIX 共享内存（`/dev/shm`）与继承式 `eventfd` 同时驱动 8 个独立 Python/CUDA `sm_120` Worker 进程执行 `in_flight_depth = 1` 双缓冲重叠推理步。

---

## 一、 代码模块与 RAII 职责边界

| 模块路径 | 核心职责与安全不变量 |
| :--- | :--- |
| [`crates/zs-wire`](../crates/zs-wire/src/lib.rs) | `#[repr(C)]` 缓存行对齐协议头（`ShmSuperblock` 128B、`WorkerSlotState` 64B、`StepHeader` 64B、`StepPlanLayout` 双槽位、`StepResultHeader` 64B、`SeqStepOutput` 104B）。 |
| [`crates/zs-shm`](../crates/zs-shm/src/lib.rs) | 纯 Rust Linux `shm_open` / `ftruncate` / `mmap` / `eventfd(EFD_CLOEXEC)` 传输层。`SharedMemoryArena` 与 `EventFd` 均实现严格 RAII `Drop`（进程正常退出或异常栈展开均自动 `munmap` + `shm_unlink` + `close`，绝不在 `Drop` 中 panic）。 |
| [`crates/zs-scheduler/src/mode_b.rs`](../crates/zs-scheduler/src/mode_b.rs) | `ModeBCoordinator`：通过 `pre_exec` `dup2` 向 8 个子进程传递精确 `plan_fd` / `done_fd`，驱动 `slot = step_id & 1` 双缓冲流水线（GPU 执行 `Step N` 的同时，Rust 主线程在另一槽位提前写入 `Step N+1`），并内置 200ms 切片子进程存活快检（`try_wait`）。 |
| [`worker/zstep_worker/shm_bridge.py`](../worker/zstep_worker/shm_bridge.py) | Python 零拷贝 `mmap` + `libatomic` Acquire/Release 原子头部视图，与 Rust `zs-wire` 字节级 100% 对齐。 |
| [`worker/zstep_worker/runtime_loop.py`](../worker/zstep_worker/runtime_loop.py) | `prctl(PR_SET_PDEATHSIG, SIGTERM)` 父死自毁保护 + 真实 `sm_120` 前向微步（融合 `rmsnorm_quantize_mxfp8_group32` + `b12x.mxfp8_linear` + `dsv41_sqrt_softplus_topk` + 8 卡 `HostMappedShmAllReduce`），并内置 `_do_compile` 零重编译守卫。 |

---

## 二、 8 × `sm_120` GPU 端到端 200 步重叠实测指标

测试命令（归档于 [`docs/benchmarks/phase1_mode_b_8gpu.json`](./benchmarks/phase1_mode_b_8gpu.json) 与 [`worker/tests/test_mode_b_e2e.py`](../worker/tests/test_mode_b_e2e.py)）：

```bash
zstep --mode-b --world-size 8 --steps 200 --token-counts 6,48,192
```

| 指标项 | 8 × `sm_120` 实测值 | 说明 |
| :--- | :---: | :--- |
| **`world_size` / 总步数** | `8 GPUs` / `200 steps` | 动态轮换 `M ∈ {6, 48, 192}`，**0 次运行时 Triton 重编译** |
| **主控发布 + 唤醒 8 个 Worker (`mean_publish_signal_us`)** | **`19.30 µs`** | 原子写入 `StepHeader.step_id` + 8 次 `eventfd` 通知总耗时 |
| **控制面往返开销中位数 (`control_plane_p50_us`)** | **`433 µs`** (`p95 = 525 µs`, `p99 = 592 µs`) | 含跨进程唤醒、Python H2D/D2H 采样回传与 `done_efd` 收集；后续绑入静态 CUDA Graph 锁页缓冲可进一步压至 `<50 µs` |
| **下一跳调度掩盖率 (`overlap_hiding_efficiency`)** | **`100.0%` (`1.000000`)** | 200 步累计 `prepare_next_us = 671 µs` 全部被 GPU 执行窗口完全掩盖（`hidden_prepare_us = 671 µs`） |
| **退出后 `/dev/shm` 残留检查** | **0 残留（全自动回收）** | `test_mode_b_e2e.py` 断言主控 IPC 与 8 卡 AllReduce 共享内存段均已彻底 `unlink` |
