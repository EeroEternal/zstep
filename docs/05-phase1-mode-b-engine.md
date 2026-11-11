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
| **主控发布 + 唤醒 8 个 Worker (`mean_publish_signal_us`)** | **`23.995 µs`** | 原子写入 `StepHeader.step_id` + 8 次 `eventfd` 通知总耗时 |
| **控制面 IPC 中位数 (`control_plane_p50_us`)** | **`325 µs`** (`p95 = 346 µs`, `p99 = 385 µs`) | 含跨进程唤醒、Python H2D/D2H 采样回传与 `done_efd` 收集；与下一拍 Rust 准备成本是不同计时口径 |
| **下一拍 Rust 准备成本 (`mean_prepare_next_us_per_step`)** | **`4.690 µs/step`** | 本轮 `prepare_next_us = 938 µs`，该字段以全部 200 步为分母；其中 199 步发生重叠。仅计算 Rust 元数据准备，不含 IPC 往返 |
| **准备成本相对 GPU 步耗时** | **`0.166%`** | `4.690 µs / 2,832.095 µs`；`hidden_prepare_us = 938 µs` 表示准备落在 GPU 执行窗口内，不表示获得 100% 端到端加速 |
| **重叠等效加速 (`overlap_speedup`)** | **`1.000269×`（约 `1.0003×`）** | 按 `(total_wall_us + hidden_prepare_us) / total_wall_us` 计算，避免将窗口内调度比例误读为性能翻倍 |
| **运行时 Triton 重编译** | **0** | 动态轮换 `M ∈ {6, 48, 192}`；Phase 2 运行时审计另监视 `WATCHED_KERNELS=17` 并要求 `TRITON_RECOMPILATIONS=0` |
| **退出后 `/dev/shm` 残留检查** | **0 残留（全自动回收）** | `test_mode_b_e2e.py` 的 1-GPU 24 步与 8-GPU 64 步回归均已重跑通过，断言主控 IPC 与 AllReduce 共享内存段彻底 `unlink` |

上一次同口径 200 步记录为 **`5.08 µs/step`**（累计 **`1,016 µs` / 199 步**）、GPU 平均步 **`3,066.94 µs`**、准备/GPU **`0.166%`**、`overlap_speedup = 1.000281×`。本轮 JSON 已更新为上述新测量；两次结果共同支持的结论仅是准备成本约 `0.17%`、等效加速约 `1.0003×`，不是夸张的控制面加速宣传。
