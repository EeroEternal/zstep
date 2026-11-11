# 01 — 技术选型、硬件边界与架构调研纪要

本文档汇总并沉淀了 `zstep` 立项阶段对 **ZML**、**TokenSpeed**、**`b12x` (FlashInfer)** 三个代码库的源码级核查结论，修正早期调研误区，并针对目标服务器（**`8 × NVIDIA RTX 6000D`**）与目标模型（**`DeepSeek-V4.1-Flash`**）确立最终工程路线。

---

## 一、 三大核心事实对齐

### 1. 目标硬件：`8 × NVIDIA RTX 6000D` (`sm_120`) 的真实物理特征
* **核心架构与指令集分裂（`sm_120` vs `sm_100`）**：
  * RTX 6000D 基于 `GB202GL` 核心（156 SMs），计算能力为 **`sm_120`**（工作站/消费级 Blackwell 架构），采用 PCIe 5.0 x16 互联，**无 NVLink**。
  * 它与数据中心 Blackwell（B200/GB200，`sm_100`/`sm_103`）存在本质的底层指令分裂：**`sm_120` 没有 Tensor Memory (`tcgen05` 指令族) 和 2-CTA Cluster Multicast**；它的第 5 代 Tensor Core 走的是寄存器累加的 `mma.sync.aligned.block_scale`（原生硬件支持 **NVFP4 (`E2M1+E4M3`)**、**MXFP4 (`E2M1+E8M0`)**、**MXFP8 (`E4M3+E8M0`)**）。
  * 因此，现有引擎中针对 `sm_100` 编写的 `tcgen05` 内核在物理指令集上无法在 RTX 6000D 上运行。
* **显存容量优势 vs 显存带宽瓶颈**：
  * **容量极大**：单卡配备 **84 GB GDDR7（带 ECC）**，整机 8 卡总计 **672 GB 显存**。跑 `DeepSeek-V4.1-Flash`（`TP=8, EP=8`）时，每卡分摊约 40~45 GB 权重后，仍剩余约 **38~40 GB/卡** 充裕显存用于 FlatKV 缓存、Engram 分片与分段 CUDA Graph。
  * **带宽受限**：单卡 448-bit GDDR7 带宽约 **~1.4 TB/s**（远低于 B200 的 8 TB/s HBM3e），且跨卡无 NVLink。这意味着 Decode 阶段处于极端的 **访存带宽受限（Memory-Bandwidth Bound）+ PCIe 小包通信延迟受限** 状态。**原生 MXFP4/NVFP4/MXFP8 计算 + DSpark 多 Token 投机验证 + 定制 PCIe P2P 低延迟 AllReduce** 是榨干该机器的三大支柱。

### 2. `b12x` 的生态位置与价值
* 原 `local-inference-lab/b12x` 独立仓库已归档（Archived），其全部核心算子、Autotuner、Workspace 管理与 JIT 编译管线已正式并入 **FlashInfer 主仓的 `flashinfer/experimental/b12x`**（通过 `flashinfer-python[b12x]` 分发）。
* 它是目前开源生态中专门针对 **`sm_120` / `sm_121`（GB202 / RTX 6000D / RTX PRO 6000 / RTX 5090）** 打磨最深入的算子库，直接提供了：
  1. `sm_120` 原生 **NVFP4 / MXFP4 / MXFP8 Block-Scaled GEMM**；
  2. 单次启动融合的 **`b12x_fused_moe`**（融合路由分发、片上激活量化、Gate/Up/SwiGLU/Down 与权重归约）；
  3. Attention 投影融合算子。
* **工程约束**：`b12x` 基于 Python 侧的 **CuTe DSL (`nvidia-cutlass-dsl`) + Triton JIT** 构建。保持数据面 Worker 对 Python/CUDA Graph 生态的零摩擦调用，才能在第一天直接复用 `b12x` 的全部 `sm_120` 算力。

### 3. 对 ZML 与 TokenSpeed 源码边界的核查
* **为什么不能直接把 ZML 重写成 Rust，也不能走 `Rust -> melior/StableHLO -> PJRT` 路线**：
  * ZML 的核心资产在于 MLIR/StableHLO 编译管线（含自定义 `cute_nvgpu` `sm120` 方言绑定）+ Bazel 工具链 + PJRT，而非语言外壳。
  * Rust 社区的 `melior` 不含 StableHLO 绑定，PJRT Rust 生态亦不成熟；更重要的是，**XLA/PJRT 等静态整图编译器会将单步执行黑盒化**，与“精细控制每一个推理 Step（动态页表、CED 第 20 层动态窄化、SWA 有界重放、DSpark 投机分支）”的核心诉求背道而驰。
  * **真正应该向 ZML 学习的**：是它**去魔法化、将模型层与推理状态表达为显式维度 Struct** 的工程组织方式。
* **为什么不能直接在原版 TokenSpeed 上硬跑 `8 × RTX 6000D`，也不能直接黑盒 `cargo add` 其调度 Crate**：
  * 原版 TokenSpeed 的 GPU 算子层面向数据中心 `sm90` / `sm100..sm107` 构建：MoE、采样与通信内核普遍硬编码排除了 `sm_120`（如 `max_arch_version=ArchVersion(10, 3)`、`__CUDA_ARCH__ < 1200`）。
  * 其 Rust 控制面 `ts-control` 的 `DataPlane` 抽象非常优秀，但底层 `ts-scheduler` 强耦合了 `pyo3` 与 `nanobind`（调度器对象由 Python 构造再传回 Rust），且采用多进程冗余调度器对账架构。因此最佳实践是**吸收其控制面/数据面隔离与统一执行路径（Unified Path）思想，用纯 Rust 为单机 8 卡定制轻量主控**。

---

## 二、 控制面与 GPU Worker 连接模式对比（模式 A vs 模式 B）

针对单机 `8 × RTX 6000D` 固定运行 `TP=8 + EP=8` 的物理形态，我们对两种跨语言连接架构进行了全面对比：

| 对比维度 | 模式 A：同进程内嵌解释器 (`ts-host` 模式) | 模式 B：单主控 + 多 Worker 共享内存 IPC（`zstep` 选定方案） |
| :--- | :--- | :--- |
| **进程拓扑** | 8 个对等进程；每个进程内含 Rust 控制线程 + `pyo3` 内嵌 Python Forward 线程。 | **1 个纯 Rust 主控进程** + **8 个独立 Python/CUDA Worker 进程**（Rank 0..7）。 |
| **调度器复杂度** | 8 个进程各自运行一份冗余调度器，每步需通过 TCP Mesh 广播与 FNV-1a 哈希对账对齐状态。 | 全机**唯一一个调度器与页表分配器**，天然一致，省去 50% 分布式控制同步代码。 |
| **GIL 与构建耦合** | 需精细管理 Python GIL 获取/释放时序与线程防呆锁；Rust 编译期强绑定特定 `libpython.so`。 | Rust 主控 **零 `pyo3`、零 `libpython`、零 CUDA 依赖**，纯静态高速编译；Python venv 可随时独立升级。 |
| **通信延迟** | 同进程传指针（~0.1 μs）。 | POSIX `shm_open` (`mmap`) + `eventfd` 跨进程唤醒（**~2–5 μs**）。在 5~15 ms 的 GPU Step 耗时及 `in_flight_depth=1` 双缓冲重叠下，**端到端吞吐损耗为 0%**。 |
| **容错与调试体验** | GPU 算子触发 `illegal memory access` 或 Segfault 时，整个控制面与客户端连接同归于尽。 | 故障域物理隔离；支持保留主控或快速重启/热重载 Worker 调试 `sm_120` 内核。 |
| **未来演进能力** | 与 CPython C API 深度绑定。 | 协议边界是纯 `#[repr(C)]` 共享内存块；未来可将 Worker 平滑替换为纯 C++/CUDA 或 Zig 进程而无需改动主控一行代码。 |

---

## 三、 目标模型 `DeepSeek-V4.1-Flash` 在 `8 × RTX 6000D` 上的核心特征

`DeepSeek-V4.1-Flash` 并非传统均质 Transformer，必须在架构设计第一天将其四大机制作为一等公民原生支持：

1. **体量与切分（`TP=8, EP=8`）**：
   * 40 层网络，`hidden_size = 5120`，每层 **384 个 MXFP4 路由专家**（`intermediate_size = 2304`, `topk = 6`, `sqrt_softplus` 门控, `swiglu_limit = 10.0`）。
   * 总权重规模超 300 GB，整机 8 张 6000D（672 GB）以 `TP=8 + EP=8` 运行单个完整实例最为理想。
2. **CED 第 20 层行数截断（Encoder-Decoder Row Narrowing）**：
   * 第 0~19 层（Encoder）处理所有调度的 Prompt Token；第 20 层（Candidate Source）写完全局 KV 后将行数窄化，第 20~39 层（Decoder）仅保留每个请求最后 128 个位置及 Decode 行。
   * Prefill CUDA Graph 必须围绕第 20 层拆分为 `EncoderGraph` 和 `DecoderGraph` 两段捕获。
3. **FlatKV 四组异构缓存与 SWA 有界重放（Bounded Replay）**：
   * 缓存分为全局 KV（512D NVFP4 `E2M1/E4M3`）、索引键（128D MXFP4 `E2M1/E8M0`）、SWA 窗口（512D MXFP8 `E4M3/E8M0`）与压缩尾部。
   * 注意到 RTX 6000D (`sm_120`) **在硬件指令层面原生支持上述全部三种低比特格式（NVFP4 / MXFP4 / MXFP8）**！
   * SWA 与压缩尾部不进入全局 Prefix Cache，命中前缀时调度器自动重放最后 128 个 Token（Bounded Replay）。
4. **mHC 多流超连接残差 + Engram 3-gram 哈希表 + DSpark 投机解码**：
   * Engram 依赖 Tokenizer 归一化后的前 3 个 Token 计算多列哈希；由于 6000D 显存充裕（每卡富余近 40 GB），`zstep` 既支持主机锁页内存 PCIe Gen5 UVA 读取，也支持直接将 Engram 分片常驻显存消除跨总线读取。
   * DSpark 同检查点投机解码每步并行验证 **6 个候选 Token**，将每层 2 次（全网 80 次/Step）PCIe AllReduce 的通信开销摊薄 4~5 倍，是攻克无 NVLink 瓶颈的核心利器。
