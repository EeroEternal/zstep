# `zstep` Master Benchmark Ledger (`sm_120`, 8×32 GB Non-P2P 4-NUMA PCIe)

本目录集中归档 `zstep` 从 **Phase 0（底层 `sm_120` 算子与跨 NUMA PCIe AllReduce）**、**Phase 1（Mode B 纯 Rust 控制面 ↔ 8 卡共享内存双缓冲重叠调度）** 到 **Phase 2（`Qwen/Qwen3.5-35B-A3B` 真权重端到端连续批处理与 `vLLM 0.33.0` 同机正面对标）** 的全部实测 JSON 原始数据与复现命令。

---

## 1. 硬件与软件基线指纹（[`step_00_env_sm120.json`](step_00_env_sm120.json)）

```text
GPU:       8 × NVIDIA GeForce RTX 5090 (sm_120, 170 SMs, 31.36 GiB / 32 GB GDDR7 per GPU)
CPU/NUMA:  Dual-Socket Xeon Gold 6530, 4 Sub-NUMA Clustering (SNC) domains
Topology:
  GPU 0, 1, 2 -> PCIe Switch A -> NUMA 0
  GPU 3       -> Root Complex  -> NUMA 1
  GPU 4, 5, 6 -> PCIe Switch B -> NUMA 2
  GPU 7       -> Root Complex  -> NUMA 3
Interconnect: PCIe Gen5 x16, NO NVLink, can_device_access_peer(i, j) == 0 across all 8×8 pairs
```

> **口径边界**：由于本机 `can_device_access_peer` `8×8` 全为 `0`，NCCL 在本机亦回退至主机共享内存传输（`degraded NCCL SHM`）。本仓库所有跨卡通信与 `TP=8` 结论均严格限定在 **无 P2P 的 4-NUMA PCIe 拓扑**，不外推至 NVLink 或支持直接 BAR1 P2P 的平台。`84 GB` / `96 GB` 大显存卡容量几何在代码中标记为 `unverified`。

---

## 2. 归档文件总览（12 份实测 JSON）

| 阶段 | 归档 JSON 文件 | 测试对象与覆盖维度 |
| :--- | :--- | :--- |
| **Phase 0.1** | [`step_00_env_sm120.json`](step_00_env_sm120.json) | 8 卡 `sm_120` 硬件拓扑、P2P 矩阵（全 `0`）、`b12x` / CuTe DSL 环境探测 |
| **Phase 0.2** | [`step_01_dense_fp8_gemm.json`](step_01_dense_fp8_gemm.json) | Triton 融合 `RMSNorm + MXFP8 Group-32` 量化（含 `>2^31` `int64` 偏移与零重编译验证）+ `b12x` `MXFP8` 稠密 GEMM (`5120×2048`) |
| **Phase 0.3** | [`step_03_moe_mxfp4_sm120.json`](step_03_moe_mxfp4_sm120.json) | `dsv41_sqrt_softplus_topk` 384 专家路由（含 `8.57 GiB` `>2^31` 元素偏移验证）+ `b12x` `MXFP4` 分组 MoE（`EP=8` vs `TP=8`） |
| **Phase 0.4** | [`step_04_dsv41_attention_sm120.json`](step_04_dsv41_attention_sm120.json) | `deepseek_v41` 分页稀疏 MLA（16 头, `D=512`, `SWA=128 + TopK=512`, `KV=2K..32K` Decode / Verify / Extend） |
| **Phase 0.5a** | [`step_05a_mhc_engram_sm120.json`](step_05a_mhc_engram_sm120.json) | 4 流 20 步 Sinkhorn `mHC` 算子 + `3.84 亿` 行 `Engram` 表（`12.29 GB` 分片，`>2^31` 字节偏移，VRAM vs 主机页锁 UVA） |
| **Phase 0.5b** | [`step_05b_pcie_allreduce_8gpu.json`](step_05b_pcie_allreduce_8gpu.json) | NUMA 绑核 + 4 KiB 页对齐 First-Touch 后 8 卡并发 `HostMappedShmAllReduce` vs 降级 `NCCL SHM`（4,000 次 CUDA Graph 重放，`p50/p95/p99`） |
| **Phase 1** | [`phase1_mode_b_1gpu.json`](phase1_mode_b_1gpu.json) | Mode B 纯 Rust 主控 ↔ 单 GPU Worker 200 步 `in_flight_depth = 1` 双缓冲重叠与 IPC 分位数 |
| **Phase 1** | [`phase1_mode_b_8gpu.json`](phase1_mode_b_8gpu.json) | Mode B 纯 Rust 主控 ↔ 8 个 NUMA 绑核 GPU Worker 200 步双缓冲重叠与 IPC 分位数 |
| **Phase 2** | [`phase2_vllm_qwen35_35b_a3b_tp8.json`](phase2_vllm_qwen35_35b_a3b_tp8.json) | 同机 `vLLM 0.33.0`（`Qwen/Qwen3.5-35B-A3B`, `TP=8`, `bfloat16`）两组三档并发（6 档）基线 |
| **Phase 2** | [`phase2_zstep_qwen35_35b_a3b_tp8.json`](phase2_zstep_qwen35_35b_a3b_tp8.json) | 同机 `zstep Mode B`（`--max-step-tokens 8192`, `allreduce_blocks=8`）6 档并发完整实测 + 16-token 贪心对齐记录 |
| **Phase 2** | [`phase2_zstep_qwen35_35b_a3b_tp8_step4k.json`](phase2_zstep_qwen35_35b_a3b_tp8_step4k.json) | 同机 `zstep Mode B`（`--max-step-tokens 4096`, `allreduce_blocks=8`）6 档并发完整实测（`in512_c128` 达 **`4,638.96 tok/s`**） |
| **Phase 2** | [`phase2_qwen35_optimization_ablation.json`](phase2_qwen35_optimization_ablation.json) | `Qwen3.5-27B` 参考基线 + `zstep Mode B` 四阶段工程优化消融全记录（Run 1 → Run 4） |

---

## 3. 核心实测数据汇总

### 3.1 Phase 2：端到端 `Qwen/Qwen3.5-35B-A3B` (`TP=8`, `bfloat16`) vs `vLLM 0.33.0`

#### 工况 A：`input_len = 512`, `output_len = 256`

| 并发度 (`prompts`) | `vLLM 0.33.0` (`tok/s`) | `zstep Mode B` `step=4K` (`tok/s`) | `zstep Mode B` `step=8K` (`tok/s`) | 最优吞吐增益 | `vLLM` Median TTFT / Mean TPOT | `zstep` (`step=4K` / `8K`) Median TTFT | `zstep` (`step=8K`) Mean / P99 TPOT |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`c = 1`** (`8`) | `184.72` | `224.18` | **`225.04`** | **`+21.8%`** | `74.51 ms` / `5.09 ms` | `67.32 ms` / **`66.59 ms`** | **`4.09 ms` / `4.11 ms`** |
| **`c = 16`** (`48`) | `1,671.63` | `1,744.61` | **`1,768.73`** | **`+5.8%`** | `392.38 ms` / `8.02 ms` | **`290.93 ms`** / `465.85 ms` | **`7.25 ms` / `7.31 ms`** |
| **`c = 128`** (`256`) | `3,271.56` | **`4,638.96`** | `4,540.92` | **`+41.8%`** | `1,240.50 ms` / `33.25 ms` | **`822.28 ms`** / `1,307.87 ms` | **`22.38 ms` / `29.57 ms`** |

#### 工况 B：`input_len = 1024`, `output_len = 256`

| 并发度 (`prompts`) | `vLLM 0.33.0` (`tok/s`) | `zstep Mode B` `step=4K` (`tok/s`) | `zstep Mode B` `step=8K` (`tok/s`) | 最优吞吐增益 | `vLLM` Median TTFT / Mean TPOT | `zstep` (`step=8K`) Median TTFT | `zstep` (`step=8K`) Mean / P99 TPOT |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`c = 1`** (`20`) | `181.96` | `257.30` | **`261.72`** | **`+43.8%`** | `143.35 ms` / `4.86 ms` | **`82.73 ms`** | **`3.51 ms` / `3.61 ms`** |
| **`c = 32`** (`200`) | `1,834.83` | `2,001.20` | **`2,165.22`** | **`+18.0%`** | `1,091.90 ms` / `11.92 ms` | **`870.08 ms`** | **`9.81 ms` / `12.88 ms`** |
| **`c = 128`** (`400`) | `2,939.12` | `2,750.47` | **`3,024.65`** | **`+2.9%`** | `1,355.88 ms` / `33.77 ms` | **`1,295.93 ms`** | **`33.19 ms` / `39.74 ms`** |

> 同机稠密模型参考基线（`vLLM 0.33.0`, `Qwen/Qwen3.5-27B`, `TP=8`, `bfloat16`）：`c = 1` 输出吞吐 `85.5 tok/s`，`c = 128` 输出吞吐 `1,060.0 tok/s`。

---

### 3.2 Phase 0.5b：8 卡无 P2P 4-NUMA PCIe AllReduce (`H = 5120`, `bfloat16`, CUDA Graph 连续调用)

来源：[`step_05b_pcie_allreduce_8gpu.json`](step_05b_pcie_allreduce_8gpu.json)（8 卡同时运行，50 轮 × Graph 内 80 次调用）

| Token 数 `M` | 载荷字节 | `zstep` `HostMappedShmAllReduce` (`p50` / `p95` / `p99` `µs`) | 降级 `NCCL SHM` (`p50` / `p95` / `p99` `µs`) | `zstep` Graph 精度 (`max_abs_err`) | 对比结论（无 P2P 4-NUMA PCIe） |
| :---: | :---: | :---: | :---: | :---: | :--- |
| `6` | `60 KiB` | `65.16` / `69.13` / `69.84` | `52.42` / `53.59` / `56.70` | **`0.0`** | 差 `12.7 µs`（注：`H=2048` `B<=4` 即 `<=16 KiB` 已启用单屏障 One-Shot 快路径） |
| `48` | `480 KiB` | `150.19` / `152.56` / **`153.04`** | `106.80` / `145.69` / `155.33` | **`0.0`** | `zstep` `p99` 反超降级 NCCL（`153.04` vs `155.33 µs`），尾延迟更稳 |
| `192` | `1.88 MiB` | `187.49` / `190.45` / `191.47` | `151.01` / `151.98` / `153.15` | **`0.0`** | 较 NUMA 绑核前（`240.43 µs`）下降 **`22.0%`** |
| **`4096`** | **`40.0 MiB`** | **`4,274.36` / `4,289.52` / `4,292.75`** | `6,206.57` / `6,233.32` / `6,243.15` | **`0.0`** | **比降级 `NCCL SHM` 快 `31.1%`（`1.45×`）**，较绑核前（`5,922.64 µs`）下降 **`27.8%`** |

---

### 3.3 Phase 1：Mode B 控制面 IPC 与双缓冲重叠效率（200 步微基准）

来源：[`phase1_mode_b_1gpu.json`](phase1_mode_b_1gpu.json) 与 [`phase1_mode_b_8gpu.json`](phase1_mode_b_8gpu.json)

| 规模 | `worker_numa_nodes` | 发布唤醒耗时 (`mean_publish_signal_us`) | 控制面 IPC `p50` / `p95` / `p99` (`µs`) | 下一拍准备耗时 (`prepare_next_us`) | 隐藏准备耗时 (`hidden_prepare_us`) | 重叠隐藏率 (`overlap_hiding_efficiency`) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `1 GPU` | `[0]` | `3.14 µs` | `319` / `375` / `417` | `845 µs` | `845 µs` | **`100.0%`** |
| `8 GPU` | `[0, 0, 0, 1, 2, 2, 2, 3]` | `25.64 µs` | **`341` / `364` / `562`** | `1,016 µs` | `1,016 µs` | **`100.0%`** |

---

### 3.4 Phase 0：`sm_120` 核心算子单层微基准汇总

来源：[`step_01_dense_fp8_gemm.json`](step_01_dense_fp8_gemm.json)、[`step_03_moe_mxfp4_sm120.json`](step_03_moe_mxfp4_sm120.json)、[`step_04_dsv41_attention_sm120.json`](step_04_dsv41_attention_sm120.json)、[`step_05a_mhc_engram_sm120.json`](step_05a_mhc_engram_sm120.json)

| 算子模块 | 规格 | `M = 6` 延迟 | `M = 48` 延迟 | `M = 192` 延迟 | `M = 4096` 延迟 | 精度 (`cos_sim`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `RMSNorm + MXFP8 Group-32` | `H = 5120`, Triton | `3.40 µs` | `3.22 µs` | `4.00 µs` | `15.52 µs` | `0.99965` |
| `b12x MXFP8 Linear` | `5120 -> 2048` | `15.52 µs` | `22.17 µs` | `23.36 µs` | `76.47 µs` (`1,123 TFLOPS`) | `0.99964 ~ 1.0` |
| `dsv41_sqrt_softplus_topk` | `384` 专家, `top_k = 6` | `5.74 µs` | `5.79 µs` | `6.89 µs` | `23.88 µs` | `1.00000` |
| `b12x MXFP4 MoE (TP=8)` | `384` 专家, `local_inter = 288` | **`70.24 µs`** (`1,173 GB/s`) | `435.07 µs` | `781.73 µs` | `2,414.73 µs` | `0.99999` |
| `b12x MXFP4 MoE (EP=8)` | `48` 本地专家, `inter = 2304` | `320.55 µs` | `574.19 µs` | `842.18 µs` | `8,865.44 µs` | `0.99999` |
| `deepseek_v41 Sparse MLA` | `16` 头, `D = 512`, `KV = 32K` | `13.52 µs` (`B=1`) / `13.14 µs` (`B=8`) | `60.75 µs` (`B=8, q=6` verify) | — | `1,429.28 µs` (`B=32, q=128`) | `0.99958 ~ 0.99963` |
| `mHC (post_pre)` | `4` 流, `20` 步 Sinkhorn, `H = 5120` | `11.41 µs` | `26.66 µs` | `83.62 µs` | `1,846.08 µs` | `1.00000` |
| `Engram Lookup (Layer 1)` | `48M` 本地行 (`12.29 GB` 分片) | `4.42 µs` (VRAM) / `6.12 µs` (UVA) | `3.65 µs` / `20.03 µs` | `5.07 µs` / `68.88 µs` | `48.25 µs` / `2,400.12 µs` | `1.00000` |

---

## 4. 一键复现命令

```bash
# 1. 编译 Rust 控制面 Release 二进制
cargo build --release

# 2. 运行 Phase 0.5b 8 卡 NUMA 绑核 AllReduce 基准
torchrun --standalone --nproc_per_node=8 worker/tests/phase0/test_05_pcie_allreduce_8gpu.py

# 3. 运行 Phase 1 Mode B 8 卡 200 步重叠微基准
./target/release/zstep --mode-b --world-size 8 --steps 200 \
  --python /usr/bin/python3 --worker-root ./worker \
  --metrics-output docs/benchmarks/phase1_mode_b_8gpu.json

# 4. 运行 Phase 2 Qwen3.5-35B-A3B 8 卡真权重贪心逐 token 对齐测试
ZSTEP_QWEN35_MODEL_DIR=/path/to/Qwen3.5-35B-A3B \
  pytest worker/tests/phase2/test_qwen35_e2e_tp8.py -v

# 5. 运行 Phase 2 Qwen3.5-35B-A3B 全部 6 档并发连续批处理基准
./target/release/zstep --mode-b --world-size 8 \
  --python /usr/bin/python3 --worker-root ./worker \
  --model-dir /path/to/Qwen3.5-35B-A3B \
  --allreduce-blocks 8 --num-kv-pages 65536 \
  --max-seqs 128 --max-step-tokens 8192 --max-decoder-rows 8192 --max-pages-per-seq 256 \
  --bench-tiers "in512_c1:1:8:512:256;in512_c16:16:48:512:256;in512_c128:128:256:512:256;in1024_c1:1:20:1024:256;in1024_c32:32:200:1024:256;in1024_c128:128:400:1024:256" \
  --metrics-output docs/benchmarks/phase2_zstep_qwen35_35b_a3b_tp8.json
```
