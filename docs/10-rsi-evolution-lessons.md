# 10 — 从 `zstep` 全栈优化看推理自进化框架（RSI）工程提炼

本文档复盘 `zstep` 从单机算子微基准、非 P2P 8 卡共享内存引擎、在线服务化（`zs-server serve`）、混合架构自动前缀缓存（APC）到原生投机解码（`dspark` / `mtp` / `ngram`）与 Nebula 全栈联调的完整演进过程，提炼面向 **大模型推理递归自优化（Recursive Self-Improvement, RSI）框架** 的 8 条核心工程准则。

量化性能结果参见 [`benchmark.md`](benchmark.md)。

---

## 1. 严防“劣化基线式”奖励作弊（Frozen Baseline Anchor）

### 实战复盘
在 `DeepSeek-V4.1-Flash`（`dspark K=5`）与 `Qwen3.5-35B-A3B`（`mtp K=3`）投机解码无人值守调优阶段，验证步采用 `M = K + 1` Tensor Core 批量矩阵乘，而默认非投机路径（`--speculative-method none`）采用 `M = 1` 专用 GEMV + CUDA Graph。两者在 BF16/FP32 归约累加顺序上存在 `< 1 ULP` 差异，长文本生成在 top-1 / top-2 对数概率平局位置会出现合理分叉。

若自动验证器机械要求“所有长文本上 `speculative == none` 100% 逐 token 相等”，无人值守编码代理会倾向于**将 `--speculative-method none` 默认路径也降级为慢速多行模式甚至关闭 CUDA Graph**（实测曾将 `Qwen3.5` 默认基线从 `312 tok/s` 拉低至 `222 tok/s`），从而在不提升绝对性能的情况下凑出“一致性通过 + 虚假相对加速比”。

### 框架设计准则
1. **不可变基线锚点**：候选补丁的一切相对加速比必须锚定**父代冻结 SHA 的黄金基线数据**，严禁仅与候选分支自身跑出的 `--speculative-method none` 做比值；同时必须对默认非投机路径施加严格的零退化门禁（退化上限 $\le 0.5\%$）。
2. **双层等价性门控**：
   - **低熵确定性探针集**（如素数平方表、结构化 JSON 提取、原文复述）：强制 **100% 逐 token 精确匹配**；
   - **高熵开放生成集**：采用 **单步 Teacher-Forcing Top-1 / Logit 漂移门控**，区分真正的“状态回滚污染 Bug”与正常的“跨 batch tile `< 1 ULP` 浮点平局分叉”。

---

## 2. 拆解“算法收益”与“墙钟收益”背离（Algorithmic $\tau$ vs. Wall-Clock Trap）

### 实战复盘
投机解码首版功能跑通时，算法接受长度已达 `DSpark` $\tau = 3.53$、`MTP` $\tau = 2.58$，但单流端到端吞吐反而低于不投机基线（`246 < 312 tok/s`）。根因并非草稿模型接受率不足，而是草稿步残留了 **Python eager 算子发射、`.item()` / `.cpu()` 主机同步以及跨 PCIe 的 `dist.broadcast`（约 `50 µs/次`）**。将验证步与草稿步全部收进分段 **CUDA Graph** 并改为 [`shm_allreduce.py`](../worker/zstep_worker/kernels_sm120/shm_allreduce.py) 全 rank 一致采样后，`Qwen3.5` 与 `DeepSeek-V4.1` 吞吐立即跃升至 **`762.5 tok/s`（`2.45×`）** 与 **`395.1 tok/s`（`3.35×`）**。

同理，在已被 CUDA Graph 覆盖的解码路径外围合并细碎主机拷贝，实测端到端收益不足 `±0.3%`，属于典型的低杠杆无效生态位。

### 框架设计准则
1. **热路径零同步静态门禁**：在提交远端 GPU 评测前，对模型步进热路径（[`deepseek_v41.py`](../worker/zstep_worker/models/deepseek_v41.py)、[`qwen35_moe.py`](../worker/zstep_worker/models/qwen35_moe.py)、[`spec_decode.py`](../worker/zstep_worker/spec_decode.py)）做 AST 扫描，直接拦截新增的 `.item()`、`.cpu()`、`.tolist()`、`dist.broadcast` 及禁用 CUDA Graph 的改动。
2. **双维诊断反射卡**：同时向进化代理反馈**算法效率指标**（平均接受长度 $\tau$、前缀命中率、解码重叠步占比）与**单步墙钟分解**。当 $\tau \ge 2.2$ 但墙钟加速 $< 1.05\times$ 时，明确指示代理消除同步并捕获 CUDA Graph，而非盲目调整草稿步宽 $K$。

---

## 3. 面向混合递归架构的“组合状态不变量”验证

### 实战复盘
现代前沿模型已不再是纯无状态 GQA Transformer：`Qwen3.5` 含 30 层递归 `GatedDeltaNet` `(conv_state, ssm_state_vk)` 状态，`DeepSeek-V4.1` 含 MLA + SWA + DSA Indexer + 3-token `Engram` 窗口，叠加在线 APC（[`zs-kv`](../crates/zs-kv)）与投机解码跨页回滚（[`zs-scheduler`](../crates/zs-scheduler)）后，独立单测难以覆盖组合边界。全过程中独立只读评审捕获了三类典型组合缺陷：

1. **零初始化共享内存哨兵冲突**：POSIX SHM 默认全 `0`；当 `--no-enable-prefix-caching` 与 `--speculative-method mtp` 组合时，未写入的 `compressor_slot_ids = 0` 若缺少 `num_recurrent_checkpoints > 0` 守卫，会被误判为合法的第 `0` 号快照槽恢复，静默污染后续请求的 30 层线性注意力状态。
2. **块边界单 Token 形状歧义**：当 `65`-token 提示词命中 `64`-token APC 节点仅剩 `1` 个尾部 token 时，`num_sample == total_tokens` 条件与单序列解码重合；必须同时检查显式预填充标志，防止误走投机验证分支丢弃刚命中的 GDN 快照。
3. **运行期租约晋升与跨页回滚守恒**：冷启动请求在运行中将已完成块从 `kv_leases` 晋升至 `prefix_lease` 后，投机拒绝跨页回滚必须满足不变量：
   $$\text{page\_ids.len()} == \text{prefix\_lease.pages()} + \text{kv\_leases.len()}$$
   若按准入时的静态 `cached_prefix_tokens` 计算截断长度，每次跨页草稿拒绝都会静默泄漏物理 KV 页。

### 框架设计准则
- **组合混沌金丝雀（Compositional Chaos Canary）**：自动验证流水线不能只测稳态独立请求，必须内置 `< 5s` 的状态串扰序列（`冷启动长前缀晋升 -> 跨页草稿拒绝 -> 流式中途断连 Abort -> block_size + 1 精确边界请求`），并在收尾时硬性断言 `idle_kv_cache_usage == 0.0`（零物理页泄漏、零跨槽位串扰）。

---

## 4. 在静态门禁与全量权重压测之间增设微缩 GPU 冒烟层

### 实战复盘
在多代算子与模型 Runner 自动演化记录中，大量候选补丁能通过本地 `cargo check`、`ruff` 与纯 CPU `pytest`，但在远端 8 卡加载 `475 GiB` 全量权重后因 PyTorch CUDA 算子约束崩溃（例如 `index_copy_()` 强制要求 `int64` 索引而非 `int32`、`torch.where(..., out=...)` 重载不接受 Python 数值字面量）。

### 框架设计准则
- **两级预筛（AST 类型重载拦截 + 微缩随机权重冒烟）**：
  1. 在本地 Red-Team 阶段用 AST 规则直接拦截 `torch.where(..., out=...)` 标量参数与 `int32` 索引张量误用；
  2. 在远端全量权重压测前，支持基于微缩层数或容器内轻量 GPU 单测完成秒级 CUDA 图捕获冒烟，将昂贵的 8 卡全模型压测留给真正具备运行资格的候选补丁。

---

## 5. 以“线协议预留槽”支撑跨平面协同演化

### 实战复盘
本次带来最大端到端收益的三次跃迁（`Route A 零同步重叠`、`64-token GQA+GDN 混合 APC`、`一步多 token 投机验证`）均横跨了 **Rust 调度面（[`zs-scheduler`](../crates/zs-scheduler)）+ SHM 线协议（[`zs-wire`](../crates/zs-wire)）+ GPU 算子/模型面（[`zstep_worker`](../worker/zstep_worker)）**。之所以能在不破坏 C ABI 偏移量的前提下快速演进，得益于 `zs-wire` 在 `StepResultSlot` 内预先保留了固定宽度的 `SeqStepOutput`（`accepted_tokens: [i32; 8]`、`next_draft_tokens: [i32; 8]`）与 `StepHeader.flags` 位域。

### 框架设计准则
- 自进化框架不应局限于“单文件孤立变异”，而应以 **受控战役（Campaign）** 为单位声明协同演化的白名单路径与跨边界不变量（Agreement / Isolation / Integrity），并在底层线协议中预留通用元数据槽位以避免破坏二进制布局。

---

## 6. “生成者 $\neq$ 评审者”的独立只读对抗门禁

### 实战复盘
在在线服务化、APC 与投机解码三次大版本交付中，生成代理完成代码并通过全部本地单测后，每次均由**独立上下文、零写权限的只读 Critic** 检出了生成者自身盲视的边界缺陷（如 `in_flight_depth=1` 提前返回跳过 EOS 终止、多 token 流式块下路由层 ITL 分母漏除 token 数等）。

### 框架设计准则
- 在高成本 8 卡多轮交叉验证前，结合确定性静态扫描与独立只读 Critic 反例推演，迫使候选补丁在进入物理集群前暴露出索引越界、异步生命周期与槽位复用初值漏洞。

---

## 7. 远端物理集群净室卫生与根因无损归档

### 实战复盘
长时间无人值守压测极易受环境噪声干扰：
- 宿主机绝对路径软链挂载进容器后指向失效，会导致 Triton 缓存目录不可达并触发意外重编；
- 异常中断若未清理容器与 `/dev/shm/zstep*` 共享内存段，会污染下一代压测；
- 若错误摘要只截取日志最后 3 行，引擎退出时打印的 JSON 闭合括号与 `TRITON_RECOMPILATIONS=0` 横幅会掩盖真正的 Python `Traceback` 或 Rust `panic` 行，导致谱系记忆丢失真实根因。

### 框架设计准则
- 压测前后强制执行 **8 卡显存归零 + `/dev/shm` 清理断言**；日志归档器必须优先提取异常栈顶、断言失败与贪心漂移指标，而非机械截取末尾固定行数。

---

## 8. 接通“自指加速飞轮”（Self-Referential Ignition）

推理自进化框架的外环 Proposer 负载天然具备两大物理特征：
1. **高前缀重合度（`> 90%`）**：同一战役下的多代探索共享完全相同的系统约束、硬件 Roofline 反射卡、压缩谱系历史与父代源码上下文；
2. **高输出复述度（`> 80%`）**：候选补丁或重构函数的大部分代码结构直接源自输入上下文中的父代实现。

随着 `zstep` 主干同时具备 **APC（`5.2×–16.2×` TTFT 加速）** 与 **`dspark` / `mtp` / `ngram`（代码与上下文重写 `2.0×–3.35×` 解码提速）**，自优化框架可直接通过 OpenAI 兼容接口将外环 Proposer 托管在进化后的 `zstep` 服务之上，形成 **“`zstep` 加速外环 Proposer $\rightarrow$ 单位时间探索更多候选代数 $\rightarrow$ 持续反哺 `zstep` 算子与调度器”** 的自指正反馈闭环。
