# Copyright (c) 2026 zstep Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Native Qwen3.5 one-layer full-attention MoE multi-token predictor (CUDA-Graphed, zero NCCL)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from zstep_worker.env_config import WorkerRuntimeConfig
from zstep_worker.kernels_sm120.qwen35_ops import (
    paged_gqa_decode_gated,
    paged_gqa_prefill_varlen_gated,
    qwen35_attn_prologue,
    skinny_linear,
    sparse_moe_tp8_forward,
    zero_centered_rmsnorm,
)
from zstep_worker.models.qwen35_moe import (
    FullAttnLayerWeights,
    _SafetensorShardReader,
)
from zstep_worker.spec_decode import VerifiedSequence


class Qwen35MtpDrafter:
    def __init__(self, runner: Any) -> None:
        self.runner = runner
        self.device = runner.device
        self.eps = runner.config.rms_norm_eps
        self.sm_scale: float = runner.config.head_dim**-0.5
        self.num_draft_tokens: int = runner.num_spec_tokens
        self.k_cache = torch.zeros(
            (runner.num_kv_pages + 1, 16, 256),
            dtype=torch.bfloat16,
            device=self.device,
        )
        self.v_cache = torch.zeros_like(self.k_cache)
        self._load_weights()

        mt = runner.max_step_tokens
        k = self.num_draft_tokens
        self.embed = torch.empty((mt, 2048), dtype=torch.bfloat16, device=self.device)
        self.fc_in = torch.empty((mt, 4096), dtype=torch.bfloat16, device=self.device)
        self.residual = torch.empty((mt, 2048), dtype=torch.bfloat16, device=self.device)
        self.normed = torch.empty_like(self.residual)
        self.proj = torch.empty_like(self.residual)
        self.output = torch.empty_like(self.residual)
        self.qkv = torch.empty((mt, 1536), dtype=torch.bfloat16, device=self.device)
        self.q = torch.empty((mt, 512), dtype=torch.bfloat16, device=self.device)
        self.gate = torch.empty_like(self.q)
        self.attn = torch.empty_like(self.q)

        self.tokens_cpu = torch.zeros(mt, dtype=torch.int32, pin_memory=True)
        self.positions_cpu = torch.zeros(mt, dtype=torch.int32, pin_memory=True)
        self.slots_cpu = torch.zeros(mt, dtype=torch.int32, pin_memory=True)
        self.tokens_gpu = torch.zeros(mt, dtype=torch.int32, device=self.device)
        self.positions_gpu = torch.zeros(mt, dtype=torch.int32, device=self.device)
        self.slots_gpu = torch.zeros(mt, dtype=torch.int32, device=self.device)
        self.qsl_gpu = torch.zeros(2, dtype=torch.int32, device=self.device)
        self.seq_lens_gpu = torch.zeros(1, dtype=torch.int32, device=self.device)

        max_verify_m = k + 1
        self.target_hidden_in_buf = torch.zeros(
            (max_verify_m, 2048), dtype=torch.bfloat16, device=self.device
        )
        self.tail_hidden_buf = torch.zeros(
            (1, 2048), dtype=torch.bfloat16, device=self.device
        )
        self.step_positions_cpu = torch.zeros(k, dtype=torch.int32, pin_memory=True)
        self.step_seq_lens_cpu = torch.zeros(k, dtype=torch.int32, pin_memory=True)
        self.step_positions_gpu = torch.zeros(k, dtype=torch.int32, device=self.device)
        self.step_seq_lens_gpu = torch.zeros(k, dtype=torch.int32, device=self.device)
        self.draft_out_tokens_gpu = torch.zeros(
            (k, 1, 2), dtype=torch.int32, device=self.device
        )
        self.draft_out_logprobs_gpu = torch.zeros(
            (k, 1, 2), dtype=torch.int32, device=self.device
        )
        self.draft_out_tokens_cpu = torch.zeros(
            (max(1, runner.max_seqs), k), dtype=torch.int32, pin_memory=True
        )
        self.draft_out_tokens_batch_gpu = torch.zeros(
            (max(1, runner.max_seqs), k), dtype=torch.int32, device=self.device
        )

        self.mtp_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.mtp_tail_graph: torch.cuda.CUDAGraph | None = None
        self._pending_chunk_tail: dict[int, tuple[int, torch.Tensor]] = {}

    def _load_weights(self) -> None:
        runner = self.runner
        rank = runner.rank
        reader = _SafetensorShardReader(runner.model_dir)
        prefix = "mtp.layers.0."
        self.pre_fc_norm_embedding = reader.get_slice(
            "mtp.pre_fc_norm_embedding.weight"
        )[:].to(self.device, dtype=torch.bfloat16).contiguous()
        self.pre_fc_norm_hidden = reader.get_slice("mtp.pre_fc_norm_hidden.weight")[:].to(
            self.device, dtype=torch.bfloat16
        ).contiguous()
        self.fc_weight = reader.get_slice("mtp.fc.weight")[:].to(
            self.device, dtype=torch.bfloat16
        ).contiguous()
        self.final_norm = reader.get_slice("mtp.norm.weight")[:].to(
            self.device, dtype=torch.bfloat16
        ).contiguous()

        in_norm = reader.get_slice(f"{prefix}input_layernorm.weight")[:].to(
            self.device, dtype=torch.bfloat16
        ).contiguous()
        post_norm = reader.get_slice(f"{prefix}post_attention_layernorm.weight")[:].to(
            self.device, dtype=torch.bfloat16
        ).contiguous()
        router_w = reader.get_slice(f"{prefix}mlp.gate.weight")[:].to(
            self.device, dtype=torch.bfloat16
        )
        shared_gate = reader.get_slice(f"{prefix}mlp.shared_expert_gate.weight")[:].to(
            self.device, dtype=torch.bfloat16
        )
        shared_gate_pad = torch.zeros((8, 2048), dtype=torch.bfloat16, device=self.device)
        shared_gate_pad[0:1].copy_(shared_gate)
        shared_gp = reader.get_slice(f"{prefix}mlp.shared_expert.gate_proj.weight")[
            rank * 64 : (rank + 1) * 64, :
        ].to(self.device, dtype=torch.bfloat16)
        shared_up = reader.get_slice(f"{prefix}mlp.shared_expert.up_proj.weight")[
            rank * 64 : (rank + 1) * 64, :
        ].to(self.device, dtype=torch.bfloat16)
        router_shared = torch.cat(
            [router_w, shared_gate_pad, shared_gp, shared_up], dim=0
        ).contiguous()
        shared_down = reader.get_slice(f"{prefix}mlp.shared_expert.down_proj.weight")[
            :, rank * 64 : (rank + 1) * 64
        ].to(self.device, dtype=torch.bfloat16).contiguous()
        experts_gu_parts = []
        experts_down_parts = []
        for expert_idx in range(256):
            expert_prefix = f"{prefix}mlp.experts.{expert_idx}."
            gate = reader.get_slice(f"{expert_prefix}gate_proj.weight")[
                rank * 64 : (rank + 1) * 64, :
            ].to(self.device, dtype=torch.bfloat16)
            up = reader.get_slice(f"{expert_prefix}up_proj.weight")[
                rank * 64 : (rank + 1) * 64, :
            ].to(self.device, dtype=torch.bfloat16)
            experts_gu_parts.append(torch.cat([gate, up], dim=0))
            experts_down_parts.append(
                reader.get_slice(f"{expert_prefix}down_proj.weight")[
                    :, rank * 64 : (rank + 1) * 64
                ].to(self.device, dtype=torch.bfloat16)
            )
        experts_gu = torch.stack(experts_gu_parts).contiguous()
        experts_down = torch.stack(experts_down_parts).contiguous()

        kv_head = rank // 4
        qp = reader.get_slice(f"{prefix}self_attn.q_proj.weight")[
            rank * 1024 : (rank + 1) * 1024, :
        ].to(self.device, dtype=torch.bfloat16)
        kp = reader.get_slice(f"{prefix}self_attn.k_proj.weight")[
            kv_head * 256 : (kv_head + 1) * 256, :
        ].to(self.device, dtype=torch.bfloat16)
        vp = reader.get_slice(f"{prefix}self_attn.v_proj.weight")[
            kv_head * 256 : (kv_head + 1) * 256, :
        ].to(self.device, dtype=torch.bfloat16)
        q_norm = reader.get_slice(f"{prefix}self_attn.q_norm.weight")[:].to(
            self.device, dtype=torch.bfloat16
        ).contiguous()
        k_norm = reader.get_slice(f"{prefix}self_attn.k_norm.weight")[:].to(
            self.device, dtype=torch.bfloat16
        ).contiguous()
        out_proj = reader.get_slice(f"{prefix}self_attn.o_proj.weight")[
            :, rank * 512 : (rank + 1) * 512
        ].to(self.device, dtype=torch.bfloat16).contiguous()
        self.layer = FullAttnLayerWeights(
            input_norm_w=in_norm,
            w_qkv=torch.cat([qp, kp, vp], dim=0).contiguous(),
            q_norm_w=q_norm,
            k_norm_w=k_norm,
            w_o_proj=out_proj,
            post_attn_norm_w=post_norm,
            w_router_shared_gu=router_shared,
            w_shared_down=shared_down,
            w_experts_gate_up=experts_gu,
            w_experts_down=experts_down,
            kv_layer_idx=0,
        )
        reader.close()

    def reset_caches(self) -> None:
        self.k_cache.zero_()
        self.v_cache.zero_()
        self._pending_chunk_tail.clear()

    def close(self) -> None:
        self.mtp_graphs.clear()
        self.mtp_tail_graph = None
        self._pending_chunk_tail.clear()

    def _forward_gpu(
        self,
        *,
        tokens_gpu: torch.Tensor,
        positions_gpu: torch.Tensor,
        slots_gpu: torch.Tensor,
        target_hidden: torch.Tensor,
        seq_lens_gpu: torch.Tensor,
        rows: int,
    ) -> torch.Tensor:
        runner = self.runner
        embed = self.embed[:rows]
        fc_in = self.fc_in[:rows]
        residual = self.residual[:rows]
        normed = self.normed[:rows]
        out = self.output[:rows]

        torch.index_select(runner.embed_weight, 0, tokens_gpu, out=embed)
        zero_centered_rmsnorm(
            embed, self.pre_fc_norm_embedding, out=fc_in[:, :2048], eps=self.eps
        )
        zero_centered_rmsnorm(
            target_hidden, self.pre_fc_norm_hidden, out=fc_in[:, 2048:], eps=self.eps
        )
        torch.matmul(fc_in, self.fc_weight.t(), out=residual)
        zero_centered_rmsnorm(
            residual, self.layer.input_norm_w, out=normed, eps=self.eps
        )
        skinny_linear(normed, self.layer.w_qkv, out=self.qkv[:rows])
        qwen35_attn_prologue(
            self.qkv[:rows],
            positions_gpu,
            slots_gpu,
            runner.page_table_gpu,
            self.layer.q_norm_w,
            self.layer.k_norm_w,
            runner.cos_cache,
            runner.sin_cache,
            q_out=self.q[:rows],
            gate_out=self.gate[:rows],
            k_cache=self.k_cache,
            v_cache=self.v_cache,
            eps=self.eps,
        )
        if rows == 1:
            paged_gqa_decode_gated(
                self.q[:1],
                self.gate[:1],
                self.k_cache,
                self.v_cache,
                runner.page_table_gpu,
                seq_lens_gpu,
                slots_gpu,
                partial_acc=runner.partial_acc_buf[:1],
                partial_m=runner.partial_m_buf[:1],
                partial_l=runner.partial_l_buf[:1],
                out=self.attn[:1],
                num_active_seqs=1,
                sm_scale=self.sm_scale,
            )
            torch.matmul(self.attn[:1], self.layer.w_o_proj.t(), out=self.proj[:1])
            runner._all_reduce_add_norm_decode(
                self.proj[:1],
                residual,
                self.layer.post_attn_norm_w,
                normed=normed,
                reduced=runner.reduced_buf[:1],
                eps=self.eps,
            )
        else:
            paged_gqa_prefill_varlen_gated(
                self.q[:rows],
                self.gate[:rows],
                self.k_cache,
                self.v_cache,
                runner.page_table_gpu,
                self.qsl_gpu,
                seq_lens_gpu,
                slots_gpu,
                out=self.attn[:rows],
                num_seqs=1,
                max_q_len=rows,
                sm_scale=self.sm_scale,
            )
            torch.matmul(
                self.attn[:rows], self.layer.w_o_proj.t(), out=self.proj[:rows]
            )
            runner._all_reduce_add_norm_prefill(
                self.proj[:rows],
                residual,
                self.layer.post_attn_norm_w,
                normed=normed,
                reduced=runner.reduced_buf[:rows],
                eps=self.eps,
            )

        moe = sparse_moe_tp8_forward(
            normed,
            self.layer.w_router_shared_gu,
            self.layer.w_shared_down,
            self.layer.w_experts_gate_up,
            self.layer.w_experts_down,
            router_shared_buf=runner.router_shared_buf,
            topk_ids_buf=runner.topk_ids_buf,
            topk_weights_buf=runner.topk_weights_buf,
            shared_gate_scale_buf=runner.shared_gate_scale_buf,
            shared_inter_buf=runner.shared_inter_buf,
            shared_out_buf=runner.shared_out_buf,
            sorted_pair_ids_buf=runner.sorted_pair_ids_buf,
            expert_ids_per_tile_buf=runner.expert_ids_per_tile_buf,
            num_valid_tiles_buf=runner.num_valid_tiles_buf,
            moe_inter_buf=runner.moe_inter_buf,
            pair_out_buf=runner.pair_out_buf,
            moe_out_buf=runner.moe_out_buf,
            num_active_rows=rows,
        )
        if rows == 1:
            runner._all_reduce_add_norm_decode(
                moe,
                residual,
                self.final_norm,
                normed=out,
                reduced=runner.reduced_buf[:1],
                eps=self.eps,
            )
        else:
            runner._all_reduce_add_norm_prefill(
                moe,
                residual,
                self.final_norm,
                normed=out,
                reduced=runner.reduced_buf[:rows],
                eps=self.eps,
            )
        return out

    def _sample_into_step(self, hidden_1row: torch.Tensor, step_idx: int) -> None:
        runner = self.runner
        logits = runner.logits_local_buf[:1]
        torch.matmul(hidden_1row, runner.lm_head_weight.t(), out=logits)
        runner.argmax_channel.gather_argmax(
            logits,
            local_top2_vals=runner.local_max_vals_buf[:1],
            local_top2_ids=runner.local_max_ids_buf[:1],
            out_tokens=self.draft_out_tokens_gpu[step_idx],
            out_logprob_bits=self.draft_out_logprobs_gpu[step_idx],
            vocab_per_rank=runner.vocab_per_rank,
        )

    def _run_autoregressive_tail(
        self, start_hidden: torch.Tensor, *, steps: int | None = None
    ) -> None:
        current_hidden = start_hidden
        slot_1 = self.slots_gpu[:1]
        tail_steps = self.num_draft_tokens if steps is None else steps
        for step_idx in range(1, tail_steps):
            current_hidden = self._forward_gpu(
                tokens_gpu=self.draft_out_tokens_gpu[step_idx - 1, :1, 0],
                positions_gpu=self.step_positions_gpu[step_idx : step_idx + 1],
                slots_gpu=slot_1,
                target_hidden=current_hidden,
                seq_lens_gpu=self.step_seq_lens_gpu[step_idx : step_idx + 1],
                rows=1,
            )
            self._sample_into_step(current_hidden, step_idx)

    def _run_verify_draft_chain(self, m: int) -> None:
        out = self._forward_gpu(
            tokens_gpu=self.tokens_gpu[:m],
            positions_gpu=self.positions_gpu[:m],
            slots_gpu=self.slots_gpu[:m],
            target_hidden=self.target_hidden_in_buf[:m],
            seq_lens_gpu=self.step_seq_lens_gpu[:1],
            rows=m,
        )
        last_hidden = out[m - 1 : m]
        self._sample_into_step(last_hidden, 0)
        if self.num_draft_tokens > 1:
            self._run_autoregressive_tail(last_hidden)

    def capture_graphs(
        self,
        graph_pool: Any,
        capture_stream: torch.cuda.Stream,
    ) -> None:
        if WorkerRuntimeConfig.from_env().disable_draft_graphs:
            return
        runner = self.runner
        k = self.num_draft_tokens

        # Warm up varlen specialization buckets (`rows > 1`, `% 16 == 0` and `% 16 != 0`)
        # so prefill MTP context writes never trigger Triton compilation inside RuntimeLoop.
        for warm_m in (5, 16):
            if warm_m > runner.max_step_tokens:
                continue
            self.tokens_gpu[:warm_m].zero_()
            self.positions_gpu[:warm_m].copy_(
                torch.arange(warm_m, dtype=torch.int32, device=self.device)
            )
            self.slots_gpu[:warm_m].fill_(runner.dummy_slot)
            self.qsl_gpu[0] = 0
            self.qsl_gpu[1] = warm_m
            self.seq_lens_gpu[0] = warm_m
            self._forward_gpu(
                tokens_gpu=self.tokens_gpu[:warm_m],
                positions_gpu=self.positions_gpu[:warm_m],
                slots_gpu=self.slots_gpu[:warm_m],
                target_hidden=runner.normed_buf[:warm_m],
                seq_lens_gpu=self.seq_lens_gpu,
                rows=warm_m,
            )
            torch.cuda.synchronize(self.device)
            dist.barrier(group=runner.exchange_group)

        for m in range(1, k + 2):
            if m > runner.max_step_tokens:
                break
            self.tokens_gpu[:m].zero_()
            self.positions_gpu[:m].copy_(
                torch.arange(m, dtype=torch.int32, device=self.device)
            )
            self.slots_gpu[:m].fill_(runner.dummy_slot)
            self.target_hidden_in_buf[:m].zero_()
            self.qsl_gpu[0] = 0
            self.qsl_gpu[1] = m
            base_pos = m - 1
            step_pos = torch.arange(base_pos, base_pos + k, dtype=torch.int32)
            self.step_positions_gpu.copy_(step_pos.to(self.device))
            self.step_seq_lens_gpu.copy_((step_pos + 1).to(self.device))

            with torch.cuda.stream(capture_stream):
                for _ in range(2):
                    self._run_verify_draft_chain(m)
            capture_stream.synchronize()
            dist.barrier(group=runner.exchange_group)

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=graph_pool, stream=capture_stream):
                self._run_verify_draft_chain(m)
            capture_stream.synchronize()
            dist.barrier(group=runner.exchange_group)
            self.mtp_graphs[m] = g

        if k > 1:
            self.slots_gpu[:1].fill_(runner.dummy_slot)
            self.tail_hidden_buf.zero_()
            self.draft_out_tokens_gpu.zero_()
            step_pos = torch.arange(1, 1 + k, dtype=torch.int32)
            self.step_positions_gpu.copy_(step_pos.to(self.device))
            self.step_seq_lens_gpu.copy_((step_pos + 1).to(self.device))
            with torch.cuda.stream(capture_stream):
                for _ in range(2):
                    self._run_autoregressive_tail(self.tail_hidden_buf)
            capture_stream.synchronize()
            dist.barrier(group=runner.exchange_group)

            tail_g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(tail_g, pool=graph_pool, stream=capture_stream):
                self._run_autoregressive_tail(self.tail_hidden_buf)
            capture_stream.synchronize()
            dist.barrier(group=runner.exchange_group)
            self.mtp_tail_graph = tail_g

        self.reset_caches()

    def _stage_and_run_prefill_context(
        self,
        *,
        tokens: np.ndarray,
        positions: np.ndarray,
        req_slot: int,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        rows = int(tokens.shape[0])
        np.copyto(self.tokens_cpu[:rows].numpy(), tokens)
        np.copyto(self.positions_cpu[:rows].numpy(), positions)
        self.slots_cpu[:rows].fill_(req_slot)
        self.tokens_gpu[:rows].copy_(self.tokens_cpu[:rows], non_blocking=True)
        self.positions_gpu[:rows].copy_(self.positions_cpu[:rows], non_blocking=True)
        self.slots_gpu[:rows].copy_(self.slots_cpu[:rows], non_blocking=True)
        self.qsl_gpu[0] = 0
        self.qsl_gpu[1] = rows
        self.seq_lens_gpu[0] = int(positions[-1]) + 1
        return self._forward_gpu(
            tokens_gpu=self.tokens_gpu[:rows],
            positions_gpu=self.positions_gpu[:rows],
            slots_gpu=self.slots_gpu[:rows],
            target_hidden=hidden,
            seq_lens_gpu=self.seq_lens_gpu,
            rows=rows,
        )

    def draft(
        self,
        verified: list[VerifiedSequence],
        *,
        token_ids: np.ndarray,
        positions: np.ndarray,
        req_slot_ids: np.ndarray,
        query_start_loc: np.ndarray,
        target_hidden: torch.Tensor,
        apc_restored_seq_indices: set[int] | frozenset[int] | None = None,
    ) -> dict[int, list[int]]:
        by_seq = {sequence.batch_index: sequence for sequence in verified}
        restored_seqs = apc_restored_seq_indices or ()
        k = self.num_draft_tokens
        max_pt_tokens = self.runner.max_pages_per_seq * 16
        max_ctx = min(
            int(getattr(self.runner, "max_model_len", max_pt_tokens)),
            max_pt_tokens,
        )
        drafted_specs: list[tuple[int, int]] = []

        for seq_idx in range(len(query_start_loc) - 1):
            begin = int(query_start_loc[seq_idx])
            end = int(query_start_loc[seq_idx + 1])
            if end <= begin:
                continue
            sequence = by_seq.get(seq_idx)
            start_pos = int(positions[begin])
            is_continuing_prefill = (
                start_pos > 0
                and seq_idx not in restored_seqs
                and (sequence is None or len(sequence.rows) == 1)
            )

            if sequence is None:
                # Partial chunked prefill: shift `token_ids` left by 1 against `target_hidden`,
                # saving the final token's hidden state until the next chunk arrives.
                req_slot = int(req_slot_ids[begin])
                prev_tail = self._pending_chunk_tail.pop(req_slot, None)
                if (
                    prev_tail is not None
                    and is_continuing_prefill
                    and prev_tail[0] + 1 == start_pos
                    and prev_tail[0] < max_ctx
                ):
                    prev_pos, prev_h = prev_tail
                    self._stage_and_run_prefill_context(
                        tokens=np.array([token_ids[begin]], dtype=np.int32),
                        positions=np.array([prev_pos], dtype=np.int32),
                        req_slot=req_slot,
                        hidden=prev_h,
                    )
                if end - begin > 1:
                    self._stage_and_run_prefill_context(
                        tokens=token_ids[begin + 1 : end].astype(np.int32, copy=False),
                        positions=positions[begin : end - 1].astype(np.int32, copy=False),
                        req_slot=req_slot,
                        hidden=target_hidden[begin : end - 1],
                    )
                self._pending_chunk_tail[req_slot] = (
                    int(positions[end - 1]),
                    target_hidden[end - 1 : end].clone(),
                )
                continue

            req_slot = sequence.req_slot
            prev_tail = self._pending_chunk_tail.pop(req_slot, None)
            if (
                prev_tail is not None
                and is_continuing_prefill
                and prev_tail[0] + 1 == start_pos
                and prev_tail[0] < max_ctx
            ):
                prev_pos, prev_h = prev_tail
                self._stage_and_run_prefill_context(
                    tokens=np.array([token_ids[begin]], dtype=np.int32),
                    positions=np.array([prev_pos], dtype=np.int32),
                    req_slot=req_slot,
                    hidden=prev_h,
                )

            if len(sequence.rows) == 1 and end - begin > 1:
                # Full / final prefill chunk: `[p_1, ..., p_{L-1}, t_0]` paired with `h[0..L-1]`.
                mtp_tokens = np.empty(end - begin, dtype=np.int32)
                mtp_tokens[:-1] = token_ids[begin + 1 : end]
                mtp_tokens[-1] = sequence.accepted_tokens[0]
                mtp_positions = positions[begin:end].astype(np.int32, copy=False)
                mtp_hidden = target_hidden[begin:end]
            else:
                # Speculative verify or 1-token decode: `accepted_tokens` are already shifted +1.
                m_acc = len(sequence.accepted_tokens)
                mtp_tokens = np.asarray(sequence.accepted_tokens, dtype=np.int32)
                acc_rows = sequence.rows[:m_acc]
                r0 = acc_rows[0]
                if acc_rows == tuple(range(r0, r0 + m_acc)):
                    mtp_positions = positions[r0 : r0 + m_acc].astype(np.int32, copy=False)
                    mtp_hidden = target_hidden[r0 : r0 + m_acc]
                else:
                    row_list = list(acc_rows)
                    mtp_positions = positions[row_list].astype(np.int32, copy=False)
                    mtp_hidden = target_hidden[
                        torch.tensor(row_list, dtype=torch.int64, device=self.device)
                    ]

            valid_ctx = (mtp_positions >= 0) & (mtp_positions < max_ctx)
            if not bool(np.all(valid_ctx)):
                if not bool(np.any(valid_ctx)):
                    continue
                mtp_tokens = mtp_tokens[valid_ctx]
                mtp_positions = mtp_positions[valid_ctx]
                mtp_hidden = mtp_hidden[
                    torch.from_numpy(np.flatnonzero(valid_ctx)).to(self.device)
                ]

            m = int(mtp_tokens.shape[0])
            last_pos = int(mtp_positions[-1])
            avail_k = min(k, max_ctx - (last_pos + 1))
            if avail_k <= 0:
                continue
            step_pos_np = self.step_positions_cpu.numpy()
            step_sl_np = self.step_seq_lens_cpu.numpy()
            for s_i in range(avail_k):
                p = last_pos + s_i
                step_pos_np[s_i] = p
                step_sl_np[s_i] = p + 1
            self.step_positions_gpu.copy_(self.step_positions_cpu, non_blocking=True)
            self.step_seq_lens_gpu.copy_(self.step_seq_lens_cpu, non_blocking=True)

            if avail_k == k and m in self.mtp_graphs:
                np.copyto(self.tokens_cpu[:m].numpy(), mtp_tokens)
                np.copyto(self.positions_cpu[:m].numpy(), mtp_positions)
                self.slots_cpu[:m].fill_(req_slot)
                self.tokens_gpu[:m].copy_(self.tokens_cpu[:m], non_blocking=True)
                self.positions_gpu[:m].copy_(self.positions_cpu[:m], non_blocking=True)
                self.slots_gpu[:m].copy_(self.slots_cpu[:m], non_blocking=True)
                self.target_hidden_in_buf[:m].copy_(mtp_hidden)
                self.qsl_gpu[0] = 0
                self.qsl_gpu[1] = m
                self.mtp_graphs[m].replay()
            else:
                out = self._stage_and_run_prefill_context(
                    tokens=mtp_tokens,
                    positions=mtp_positions,
                    req_slot=req_slot,
                    hidden=mtp_hidden,
                )
                last_hidden = out[m - 1 : m]
                self._sample_into_step(last_hidden, 0)
                if avail_k > 1:
                    if avail_k == k and self.mtp_tail_graph is not None:
                        self.slots_cpu[:1].fill_(req_slot)
                        self.slots_gpu[:1].copy_(self.slots_cpu[:1], non_blocking=True)
                        self.tail_hidden_buf.copy_(last_hidden)
                        self.mtp_tail_graph.replay()
                    else:
                        self.slots_cpu[:1].fill_(req_slot)
                        self.slots_gpu[:1].copy_(self.slots_cpu[:1], non_blocking=True)
                        self._run_autoregressive_tail(last_hidden, steps=avail_k)

            out_idx = len(drafted_specs)
            self.draft_out_tokens_batch_gpu[out_idx].copy_(
                self.draft_out_tokens_gpu[:, 0, 0]
            )
            drafted_specs.append((req_slot, avail_k))

        if not drafted_specs or self.runner.rank != 0:
            return {}
        n_drafted = len(drafted_specs)
        self.draft_out_tokens_cpu[:n_drafted].copy_(
            self.draft_out_tokens_batch_gpu[:n_drafted], non_blocking=True
        )
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()
        rows_list = self.draft_out_tokens_cpu[:n_drafted].tolist()
        return {
            slot: [int(tok) for tok in rows_list[i][:avail_k]]
            for i, (slot, avail_k) in enumerate(drafted_specs)
        }
