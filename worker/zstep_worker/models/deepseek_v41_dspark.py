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

"""DeepSeek-V4.1 native auxiliary-hidden DSpark drafting (CUDA-Graphed, zero NCCL)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from zstep_worker.env_config import WorkerRuntimeConfig
from zstep_worker.shm_bridge import DECODE_GRAPH_BUCKETS, select_decode_bucket
from zstep_worker.spec_decode import VerifiedSequence


class DeepSeekV41DSparkDrafter:
    def __init__(self, runner: Any) -> None:
        self.runner = runner
        self.model = runner.dspark_model
        config = runner.vllm_config.speculative_config.draft_model_config.hf_config
        self.noise_token_id: int = int(config.dspark_noise_token_id)
        self.num_draft_tokens: int = runner.num_spec_tokens
        self.layer_list: list[str] = list(self.model.get_draft_kv_cache_layer_names())
        self.layer_names: set[str] = set(self.layer_list)
        self.num_draft_layers: int = len(self.layer_list)

        vpr = runner.vocab_per_rank
        rank = runner.rank
        markov_head = self.model.model.markov_head
        self.markov_w1_weight: torch.Tensor = markov_head.markov_w1.weight
        self.markov_w2_shard: torch.Tensor = markov_head.markov_w2.weight[
            rank * vpr : (rank + 1) * vpr
        ].contiguous()

        k = self.num_draft_tokens
        max_draft_rows = min(256, runner.max_step_tokens)
        self.draft_buckets: list[int] = [
            b
            for b in DECODE_GRAPH_BUCKETS
            if b <= runner.max_graph_bucket and b * k <= max_draft_rows
        ]
        max_b = self.draft_buckets[-1] if self.draft_buckets else max(1, runner.max_seqs)
        max_rows = max_b * k

        self.draft_qsl_cpu: torch.Tensor = (
            torch.arange(max_b + 1, dtype=torch.int32) * k
        )
        self.draft_qsl_gpu: torch.Tensor = self.draft_qsl_cpu.to(runner.device)
        self.draft_tokens_cpu: torch.Tensor = torch.full(
            (max_rows,), self.noise_token_id, dtype=torch.int32, pin_memory=True
        )
        self.draft_tokens_gpu: torch.Tensor = torch.full(
            (max_rows,), self.noise_token_id, dtype=torch.int32, device=runner.device
        )
        self.draft_positions_cpu: torch.Tensor = torch.zeros(
            max_rows, dtype=torch.int64, pin_memory=True
        )
        self.draft_positions_gpu: torch.Tensor = torch.zeros(
            max_rows, dtype=torch.int64, device=runner.device
        )
        self.draft_slot_mapping_cpu: torch.Tensor = torch.full(
            (max_rows,), -1, dtype=torch.int64, pin_memory=True
        )
        self.draft_slot_mapping_gpu: torch.Tensor = torch.full(
            (max_rows,), -1, dtype=torch.int64, device=runner.device
        )
        self.ctx_positions_cpu: torch.Tensor = torch.zeros(
            max(runner.max_step_tokens, 8192), dtype=torch.int64, pin_memory=True
        )
        self.ctx_positions_gpu: torch.Tensor = torch.zeros(
            max(runner.max_step_tokens, 8192), dtype=torch.int64, device=runner.device
        )
        self.ctx_slot_mapping_cpu: torch.Tensor = torch.full(
            (max(runner.max_step_tokens, 8192),), -1, dtype=torch.int64, pin_memory=True
        )
        self.ctx_slot_mapping_gpu: torch.Tensor = torch.full(
            (max(runner.max_step_tokens, 8192),),
            -1,
            dtype=torch.int64,
            device=runner.device,
        )
        self.ctx_rows_cpu: torch.Tensor = torch.zeros(
            max(runner.max_step_tokens, 8192), dtype=torch.int64, pin_memory=True
        )
        self.ctx_rows_gpu: torch.Tensor = torch.zeros(
            max(runner.max_step_tokens, 8192), dtype=torch.int64, device=runner.device
        )

        self.step_logits_buf: torch.Tensor = torch.empty(
            (max_b, vpr), dtype=torch.bfloat16, device=runner.device
        )
        self.draft_out_tokens_gpu: torch.Tensor = torch.zeros(
            (k, max_b, 2), dtype=torch.int32, device=runner.device
        )
        self.draft_out_logprobs_gpu: torch.Tensor = torch.zeros(
            (k, max_b, 2), dtype=torch.int32, device=runner.device
        )
        self.draft_out_tokens_cpu: torch.Tensor = torch.zeros(
            (k, max_b), dtype=torch.int32, pin_memory=True
        )
        self.draft_graphs: dict[int, torch.cuda.CUDAGraph] = {}

    def _run_draft_forward_and_markov(self, b: int) -> None:
        runner = self.runner
        k = self.num_draft_tokens
        vpr = runner.vocab_per_rank
        rows = b * k

        head_hidden = self.model(
            input_ids=self.draft_tokens_gpu[:rows],
            positions=self.draft_positions_gpu[:rows],
        )
        normed = self.model.model.norm(head_hidden)
        base_logits = runner.model.compute_logits_local(normed).view(b, k, vpr)
        prev_tok = self.draft_tokens_gpu[0:rows:k]
        for step_idx in range(k):
            markov_vec = torch.index_select(self.markov_w1_weight, 0, prev_tok)
            self.step_logits_buf[:b].copy_(base_logits[:, step_idx, :])
            self.step_logits_buf[:b].addmm_(markov_vec, self.markov_w2_shard.t())
            runner.argmax_channel.gather_top2_from_logits(
                self.step_logits_buf[:b],
                local_top2_vals_buf=runner.local_max_vals_buf[:b],
                local_top2_ids_buf=runner.local_max_ids_buf[:b],
                out_tokens=self.draft_out_tokens_gpu[step_idx, :b],
                out_logprob_bits=self.draft_out_logprobs_gpu[step_idx, :b],
                vocab_per_rank=vpr,
                all_ranks=True,
            )
            prev_tok = self.draft_out_tokens_gpu[step_idx, :b, 0]

    @torch.inference_mode()
    def capture_graphs(
        self,
        graph_pool: Any,
        capture_stream: torch.cuda.Stream,
    ) -> None:
        if WorkerRuntimeConfig.from_env().disable_draft_graphs:
            return
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import set_forward_context

        runner = self.runner
        k = self.num_draft_tokens
        runner.mla_bt_gpu.zero_()
        runner.swa_bt_gpu.zero_()
        runner.comp_bt_gpu.fill_(runner.dummy_comp_block)

        for b in self.draft_buckets:
            rows = b * k
            self.draft_tokens_gpu[:rows].fill_(self.noise_token_id)
            self.draft_tokens_gpu[0:rows:k].zero_()
            self.draft_positions_gpu[:rows].copy_(
                torch.arange(rows, dtype=torch.int64, device=runner.device) % k
            )
            self.draft_slot_mapping_gpu[:rows].fill_(-1)
            runner.seq_lens_gpu[:b].fill_(k)
            runner.seq_lens_gpu[b:].zero_()
            sl_cpu = torch.full((b,), k, dtype=torch.int32)

            with torch.cuda.stream(capture_stream):
                for _ in range(2):
                    attn_meta = runner._build_step_attn_metadata(
                        num_reqs=b,
                        num_tokens=rows,
                        max_query_len=k,
                        max_seq_len=runner.decode_graph_max_seq_len,
                        qsl_gpu=self.draft_qsl_gpu[: b + 1],
                        qsl_cpu=self.draft_qsl_cpu[: b + 1],
                        seq_lens_gpu=runner.seq_lens_gpu[:b],
                        seq_lens_cpu=sl_cpu,
                        positions_gpu=self.draft_positions_gpu[:rows],
                        slot_mapping_gpu=self.draft_slot_mapping_gpu[:rows],
                        is_prefilling_cpu=runner.decode_is_prefilling_cpu[:b],
                        for_capture=True,
                        causal=False,
                        only_layer_names=self.layer_names,
                    )
                    with set_forward_context(
                        attn_meta,
                        runner.vllm_config,
                        num_tokens=rows,
                        cudagraph_runtime_mode=CUDAGraphMode.NONE,
                    ):
                        self._run_draft_forward_and_markov(b)
            capture_stream.synchronize()
            dist.barrier(group=runner.exchange_group)

            with torch.cuda.stream(capture_stream):
                attn_meta = runner._build_step_attn_metadata(
                    num_reqs=b,
                    num_tokens=rows,
                    max_query_len=k,
                    max_seq_len=runner.decode_graph_max_seq_len,
                    qsl_gpu=self.draft_qsl_gpu[: b + 1],
                    qsl_cpu=self.draft_qsl_cpu[: b + 1],
                    seq_lens_gpu=runner.seq_lens_gpu[:b],
                    seq_lens_cpu=sl_cpu,
                    positions_gpu=self.draft_positions_gpu[:rows],
                    slot_mapping_gpu=self.draft_slot_mapping_gpu[:rows],
                    is_prefilling_cpu=runner.decode_is_prefilling_cpu[:b],
                    for_capture=True,
                    causal=False,
                    only_layer_names=self.layer_names,
                )
            capture_stream.synchronize()
            g = torch.cuda.CUDAGraph()
            with set_forward_context(
                attn_meta,
                runner.vllm_config,
                num_tokens=rows,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
            ):
                with torch.cuda.graph(g, pool=graph_pool, stream=capture_stream):
                    self._run_draft_forward_and_markov(b)
            capture_stream.synchronize()
            dist.barrier(group=runner.exchange_group)
            self.draft_graphs[b] = g

    def close(self) -> None:
        self.draft_graphs.clear()

    def _store_contexts_batched(
        self,
        *,
        main_x: torch.Tensor,
        ctx_rows: np.ndarray,
        ctx_slots: np.ndarray,
        positions: np.ndarray,
        page_table: np.ndarray,
    ) -> None:
        n_ctx = int(ctx_rows.shape[0])
        if n_ctx == 0:
            return
        max_ctx = min(self.runner.max_model_len, page_table.shape[1] * 128)
        ctx_pos_i64 = positions[ctx_rows].astype(np.int64)
        valid = (
            (ctx_pos_i64 >= 0)
            & (ctx_pos_i64 < max_ctx)
            & ((ctx_pos_i64 // 128) < page_table.shape[1])
        )
        if not bool(np.all(valid)):
            ctx_rows = ctx_rows[valid]
            ctx_slots = ctx_slots[valid]
            ctx_pos_i64 = ctx_pos_i64[valid]
            n_ctx = int(ctx_rows.shape[0])
            if n_ctx == 0:
                return
        physical = page_table[ctx_slots, ctx_pos_i64 // 128].astype(np.int64) + 1
        swa_pool = getattr(self.runner, "_swa_pool", None)
        if swa_pool is not None:
            physical = swa_pool.lookup(physical).astype(np.int64)
        slots_i64 = physical * 128 + (ctx_pos_i64 % 128)

        np.copyto(self.ctx_positions_cpu[:n_ctx].numpy(), ctx_pos_i64)
        np.copyto(self.ctx_slot_mapping_cpu[:n_ctx].numpy(), slots_i64)
        self.ctx_positions_gpu[:n_ctx].copy_(
            self.ctx_positions_cpu[:n_ctx], non_blocking=True
        )
        self.ctx_slot_mapping_gpu[:n_ctx].copy_(
            self.ctx_slot_mapping_cpu[:n_ctx], non_blocking=True
        )
        if n_ctx == main_x.shape[0] and bool(np.all(ctx_rows == np.arange(n_ctx))):
            ctx_main_x = main_x
        else:
            np.copyto(self.ctx_rows_cpu[:n_ctx].numpy(), ctx_rows)
            self.ctx_rows_gpu[:n_ctx].copy_(
                self.ctx_rows_cpu[:n_ctx], non_blocking=True
            )
            ctx_main_x = main_x[self.ctx_rows_gpu[:n_ctx]]
        slot_view = self.ctx_slot_mapping_gpu[:n_ctx]
        self.model.precompute_and_store_context_kv(
            ctx_main_x,
            self.ctx_positions_gpu[:n_ctx],
            [slot_view] * self.num_draft_layers,
        )

    def _draft_batch(
        self,
        sequences: list[VerifiedSequence],
        *,
        positions: np.ndarray,
        page_table: np.ndarray,
    ) -> dict[int, list[int]]:
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import set_forward_context

        n = len(sequences)
        if n == 0:
            return {}
        runner = self.runner
        k = self.num_draft_tokens
        max_ctx = min(runner.max_model_len, page_table.shape[1] * 128)
        sequences = [
            seq
            for seq in sequences
            if int(positions[seq.rows[len(seq.accepted_tokens) - 1]]) + 1 < max_ctx
        ]
        n = len(sequences)
        if n == 0:
            return {}
        req_slots = np.array([seq.req_slot for seq in sequences], dtype=np.intp)
        anchors = np.array(
            [seq.accepted_tokens[-1] for seq in sequences], dtype=np.int32
        )
        pos_next = np.array(
            [
                int(positions[seq.rows[len(seq.accepted_tokens) - 1]]) + 1
                for seq in sequences
            ],
            dtype=np.int64,
        )
        seq_lens_np = np.minimum(pos_next + k, max_ctx).astype(np.int32)
        max_s_len = int(seq_lens_np.max())

        bucket = (
            select_decode_bucket(n, self.draft_buckets[-1])
            if self.draft_buckets and n <= self.draft_buckets[-1]
            else 0
        )
        use_graph = (
            bucket in self.draft_graphs and max_s_len <= runner.decode_graph_max_seq_len
        )
        eff_b = bucket if use_graph else n
        rows = eff_b * k
        actual_rows = n * k

        active_pt = page_table[req_slots, :] + 1
        swa_allocated = False
        if getattr(runner, "_swa_pool", None) is not None:
            swa_pt, swa_allocated = runner._swa_pool.map_active_windows(
                active_pt,
                pos_next,
                np.minimum(pos_next + k - 1, max_ctx - 1),
                pos_next,
                active_slots=req_slots,
            )
        else:
            swa_pt = active_pt
        bt_cached = (not swa_allocated) and (
            getattr(runner, "_last_verify_bt_sig", None)
            == (eff_b, tuple(int(s) for s in req_slots))
        )
        if not bt_cached:
            mla_bt_np = runner.mla_bt_cpu[:eff_b].numpy()
            mla_bt_np[:n] = active_pt
            if eff_b > n:
                mla_bt_np[n:eff_b] = 0

            swa_bt_np = runner.swa_bt_cpu[:eff_b].numpy()
            swa_bt_np[:n, 0::2] = swa_pt * 2
            swa_bt_np[:n, 1::2] = swa_pt * 2 + 1
            if eff_b > n:
                swa_bt_np[n:eff_b] = 0

            comp_bt_np = runner.comp_bt_cpu[:eff_b, 0].numpy()
            comp_bt_np[:n] = req_slots + 1
            if eff_b > n:
                comp_bt_np[n:eff_b] = runner.dummy_comp_block
            runner._last_verify_bt_sig = None

        sl_cpu_np = runner.seq_lens_cpu[:eff_b].numpy()
        sl_cpu_np[:n] = seq_lens_np
        if eff_b > n:
            sl_cpu_np[n:eff_b] = k

        tok_np = self.draft_tokens_cpu[:rows].numpy()
        tok_np[:rows] = self.noise_token_id
        tok_np[0:actual_rows:k] = anchors
        if eff_b > n:
            tok_np[actual_rows:rows:k] = 0

        q_pos_2d = pos_next[:, None] + np.arange(k, dtype=np.int64)[None, :]
        in_bounds = (q_pos_2d < max_ctx) & ((q_pos_2d // 128) < active_pt.shape[1])
        pos_cpu_np = self.draft_positions_cpu[:rows].numpy()
        pos_cpu_np[:actual_rows] = np.minimum(q_pos_2d, max_ctx - 1).reshape(-1)
        if eff_b > n:
            pos_cpu_np[actual_rows:rows] = (
                np.arange(rows - actual_rows, dtype=np.int64) % k
            )

        safe_cols = np.where(in_bounds, q_pos_2d // 128, 0)
        phys_2d = swa_pt[np.arange(n)[:, None], safe_cols].astype(np.int64)
        slots_2d = np.where(in_bounds, phys_2d * 128 + (q_pos_2d % 128), -1)
        sm_cpu_np = self.draft_slot_mapping_cpu[:rows].numpy()
        sm_cpu_np[:actual_rows] = slots_2d.reshape(-1)
        if eff_b > n:
            sm_cpu_np[actual_rows:rows] = -1

        if not bt_cached:
            runner.mla_bt_gpu[:eff_b].copy_(runner.mla_bt_cpu[:eff_b], non_blocking=True)
            runner.swa_bt_gpu[:eff_b].copy_(runner.swa_bt_cpu[:eff_b], non_blocking=True)
            runner.comp_bt_gpu[:eff_b].copy_(runner.comp_bt_cpu[:eff_b], non_blocking=True)
        runner.seq_lens_gpu[:eff_b].copy_(
            runner.seq_lens_cpu[:eff_b], non_blocking=True
        )
        self.draft_tokens_gpu[:rows].copy_(
            self.draft_tokens_cpu[:rows], non_blocking=True
        )
        self.draft_positions_gpu[:rows].copy_(
            self.draft_positions_cpu[:rows], non_blocking=True
        )
        self.draft_slot_mapping_gpu[:rows].copy_(
            self.draft_slot_mapping_cpu[:rows], non_blocking=True
        )

        attn_meta = runner._build_step_attn_metadata(
            num_reqs=eff_b,
            num_tokens=rows,
            max_query_len=k,
            max_seq_len=(
                runner.decode_graph_max_seq_len if use_graph else max_s_len
            ),
            qsl_gpu=self.draft_qsl_gpu[: eff_b + 1],
            qsl_cpu=self.draft_qsl_cpu[: eff_b + 1],
            seq_lens_gpu=runner.seq_lens_gpu[:eff_b],
            seq_lens_cpu=runner.seq_lens_cpu[:eff_b],
            positions_gpu=self.draft_positions_gpu[:rows],
            slot_mapping_gpu=self.draft_slot_mapping_gpu[:rows],
            is_prefilling_cpu=runner.decode_is_prefilling_cpu[:eff_b],
            for_capture=False,
            causal=False,
            only_layer_names=self.layer_names,
        )
        if use_graph:
            self.draft_graphs[eff_b].replay()
        else:
            with set_forward_context(
                attn_meta,
                runner.vllm_config,
                num_tokens=rows,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
            ):
                self._run_draft_forward_and_markov(eff_b)

        if runner.rank != 0:
            return {}
        self.draft_out_tokens_cpu[:, :n].copy_(
            self.draft_out_tokens_gpu[:, :n, 0], non_blocking=True
        )
        if runner.device.type == "cuda":
            torch.cuda.current_stream(runner.device).synchronize()
        # `draft_out_tokens_cpu[:, :n]` has shape `[k, n]`; transpose to `[n, k]`.
        draft_matrix = self.draft_out_tokens_cpu[:, :n].numpy().T.tolist()
        return {
            sequences[i].req_slot: [
                int(tok)
                for tok in draft_matrix[i][: min(k, max_ctx - int(pos_next[i]))]
            ]
            for i in range(n)
        }

    def draft(
        self,
        verified: list[VerifiedSequence],
        *,
        aux_hidden_states: list[torch.Tensor],
        positions: np.ndarray,
        req_slot_ids: np.ndarray,
        query_start_loc: np.ndarray,
        page_table: np.ndarray,
    ) -> dict[int, list[int]]:
        main_x = self.model.combine_hidden_states(
            torch.cat(aux_hidden_states, dim=-1)
        )
        by_seq = {sequence.batch_index: sequence for sequence in verified}
        ctx_row_parts: list[np.ndarray] = []
        ctx_slot_parts: list[np.ndarray] = []
        for seq_idx in range(len(query_start_loc) - 1):
            begin = int(query_start_loc[seq_idx])
            end = int(query_start_loc[seq_idx + 1])
            if end <= begin:
                continue
            sequence = by_seq.get(seq_idx)
            if sequence is None or len(sequence.rows) == 1:
                rows_arr = np.arange(begin, end, dtype=np.intp)
                req_slot = (
                    int(req_slot_ids[begin]) if sequence is None else sequence.req_slot
                )
            else:
                accepted_count = len(sequence.accepted_tokens)
                rows_arr = np.asarray(sequence.rows[:accepted_count], dtype=np.intp)
                req_slot = sequence.req_slot
            if rows_arr.size > 0:
                ctx_row_parts.append(rows_arr)
                ctx_slot_parts.append(np.full(rows_arr.size, req_slot, dtype=np.intp))

        if ctx_row_parts:
            self._store_contexts_batched(
                main_x=main_x,
                ctx_rows=np.concatenate(ctx_row_parts),
                ctx_slots=np.concatenate(ctx_slot_parts),
                positions=positions,
                page_table=page_table,
            )

        max_ctx = min(self.runner.max_model_len, page_table.shape[1] * 128)
        eligible = [
            seq
            for seq in verified
            if int(positions[seq.rows[len(seq.accepted_tokens) - 1]]) + 1 < max_ctx
        ]
        if not eligible:
            return {}

        # Guard against two sequences sharing the dummy overflow page if a draft window
        # crosses an unallocated 128-token boundary before the next step's page allocation.
        if len(eligible) > 1:
            dummy_page = self.runner.num_kv_pages
            safe_seqs: list[VerifiedSequence] = []
            ovf_seqs: list[VerifiedSequence] = []
            for seq in eligible:
                pos_next_seq = (
                    int(positions[seq.rows[len(seq.accepted_tokens) - 1]]) + 1
                )
                last_in_bounds_pos = (
                    min(pos_next_seq + self.num_draft_tokens, max_ctx) - 1
                )
                page_col = last_in_bounds_pos // 128
                if (
                    page_col >= page_table.shape[1]
                    or page_table[seq.req_slot, page_col] >= dummy_page
                ):
                    ovf_seqs.append(seq)
                else:
                    safe_seqs.append(seq)
            if len(ovf_seqs) > 1:
                # Batch all non-overflowing sequences together with the first overflowing
                # sequence (at most 1 sequence touches dummy_page per batch), and only
                # run the remaining overflowing sequences individually.
                merged = self._draft_batch(
                    safe_seqs + [ovf_seqs[0]],
                    positions=positions,
                    page_table=page_table,
                )
                for seq in ovf_seqs[1:]:
                    if self.runner.rank != 0 and self.runner.device.type == "cuda":
                        torch.cuda.current_stream(self.runner.device).synchronize()
                    merged.update(
                        self._draft_batch([seq], positions=positions, page_table=page_table)
                    )
                return merged

        return self._draft_batch(eligible, positions=positions, page_table=page_table)
