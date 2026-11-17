// Copyright (c) 2026 zstep Contributors
// SPDX-License-Identifier: MIT

//! Zero-profiling online NNLS step-time predictor and quadratic SLO chunk solver (`EvolvingInfra` §5.2).

#[derive(Debug, Clone, Copy)]
pub struct StepObservation {
    pub prefill_tokens: usize,
    pub decode_tokens: usize,
    pub cached_tokens: usize,
    pub kv_pages: usize,
    pub sample_rows: usize,
    pub step_ms: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct StepTimePredictor {
    /// `[base, prefill_tok, decode_tok, attn_pair, kv_page, expert_frac, sample_row]` in ms.
    pub theta: [f64; 7],
    pub experts_per_token: usize,
    pub num_routed_experts: usize,
    pub holdout_mape: f64,
}

pub fn expert_fraction(
    batch_tokens: usize,
    experts_per_token: usize,
    num_routed_experts: usize,
) -> f64 {
    if batch_tokens == 0 || experts_per_token == 0 || num_routed_experts == 0 {
        return 0.0;
    }
    if experts_per_token >= num_routed_experts {
        return 1.0;
    }
    let miss_prob = 1.0 - (experts_per_token as f64 / num_routed_experts as f64);
    1.0 - miss_prob.powi((batch_tokens.min(65_536)) as i32)
}

pub fn attention_pairs(uncached_tokens: usize, cached_tokens: usize) -> f64 {
    let u = uncached_tokens as f64;
    let c = cached_tokens as f64;
    u * c + 0.5 * u * (u + 1.0)
}

impl StepTimePredictor {
    pub fn features(&self, obs: &StepObservation) -> [f64; 7] {
        let total = obs.prefill_tokens.saturating_add(obs.decode_tokens);
        [
            1.0,
            obs.prefill_tokens as f64,
            obs.decode_tokens as f64,
            attention_pairs(obs.prefill_tokens, obs.cached_tokens) * 1e-6,
            obs.kv_pages as f64,
            expert_fraction(total, self.experts_per_token, self.num_routed_experts),
            obs.sample_rows as f64,
        ]
    }

    pub fn predict_ms(&self, obs: &StepObservation) -> f64 {
        let x = self.features(obs);
        x.iter().zip(&self.theta).map(|(a, b)| a * b).sum()
    }

    /// Fit non-negative coefficients via projected Gauss-Seidel NNLS on chronological 70/30 split.
    pub fn fit(
        observations: &[StepObservation],
        experts_per_token: usize,
        num_routed_experts: usize,
    ) -> Option<Self> {
        if observations.len() < 10 {
            return None;
        }
        let split = ((observations.len() * 7) / 10).clamp(7, observations.len() - 1);
        let mut model = Self {
            theta: [0.0; 7],
            experts_per_token,
            num_routed_experts,
            holdout_mape: 0.0,
        };
        let mut gram = [[0.0f64; 7]; 7];
        let mut rhs = [0.0f64; 7];
        for obs in &observations[..split] {
            if !obs.step_ms.is_finite() || obs.step_ms <= 0.0 {
                return None;
            }
            let x = model.features(obs);
            for i in 0..7 {
                rhs[i] += x[i] * obs.step_ms;
                for j in 0..7 {
                    gram[i][j] += x[i] * x[j];
                }
            }
        }
        for (i, row) in gram.iter_mut().enumerate() {
            row[i] += 1e-9;
        }
        for _ in 0..400 {
            let mut max_delta = 0.0f64;
            for i in 0..7 {
                let diag = gram[i][i];
                if diag <= 0.0 {
                    continue;
                }
                let mut dot = 0.0;
                for (j, (&g_ij, &th_j)) in gram[i].iter().zip(&model.theta).enumerate() {
                    if j != i {
                        dot += g_ij * th_j;
                    }
                }
                let next = ((rhs[i] - dot) / diag).max(0.0);
                max_delta = max_delta.max((next - model.theta[i]).abs());
                model.theta[i] = next;
            }
            if max_delta < 1e-10 {
                break;
            }
        }
        let holdout = &observations[split..];
        let mut ape_sum = 0.0;
        for obs in holdout {
            if obs.step_ms <= 0.0 {
                return None;
            }
            ape_sum += (model.predict_ms(obs) - obs.step_ms).abs() / obs.step_ms;
        }
        model.holdout_mape = ape_sum / (holdout.len() as f64);
        Some(model)
    }

    /// Solve quadratic causal-attention + linear prefill budget under `target_step_ms` SLO.
    pub fn recommend_prefill_chunk(
        &self,
        target_step_ms: f64,
        decode_tokens: usize,
        cached_tokens: usize,
        kv_pages: usize,
        min_chunk: usize,
        max_chunk: usize,
    ) -> usize {
        if max_chunk <= min_chunk {
            return max_chunk;
        }
        let decode_only = self.predict_ms(&StepObservation {
            prefill_tokens: 0,
            decode_tokens,
            cached_tokens: 0,
            kv_pages,
            sample_rows: decode_tokens,
            step_ms: 0.0,
        });
        let slack = target_step_ms - decode_only - self.theta[5] * 0.25;
        if slack <= 0.0 {
            return min_chunk;
        }
        let c_attn = self.theta[3] * 1e-6;
        let a = 0.5 * c_attn;
        let b = self.theta[1] + c_attn * (0.5 + cached_tokens as f64);
        let raw = if a > 1e-15 {
            let disc = (b * b + 4.0 * a * slack).max(0.0);
            ((-b + disc.sqrt()) / (2.0 * a)).floor() as usize
        } else if b > 1e-12 {
            (slack / b).floor() as usize
        } else {
            max_chunk
        };
        raw.clamp(min_chunk, max_chunk)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nnls_fits_synthetic_moe_steps_and_shrinks_chunk_under_long_cached_context() {
        let mut obs = Vec::new();
        for idx in 0..24usize {
            let pf = (idx % 6) * 256;
            let dec = 8 + (idx % 5) * 8;
            let cached = (idx % 4) * 1024;
            let pages = (pf + cached + dec) / 64 + 1;
            let pairs_m = attention_pairs(pf, cached) * 1e-6;
            let ef = expert_fraction(pf + dec, 8, 256);
            let ms = 1.2
                + 0.004 * pf as f64
                + 0.02 * dec as f64
                + 0.35 * pairs_m
                + 0.01 * pages as f64
                + 2.5 * ef
                + 0.005 * dec as f64;
            obs.push(StepObservation {
                prefill_tokens: pf,
                decode_tokens: dec,
                cached_tokens: cached,
                kv_pages: pages,
                sample_rows: dec,
                step_ms: ms,
            });
        }
        let model = StepTimePredictor::fit(&obs, 8, 256).expect("fit succeeds");
        assert!(model.theta.iter().all(|&c| c >= 0.0));
        assert!(model.holdout_mape < 0.05, "mape={}", model.holdout_mape);
        let short_ctx = model.recommend_prefill_chunk(18.0, 16, 0, 32, 64, 2048);
        let long_ctx = model.recommend_prefill_chunk(18.0, 16, 16_384, 288, 64, 2048);
        assert!(long_ctx < short_ctx, "{long_ctx} !< {short_ctx}");
    }
}
