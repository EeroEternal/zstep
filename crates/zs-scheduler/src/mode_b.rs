// Copyright (c) 2026 zstep Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

//! Mode B subprocess orchestration and `in_flight_depth = 1` overlap loop.

use std::fmt;
use std::fs;
use std::io;
use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use zs_shm::{EventFd, SharedMemoryArena, ShmError, ShmLayout};
use zs_wire::{
    StepHeader, StepOpcode, WORKER_STATUS_ERROR, WORKER_STATUS_READY, WORKER_STATUS_RUNNING,
};

const INHERITED_PLAN_FD_BASE: i32 = 100;
const INHERITED_DONE_FD_BASE: i32 = 200;

/// Explicit Python worker and micro-step launch parameters.
#[derive(Debug, Clone)]
pub struct WorkerLaunchConfig {
    pub python_executable: PathBuf,
    pub worker_root: PathBuf,
    pub hidden_size: u32,
    pub output_features: u32,
    pub router_experts: u32,
    pub router_top_k: u32,
    pub allreduce_blocks: u32,
    pub allreduce_threads: u32,
}

/// Complete Mode B run configuration; every behavioral parameter is explicit.
#[derive(Debug, Clone)]
pub struct ModeBConfig {
    pub shm_name: String,
    pub geometry: zs_wire::ShmGeometry,
    pub steps: u64,
    pub token_counts: Vec<u32>,
    pub launch: WorkerLaunchConfig,
    pub ready_timeout: Duration,
    pub step_timeout: Duration,
}

#[derive(Debug)]
pub enum ModeBError {
    Transport(ShmError),
    Io(io::Error),
    Protocol(String),
}

impl fmt::Display for ModeBError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Transport(error) => write!(formatter, "shared-memory transport: {error}"),
            Self::Io(error) => write!(formatter, "worker I/O: {error}"),
            Self::Protocol(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for ModeBError {}

impl From<ShmError> for ModeBError {
    fn from(error: ShmError) -> Self {
        Self::Transport(error)
    }
}

impl From<io::Error> for ModeBError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

pub type Result<T> = std::result::Result<T, ModeBError>;

#[derive(Debug, Default)]
struct WorkerProcesses(Vec<Child>);

impl WorkerProcesses {
    fn check_alive(&mut self) -> Result<()> {
        for (rank, child) in self.0.iter_mut().enumerate() {
            if let Some(status) = child.try_wait()? {
                return Err(ModeBError::Protocol(format!(
                    "GPU worker rank {rank} exited prematurely with status {status}"
                )));
            }
        }
        Ok(())
    }

    fn wait(&mut self) -> Result<()> {
        for child in &mut self.0 {
            let status = child.wait()?;
            if !status.success() {
                return Err(ModeBError::Protocol(format!(
                    "GPU worker exited with status {status}"
                )));
            }
        }
        Ok(())
    }
}

impl Drop for WorkerProcesses {
    fn drop(&mut self) {
        for child in &mut self.0 {
            let _ = child.kill();
        }
        for child in &mut self.0 {
            let _ = child.wait();
        }
    }
}

/// Aggregate timing measurements. All latency fields are integer microseconds.
#[derive(Debug, Clone)]
pub struct ModeBMetrics {
    pub world_size: u32,
    pub steps: u64,
    pub init_wall_us: u64,
    pub total_wall_us: u64,
    pub gpu_exec_us: u64,
    pub prepare_next_us: u64,
    pub hidden_prepare_us: u64,
    pub control_plane_ipc_us: u64,
    pub publish_signal_us: u64,
    pub accepted_tokens: u64,
    pub round_trip_samples_us: Vec<u64>,
    pub control_plane_samples_us: Vec<u64>,
}

impl ModeBMetrics {
    pub fn overlap_hiding_efficiency(&self) -> f64 {
        ratio(self.hidden_prepare_us, self.prepare_next_us)
    }

    pub fn serial_equivalent_us(&self) -> u64 {
        self.total_wall_us + self.hidden_prepare_us
    }

    pub fn overlap_speedup(&self) -> f64 {
        ratio(self.serial_equivalent_us(), self.total_wall_us)
    }

    pub fn to_json_pretty(&self) -> String {
        format!(
            concat!(
                "{{\n",
                "  \"mode\": \"B\",\n",
                "  \"world_size\": {},\n",
                "  \"steps\": {},\n",
                "  \"init_wall_us\": {},\n",
                "  \"total_wall_us\": {},\n",
                "  \"mean_step_us\": {:.3},\n",
                "  \"mean_gpu_exec_us\": {:.3},\n",
                "  \"mean_control_plane_ipc_us\": {:.3},\n",
                "  \"control_plane_p50_us\": {},\n",
                "  \"control_plane_p95_us\": {},\n",
                "  \"control_plane_p99_us\": {},\n",
                "  \"round_trip_p50_us\": {},\n",
                "  \"round_trip_p95_us\": {},\n",
                "  \"round_trip_p99_us\": {},\n",
                "  \"mean_publish_signal_us\": {:.3},\n",
                "  \"prepare_next_us\": {},\n",
                "  \"hidden_prepare_us\": {},\n",
                "  \"overlap_hiding_efficiency\": {:.6},\n",
                "  \"overlap_speedup\": {:.6},\n",
                "  \"accepted_tokens\": {}\n",
                "}}\n"
            ),
            self.world_size,
            self.steps,
            self.init_wall_us,
            self.total_wall_us,
            ratio(self.total_wall_us, self.steps),
            ratio(self.gpu_exec_us, self.steps),
            ratio(self.control_plane_ipc_us, self.steps),
            percentile(&self.control_plane_samples_us, 50),
            percentile(&self.control_plane_samples_us, 95),
            percentile(&self.control_plane_samples_us, 99),
            percentile(&self.round_trip_samples_us, 50),
            percentile(&self.round_trip_samples_us, 95),
            percentile(&self.round_trip_samples_us, 99),
            ratio(self.publish_signal_us, self.steps),
            self.prepare_next_us,
            self.hidden_prepare_us,
            self.overlap_hiding_efficiency(),
            self.overlap_speedup(),
            self.accepted_tokens,
        )
    }
}

fn ratio(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn percentile(samples: &[u64], percentile: usize) -> u64 {
    if samples.is_empty() {
        return 0;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let index = (sorted.len() - 1) * percentile / 100;
    sorted[index]
}

/// Owns every worker, descriptor, mapping, and rendezvous file for one run.
#[derive(Debug)]
pub struct ModeBCoordinator {
    workers: WorkerProcesses,
    plan_events: Vec<EventFd>,
    done_events: Vec<EventFd>,
    arena: SharedMemoryArena,
    layout: ShmLayout,
    config: ModeBConfig,
    rendezvous_file: PathBuf,
}

impl ModeBCoordinator {
    /// Create the ABI region, spawn ranks `0..world_size`, and inherit explicit eventfds.
    pub fn launch(config: ModeBConfig) -> Result<Self> {
        validate_config(&config)?;
        let layout = ShmLayout::compute(config.geometry)?;
        let mut arena = SharedMemoryArena::create(&config.shm_name, layout.total_bytes)?;
        arena.initialize(&layout)?;
        let plan_events = (0..config.geometry.world_size)
            .map(|_| EventFd::new())
            .collect::<zs_shm::Result<Vec<_>>>()?;
        let done_events = (0..config.geometry.world_size)
            .map(|_| EventFd::new())
            .collect::<zs_shm::Result<Vec<_>>>()?;
        let rendezvous_file = std::env::temp_dir().join(format!("{}_gloo", config.shm_name));
        let workers = spawn_workers(&config, &plan_events, &done_events, &rendezvous_file)?;
        Ok(Self {
            workers,
            plan_events,
            done_events,
            arena,
            layout,
            config,
            rendezvous_file,
        })
    }

    /// Wait for GPU initialization, execute all steps, and gracefully stop every worker.
    pub fn run(mut self) -> Result<ModeBMetrics> {
        let init_start = Instant::now();
        self.wait_all(0, self.config.ready_timeout, true)?;
        let init_wall_us = duration_us(init_start.elapsed());
        let wall_start = Instant::now();
        self.prepare_step(1, StepOpcode::NormalStep)?;
        let mut metrics = ModeBMetrics {
            world_size: self.config.geometry.world_size,
            steps: self.config.steps,
            init_wall_us,
            total_wall_us: 0,
            gpu_exec_us: 0,
            prepare_next_us: 0,
            hidden_prepare_us: 0,
            control_plane_ipc_us: 0,
            publish_signal_us: 0,
            accepted_tokens: 0,
            round_trip_samples_us: Vec::with_capacity(self.config.steps as usize),
            control_plane_samples_us: Vec::with_capacity(self.config.steps as usize),
        };

        for step_id in 1..=self.config.steps {
            let round_trip_start = Instant::now();
            let signal_start = Instant::now();
            self.publish_prepared(step_id)?;
            let signal_us = duration_us(signal_start.elapsed());
            metrics.publish_signal_us += signal_us;

            let mut prepare_us = 0;
            if step_id < self.config.steps {
                let prepare_start = Instant::now();
                self.prepare_step(step_id + 1, StepOpcode::NormalStep)?;
                prepare_us = duration_us(prepare_start.elapsed());
                metrics.prepare_next_us += prepare_us;
            }

            self.wait_all(step_id, self.config.step_timeout, false)?;
            let result = self.arena.result_slot(&self.layout, slot(step_id))?;
            if result.completed_step_id()? != step_id {
                return Err(ModeBError::Protocol(format!(
                    "rank 0 returned the wrong completed step for {step_id}"
                )));
            }
            let header = result.header()?;
            if header.status_code != 0 {
                return Err(ModeBError::Protocol(format!(
                    "rank 0 failed step {step_id} with status {}",
                    header.status_code
                )));
            }
            let round_trip_us = duration_us(round_trip_start.elapsed());
            let gpu_us = u64::from(header.gpu_exec_us);
            let control_us = round_trip_us.saturating_sub(gpu_us.max(prepare_us));
            metrics.gpu_exec_us += gpu_us;
            metrics.control_plane_ipc_us += control_us;
            metrics.hidden_prepare_us += prepare_us.min(gpu_us);
            metrics.round_trip_samples_us.push(round_trip_us);
            metrics.control_plane_samples_us.push(control_us);
            for output in &result.outputs()?[..header.num_seq_results as usize] {
                metrics.accepted_tokens += u64::from(output.num_accepted);
            }
        }

        self.shutdown(self.config.steps + 1)?;
        self.workers.wait()?;
        metrics.total_wall_us = duration_us(wall_start.elapsed());
        Ok(metrics)
    }

    fn prepare_step(&mut self, step_id: u64, opcode: StepOpcode) -> Result<()> {
        let token_count = if opcode == StepOpcode::Shutdown {
            0
        } else {
            self.config.token_counts[(step_id - 1) as usize % self.config.token_counts.len()]
        };
        if token_count > self.config.geometry.max_step_tokens
            || token_count > self.config.geometry.max_decoder_rows
        {
            return Err(ModeBError::Protocol(format!(
                "token count {token_count} exceeds configured geometry"
            )));
        }
        let num_seqs = token_count.min(self.config.geometry.max_seqs);
        let mut plan = self.arena.plan_slot_mut(&self.layout, slot(step_id))?;
        *plan.header_mut()? = StepHeader {
            step_id: AtomicU64::new(0),
            opcode: opcode as u32,
            num_seqs,
            num_encoder_tokens: token_count,
            encoder_graph_bucket: 0,
            num_decoder_rows: token_count,
            decoder_graph_bucket: 0,
            num_sample_rows: num_seqs,
            is_dspark_verify: 0,
            flags: 0,
            _reserved: [0; 3],
        };
        for row in 0..token_count as usize {
            plan.token_ids_mut()?[row] = ((step_id as i32 * 131 + row as i32 * 17) % 32_000).abs();
            plan.positions_mut()?[row] = step_id as i32 - 1;
            plan.req_slot_ids_mut()?[row] = (row as u32 % num_seqs.max(1)) as i32;
        }
        for index in 0..num_seqs as usize {
            plan.sample_row_indices_mut()?[index] = index as i32;
            plan.seq_lens_mut()?[index] = step_id as i32;
            plan.query_start_loc_mut()?[index] = index as i32;
        }
        plan.query_start_loc_mut()?[num_seqs as usize] = num_seqs as i32;
        Ok(())
    }

    fn publish_prepared(&mut self, step_id: u64) -> Result<()> {
        self.arena
            .plan_slot_mut(&self.layout, slot(step_id))?
            .publish(step_id)?;
        for event in &self.plan_events {
            event.write(1)?;
        }
        Ok(())
    }

    fn wait_all(&mut self, step_id: u64, timeout: Duration, initializing: bool) -> Result<()> {
        let deadline = Instant::now() + timeout;
        for rank in 0..self.config.geometry.world_size as usize {
            loop {
                let now = Instant::now();
                if now >= deadline {
                    return Err(ModeBError::Protocol(format!(
                        "rank {rank} timed out waiting for phase {step_id}"
                    )));
                }
                let slice_end = (now + Duration::from_millis(200)).min(deadline);
                match self.done_events[rank].wait_until(slice_end) {
                    Ok(()) => break,
                    Err(_) => {
                        self.workers.check_alive()?;
                    }
                }
            }
            self.done_events[rank].read()?;
            let state = self.arena.worker_state(rank)?;
            let status = state.status_code.load(Ordering::Acquire);
            let error_code = state.error_code.load(Ordering::Acquire);
            if status == WORKER_STATUS_ERROR || error_code != 0 {
                return Err(ModeBError::Protocol(format!(
                    "rank {rank} reported worker error {error_code} in phase {step_id}"
                )));
            }
            if status != WORKER_STATUS_READY {
                return Err(ModeBError::Protocol(format!(
                    "rank {rank} has unexpected status {status} in phase {step_id}"
                )));
            }
            if !initializing && state.ready_step_id.load(Ordering::Acquire) != step_id {
                return Err(ModeBError::Protocol(format!(
                    "rank {rank} completed the wrong step in phase {step_id}"
                )));
            }
            if !initializing && status == WORKER_STATUS_RUNNING {
                return Err(ModeBError::Protocol("worker remained running".to_owned()));
            }
        }
        Ok(())
    }

    fn shutdown(&mut self, step_id: u64) -> Result<()> {
        self.prepare_step(step_id, StepOpcode::Shutdown)?;
        self.publish_prepared(step_id)?;
        self.wait_all(step_id, self.config.step_timeout, false)
    }
}

impl Drop for ModeBCoordinator {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.rendezvous_file);
    }
}

fn spawn_workers(
    config: &ModeBConfig,
    plan_events: &[EventFd],
    done_events: &[EventFd],
    rendezvous_file: &PathBuf,
) -> Result<WorkerProcesses> {
    let mut workers = WorkerProcesses::default();
    let allreduce_name = format!("{}_ar", config.shm_name);
    for rank in 0..config.geometry.world_size as usize {
        let inherited_plan_fd = INHERITED_PLAN_FD_BASE + rank as i32;
        let inherited_done_fd = INHERITED_DONE_FD_BASE + rank as i32;
        let source_plan_fd = plan_events[rank].as_raw_fd();
        let source_done_fd = done_events[rank].as_raw_fd();
        let mut command = Command::new(&config.launch.python_executable);
        command
            .arg("-u")
            .arg("-m")
            .arg("zstep_worker.runtime_loop")
            .arg("--shm-name")
            .arg(&config.shm_name)
            .arg("--rank")
            .arg(rank.to_string())
            .arg("--plan-fd")
            .arg(inherited_plan_fd.to_string())
            .arg("--done-fd")
            .arg(inherited_done_fd.to_string())
            .arg("--allreduce-name")
            .arg(&allreduce_name)
            .arg("--dist-init-file")
            .arg(rendezvous_file)
            .arg("--hidden-size")
            .arg(config.launch.hidden_size.to_string())
            .arg("--output-features")
            .arg(config.launch.output_features.to_string())
            .arg("--router-experts")
            .arg(config.launch.router_experts.to_string())
            .arg("--router-top-k")
            .arg(config.launch.router_top_k.to_string())
            .arg("--allreduce-blocks")
            .arg(config.launch.allreduce_blocks.to_string())
            .arg("--allreduce-threads")
            .arg(config.launch.allreduce_threads.to_string())
            .env("PYTHONPATH", &config.launch.worker_root)
            .env("PYTHONUNBUFFERED", "1")
            .env("NCCL_P2P_DISABLE", "1")
            .env("NCCL_IB_DISABLE", "1")
            .stdin(Stdio::null())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit());
        // SAFETY: the closure only calls async-signal-safe `dup2`; the targets are disjoint from
        // normal descriptors and dup2 clears CLOEXEC on exactly the two inherited descriptors.
        unsafe {
            command.pre_exec(move || {
                if libc::dup2(source_plan_fd, inherited_plan_fd) < 0 {
                    return Err(io::Error::last_os_error());
                }
                if libc::dup2(source_done_fd, inherited_done_fd) < 0 {
                    return Err(io::Error::last_os_error());
                }
                Ok(())
            });
        }
        let child = command.spawn().map_err(|error| {
            ModeBError::Protocol(format!("failed to spawn GPU worker rank {rank}: {error}"))
        })?;
        workers.0.push(child);
    }
    Ok(workers)
}

fn validate_config(config: &ModeBConfig) -> Result<()> {
    if config.steps == 0 {
        return Err(ModeBError::Protocol("steps must be positive".to_owned()));
    }
    if config.token_counts.is_empty() || config.token_counts.contains(&0) {
        return Err(ModeBError::Protocol(
            "token_counts must contain positive runtime values".to_owned(),
        ));
    }
    if config.launch.hidden_size == 0
        || config.launch.hidden_size % 32 != 0
        || config.launch.output_features < config.launch.router_experts
        || config.launch.router_experts == 0
        || config.launch.router_top_k == 0
        || config.launch.router_top_k > config.launch.router_experts
    {
        return Err(ModeBError::Protocol(
            "invalid micro-step dimensions".to_owned(),
        ));
    }
    if config.ready_timeout.is_zero() || config.step_timeout.is_zero() {
        return Err(ModeBError::Protocol("timeouts must be positive".to_owned()));
    }
    Ok(())
}

fn slot(step_id: u64) -> usize {
    (step_id & 1) as usize
}

fn duration_us(duration: Duration) -> u64 {
    duration.as_micros().min(u128::from(u64::MAX)) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentiles_are_deterministic() {
        assert_eq!(percentile(&[10, 20, 30, 40], 50), 20);
        assert_eq!(percentile(&[10, 20, 30, 40], 95), 30);
        assert_eq!(percentile(&[], 99), 0);
    }

    #[test]
    fn slot_alternates_for_depth_one_overlap() {
        assert_eq!(slot(1), 1);
        assert_eq!(slot(2), 0);
        assert_eq!(slot(3), 1);
    }
}
