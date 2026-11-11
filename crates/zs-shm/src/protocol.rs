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

use std::sync::atomic::{Ordering, fence};

use zs_wire::{
    MAX_WORKERS, SHM_SUPERBLOCK_REGION_BYTES, SeqStepOutput, ShmGeometry, ShmSuperblock,
    StepHeader, StepPlanLayout, StepResultHeader, WORKER_STATES_OFFSET, WORKER_STATES_REGION_BYTES,
    WorkerSlotState, ZSTEP_PROTOCOL_VERSION, ZSTEP_SHM_MAGIC,
};

use crate::{Result, SharedMemoryArena, ShmError};

const PAGE_ALIGN: usize = 4096;

const fn align_up(value: usize, alignment: usize) -> usize {
    (value + alignment - 1) & !(alignment - 1)
}

/// Fully computed offsets for all fixed and double-buffered regions.
#[derive(Debug, Clone, Copy)]
pub struct ShmLayout {
    pub geometry: ShmGeometry,
    pub plan: StepPlanLayout,
    pub plan_slot_offsets: [usize; 2],
    pub result_slot_offsets: [usize; 2],
    pub result_slot_stride: usize,
    pub total_bytes: usize,
}

impl ShmLayout {
    pub fn compute(geometry: ShmGeometry) -> Result<Self> {
        if !(1..=MAX_WORKERS as u32).contains(&geometry.world_size) {
            return Err(ShmError::invalid_input("world_size must be in 1..=8"));
        }
        if geometry.max_seqs == 0
            || geometry.max_step_tokens == 0
            || geometry.max_decoder_rows == 0
            || geometry.max_pages_per_seq == 0
            || geometry.page_block_tokens == 0
            || geometry.spec_verify_width == 0
            || geometry.spec_verify_width > 8
        {
            return Err(ShmError::invalid_input("shared-memory geometry is invalid"));
        }
        if geometry.max_decoder_rows > geometry.max_step_tokens {
            return Err(ShmError::invalid_input(
                "max_decoder_rows cannot exceed max_step_tokens",
            ));
        }
        let plan = StepPlanLayout::compute(geometry);
        let plans_offset = SHM_SUPERBLOCK_REGION_BYTES + WORKER_STATES_REGION_BYTES;
        let second_plan = plans_offset
            .checked_add(plan.total_slot_bytes)
            .ok_or_else(|| ShmError::invalid_input("plan offset overflow"))?;
        let results_offset = second_plan
            .checked_add(plan.total_slot_bytes)
            .ok_or_else(|| ShmError::invalid_input("result offset overflow"))?;
        let result_payload = size_of::<StepResultHeader>()
            .checked_add(
                (geometry.max_seqs as usize)
                    .checked_mul(size_of::<SeqStepOutput>())
                    .ok_or_else(|| ShmError::invalid_input("result size overflow"))?,
            )
            .ok_or_else(|| ShmError::invalid_input("result size overflow"))?;
        let result_slot_stride = align_up(result_payload, PAGE_ALIGN);
        let second_result = results_offset
            .checked_add(result_slot_stride)
            .ok_or_else(|| ShmError::invalid_input("result offset overflow"))?;
        let total_bytes = second_result
            .checked_add(result_slot_stride)
            .ok_or_else(|| ShmError::invalid_input("total shared-memory size overflow"))?;
        Ok(Self {
            geometry,
            plan,
            plan_slot_offsets: [plans_offset, second_plan],
            result_slot_offsets: [results_offset, second_result],
            result_slot_stride,
            total_bytes,
        })
    }

    fn slot_index(slot: usize) -> Result<usize> {
        if slot < 2 {
            Ok(slot)
        } else {
            Err(ShmError::invalid_input("double-buffer slot must be 0 or 1"))
        }
    }
}

impl SharedMemoryArena {
    /// Zero the mapping and publish its immutable geometry superblock.
    pub fn initialize(&mut self, layout: &ShmLayout) -> Result<()> {
        if self.len() != layout.total_bytes {
            return Err(ShmError::invalid_input("mapping size differs from layout"));
        }
        self.bytes_mut().fill(0);
        let superblock = ShmSuperblock {
            magic: ZSTEP_SHM_MAGIC,
            protocol_version: ZSTEP_PROTOCOL_VERSION,
            world_size: layout.geometry.world_size,
            max_seqs: layout.geometry.max_seqs,
            max_step_tokens: layout.geometry.max_step_tokens,
            max_decoder_rows: layout.geometry.max_decoder_rows,
            max_pages_per_seq: layout.geometry.max_pages_per_seq,
            page_block_tokens: layout.geometry.page_block_tokens,
            spec_verify_width: layout.geometry.spec_verify_width,
            master_pid: std::process::id(),
            total_shm_bytes: layout.total_bytes as u64,
            plan_slot_offsets: [
                layout.plan_slot_offsets[0] as u64,
                layout.plan_slot_offsets[1] as u64,
            ],
            plan_slot_stride: layout.plan.total_slot_bytes as u64,
            result_slot_offsets: [
                layout.result_slot_offsets[0] as u64,
                layout.result_slot_offsets[1] as u64,
            ],
            _reserved: [layout.result_slot_stride as u64, 0, 0, 0],
        };
        *self.view_mut::<ShmSuperblock>(0)? = superblock;
        Ok(())
    }

    pub fn superblock(&self) -> Result<&ShmSuperblock> {
        self.view::<ShmSuperblock>(0)
    }

    pub fn worker_state(&self, rank: usize) -> Result<&WorkerSlotState> {
        if rank >= MAX_WORKERS {
            return Err(ShmError::invalid_input("worker rank is out of range"));
        }
        self.view::<WorkerSlotState>(WORKER_STATES_OFFSET + rank * size_of::<WorkerSlotState>())
    }

    pub fn plan_slot(&self, layout: &ShmLayout, slot: usize) -> Result<StepPlanSlot<'_>> {
        let index = ShmLayout::slot_index(slot)?;
        let base = layout.plan_slot_offsets[index];
        self.checked_range(base, layout.plan.total_slot_bytes, 64)?;
        Ok(StepPlanSlot {
            arena: self,
            base,
            geometry: layout.geometry,
            layout: layout.plan,
        })
    }

    pub fn plan_slot_mut(
        &mut self,
        layout: &ShmLayout,
        slot: usize,
    ) -> Result<StepPlanSlotMut<'_>> {
        let index = ShmLayout::slot_index(slot)?;
        let base = layout.plan_slot_offsets[index];
        self.checked_range(base, layout.plan.total_slot_bytes, 64)?;
        Ok(StepPlanSlotMut {
            arena: self,
            base,
            geometry: layout.geometry,
            layout: layout.plan,
        })
    }

    pub fn result_slot(&self, layout: &ShmLayout, slot: usize) -> Result<ResultSlot<'_>> {
        let index = ShmLayout::slot_index(slot)?;
        let base = layout.result_slot_offsets[index];
        self.checked_range(base, layout.result_slot_stride, 64)?;
        Ok(ResultSlot {
            arena: self,
            base,
            max_seqs: layout.geometry.max_seqs as usize,
        })
    }

    pub fn result_slot_mut(
        &mut self,
        layout: &ShmLayout,
        slot: usize,
    ) -> Result<ResultSlotMut<'_>> {
        let index = ShmLayout::slot_index(slot)?;
        let base = layout.result_slot_offsets[index];
        self.checked_range(base, layout.result_slot_stride, 64)?;
        Ok(ResultSlotMut {
            arena: self,
            base,
            max_seqs: layout.geometry.max_seqs as usize,
        })
    }

    fn view<T>(&self, offset: usize) -> Result<&T> {
        self.checked_range(offset, size_of::<T>(), std::mem::align_of::<T>())?;
        // SAFETY: the range and alignment were checked, and all instantiated types are POD wire
        // structs for which every bit pattern is valid.
        Ok(unsafe { &*self.pointer(offset).cast::<T>() })
    }

    fn view_mut<T>(&mut self, offset: usize) -> Result<&mut T> {
        self.checked_range(offset, size_of::<T>(), std::mem::align_of::<T>())?;
        // SAFETY: the range and alignment were checked and `&mut self` makes this view exclusive.
        Ok(unsafe { &mut *self.pointer(offset).cast::<T>() })
    }

    fn slice_view<T>(&self, offset: usize, len: usize) -> Result<&[T]> {
        let bytes = len
            .checked_mul(size_of::<T>())
            .ok_or_else(|| ShmError::invalid_input("slice byte length overflow"))?;
        self.checked_range(offset, bytes, std::mem::align_of::<T>())?;
        // SAFETY: the complete range is live and aligned; scalar wire elements accept all bits.
        Ok(unsafe { std::slice::from_raw_parts(self.pointer(offset).cast::<T>(), len) })
    }

    fn slice_view_mut<T>(&mut self, offset: usize, len: usize) -> Result<&mut [T]> {
        let bytes = len
            .checked_mul(size_of::<T>())
            .ok_or_else(|| ShmError::invalid_input("slice byte length overflow"))?;
        self.checked_range(offset, bytes, std::mem::align_of::<T>())?;
        // SAFETY: the complete range is live and aligned, and `&mut self` makes it exclusive.
        Ok(unsafe { std::slice::from_raw_parts_mut(self.pointer(offset).cast::<T>(), len) })
    }
}

#[derive(Debug)]
pub struct StepPlanSlot<'a> {
    arena: &'a SharedMemoryArena,
    base: usize,
    geometry: ShmGeometry,
    layout: StepPlanLayout,
}

impl StepPlanSlot<'_> {
    pub fn header(&self) -> Result<&StepHeader> {
        self.arena.view(self.base + self.layout.header_offset)
    }

    /// Acquire the published step and make every preceding payload write visible.
    pub fn published_step_id(&self) -> Result<u64> {
        let _geometry = self.geometry;
        let step_id = self.header()?.step_id.load(Ordering::Acquire);
        fence(Ordering::Acquire);
        Ok(step_id)
    }
}

#[derive(Debug)]
pub struct StepPlanSlotMut<'a> {
    arena: &'a mut SharedMemoryArena,
    base: usize,
    geometry: ShmGeometry,
    layout: StepPlanLayout,
}

macro_rules! plan_accessors {
    ($name:ident, $name_mut:ident, $offset:ident, $ty:ty, $count:expr) => {
        pub fn $name(&self) -> Result<&[$ty]> {
            self.arena
                .slice_view(self.base + self.layout.$offset, $count(self.geometry))
        }

        pub fn $name_mut(&mut self) -> Result<&mut [$ty]> {
            self.arena
                .slice_view_mut(self.base + self.layout.$offset, $count(self.geometry))
        }
    };
}

impl StepPlanSlotMut<'_> {
    pub fn header_mut(&mut self) -> Result<&mut StepHeader> {
        self.arena.view_mut(self.base + self.layout.header_offset)
    }

    plan_accessors!(
        token_ids,
        token_ids_mut,
        token_ids_offset,
        i32,
        |g: ShmGeometry| g.max_step_tokens as usize
    );
    plan_accessors!(
        positions,
        positions_mut,
        positions_offset,
        i32,
        |g: ShmGeometry| g.max_step_tokens as usize
    );
    plan_accessors!(
        req_slot_ids,
        req_slot_ids_mut,
        req_slot_ids_offset,
        i32,
        |g: ShmGeometry| g.max_step_tokens as usize
    );
    plan_accessors!(
        replay_mask,
        replay_mask_mut,
        replay_mask_offset,
        u8,
        |g: ShmGeometry| g.max_step_tokens as usize
    );
    plan_accessors!(
        engram_prev_window,
        engram_prev_window_mut,
        engram_prev_window_offset,
        i32,
        |g: ShmGeometry| g.max_step_tokens as usize * 3
    );
    plan_accessors!(
        narrow_keep_indices,
        narrow_keep_indices_mut,
        narrow_keep_indices_offset,
        i32,
        |g: ShmGeometry| g.max_decoder_rows as usize
    );
    plan_accessors!(
        sample_row_indices,
        sample_row_indices_mut,
        sample_row_indices_offset,
        i32,
        |g: ShmGeometry| g.max_decoder_rows as usize
    );
    plan_accessors!(
        seq_lens,
        seq_lens_mut,
        seq_lens_offset,
        i32,
        |g: ShmGeometry| g.max_seqs as usize
    );
    plan_accessors!(
        query_start_loc,
        query_start_loc_mut,
        query_start_loc_offset,
        i32,
        |g: ShmGeometry| g.max_seqs as usize + 1
    );
    plan_accessors!(
        global_kv_page_table,
        global_kv_page_table_mut,
        global_kv_page_table_offset,
        i32,
        |g: ShmGeometry| g.max_seqs as usize * g.max_pages_per_seq as usize
    );
    plan_accessors!(
        swa_page_table,
        swa_page_table_mut,
        swa_page_table_offset,
        i32,
        |g: ShmGeometry| g.max_seqs as usize * 4
    );
    plan_accessors!(
        compressor_slot_ids,
        compressor_slot_ids_mut,
        compressor_slot_ids_offset,
        i32,
        |g: ShmGeometry| g.max_seqs as usize
    );
    plan_accessors!(
        temperatures,
        temperatures_mut,
        temperatures_offset,
        f32,
        |g: ShmGeometry| g.max_seqs as usize
    );

    /// Release-publish the step ID after all header fields and payload arrays are complete.
    pub fn publish(self, step_id: u64) -> Result<()> {
        if step_id == 0 {
            return Err(ShmError::invalid_input("published step IDs start at one"));
        }
        fence(Ordering::Release);
        self.arena
            .view::<StepHeader>(self.base)?
            .step_id
            .store(step_id, Ordering::Release);
        Ok(())
    }
}

#[derive(Debug)]
pub struct ResultSlot<'a> {
    arena: &'a SharedMemoryArena,
    base: usize,
    max_seqs: usize,
}

impl ResultSlot<'_> {
    pub fn header(&self) -> Result<&StepResultHeader> {
        self.arena.view(self.base)
    }

    pub fn outputs(&self) -> Result<&[SeqStepOutput]> {
        self.arena
            .slice_view(self.base + size_of::<StepResultHeader>(), self.max_seqs)
    }

    /// Acquire the completed step before reading timing, status, or sequence outputs.
    pub fn completed_step_id(&self) -> Result<u64> {
        let step_id = self.header()?.completed_step_id.load(Ordering::Acquire);
        fence(Ordering::Acquire);
        Ok(step_id)
    }

    pub fn max_seqs(&self) -> usize {
        self.max_seqs
    }
}

#[derive(Debug)]
pub struct ResultSlotMut<'a> {
    arena: &'a mut SharedMemoryArena,
    base: usize,
    max_seqs: usize,
}

impl ResultSlotMut<'_> {
    pub fn header_mut(&mut self) -> Result<&mut StepResultHeader> {
        self.arena.view_mut(self.base)
    }

    pub fn outputs_mut(&mut self) -> Result<&mut [SeqStepOutput]> {
        self.arena
            .slice_view_mut(self.base + size_of::<StepResultHeader>(), self.max_seqs)
    }

    pub fn max_seqs(&self) -> usize {
        self.max_seqs
    }

    pub fn publish(self, step_id: u64) -> Result<()> {
        fence(Ordering::Release);
        self.arena
            .view::<StepResultHeader>(self.base)?
            .completed_step_id
            .store(step_id, Ordering::Release);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry() -> ShmGeometry {
        ShmGeometry {
            world_size: 2,
            max_seqs: 4,
            max_step_tokens: 16,
            max_decoder_rows: 8,
            max_pages_per_seq: 8,
            page_block_tokens: 64,
            spec_verify_width: 6,
        }
    }

    #[test]
    fn layout_regions_are_page_aligned_and_bounded() {
        let layout = ShmLayout::compute(geometry()).expect("compute layout");
        assert_eq!(layout.plan_slot_offsets[0], 8192);
        assert_eq!(layout.plan_slot_offsets[1] % PAGE_ALIGN, 0);
        assert_eq!(layout.result_slot_offsets[0] % PAGE_ALIGN, 0);
        assert_eq!(layout.total_bytes % PAGE_ALIGN, 0);
        assert_eq!(
            layout.result_slot_offsets[1],
            layout.result_slot_offsets[0] + layout.result_slot_stride
        );
    }

    #[test]
    fn invalid_geometry_is_rejected() {
        let mut invalid = geometry();
        invalid.world_size = 9;
        assert!(ShmLayout::compute(invalid).is_err());
        invalid = geometry();
        invalid.max_decoder_rows = invalid.max_step_tokens + 1;
        assert!(ShmLayout::compute(invalid).is_err());
    }

    #[test]
    fn atomic_types_have_wire_compatible_representation() {
        assert_eq!(size_of::<std::sync::atomic::AtomicU64>(), size_of::<u64>());
        assert_eq!(
            std::mem::align_of::<std::sync::atomic::AtomicU64>(),
            std::mem::align_of::<u64>()
        );
    }
}
