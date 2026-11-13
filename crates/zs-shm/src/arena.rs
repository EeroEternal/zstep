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

use std::ffi::CString;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::ptr::NonNull;

use crate::{Result, ShmError};

#[derive(Debug)]
struct Mapping {
    ptr: NonNull<libc::c_void>,
    len: usize,
}

impl Drop for Mapping {
    fn drop(&mut self) {
        // SAFETY: this object uniquely owns a successful mmap region of exactly `len` bytes.
        let _ = unsafe { libc::munmap(self.ptr.as_ptr(), self.len) };
    }
}

#[derive(Debug)]
struct ShmObject {
    _fd: OwnedFd,
    posix_name: CString,
    unlink_on_drop: bool,
}

impl Drop for ShmObject {
    fn drop(&mut self) {
        if self.unlink_on_drop {
            // SAFETY: the name is a live NUL-terminated string owned by this object.
            let _ = unsafe { libc::shm_unlink(self.posix_name.as_ptr()) };
        }
    }
}

/// An owned POSIX shared-memory object and its `mmap` mapping.
///
/// The creator unlinks the object on drop. Both the descriptor and mapping are
/// released on every construction error and during unwind without panicking.
#[derive(Debug)]
pub struct SharedMemoryArena {
    mapping: Mapping,
    _object: ShmObject,
    name: String,
    writable: bool,
}

// The mapping is owned and all mutable slices require `&mut self`.
unsafe impl Send for SharedMemoryArena {}

impl SharedMemoryArena {
    /// Create and exclusively own `/dev/shm/<name>`, with no leading slash in `name`.
    pub fn create(name: &str, len: usize) -> Result<Self> {
        let (plain_name, posix_name) = validate_name(name)?;
        if len == 0 {
            return Err(ShmError::invalid_input(
                "shared-memory length must be nonzero",
            ));
        }
        // SAFETY: the name is NUL-terminated and contains no interior NUL bytes.
        let raw = unsafe {
            libc::shm_open(
                posix_name.as_ptr(),
                libc::O_CREAT | libc::O_EXCL | libc::O_RDWR,
                0o600,
            )
        };
        if raw < 0 {
            return Err(ShmError::last_os("shm_open create"));
        }
        // SAFETY: `shm_open` returned a newly owned descriptor.
        let fd = unsafe { OwnedFd::from_raw_fd(raw) };
        let object = ShmObject {
            _fd: fd,
            posix_name,
            unlink_on_drop: true,
        };
        // SAFETY: the descriptor refers to a new shared-memory object.
        if unsafe { libc::ftruncate(object._fd.as_raw_fd(), len as libc::off_t) } < 0 {
            return Err(ShmError::last_os("ftruncate"));
        }
        let mapping = map(object._fd.as_raw_fd(), len, true)?;
        Ok(Self {
            mapping,
            _object: object,
            name: plain_name,
            writable: true,
        })
    }

    /// Attach to an existing object. `len = 0` uses the exact `fstat` size.
    pub fn open(name: &str, len: usize, writable: bool) -> Result<Self> {
        let (plain_name, posix_name) = validate_name(name)?;
        let flags = if writable {
            libc::O_RDWR
        } else {
            libc::O_RDONLY
        };
        // SAFETY: the name is a valid NUL-terminated POSIX shared-memory name.
        let raw = unsafe { libc::shm_open(posix_name.as_ptr(), flags, 0) };
        if raw < 0 {
            return Err(ShmError::last_os("shm_open open"));
        }
        // SAFETY: `shm_open` returned a newly owned descriptor.
        let fd = unsafe { OwnedFd::from_raw_fd(raw) };
        let object = ShmObject {
            _fd: fd,
            posix_name,
            unlink_on_drop: false,
        };
        let actual_len = if len == 0 {
            let mut stat = std::mem::MaybeUninit::<libc::stat>::uninit();
            // SAFETY: `stat` points to writable storage and the descriptor is valid.
            if unsafe { libc::fstat(object._fd.as_raw_fd(), stat.as_mut_ptr()) } < 0 {
                return Err(ShmError::last_os("fstat"));
            }
            // SAFETY: successful `fstat` initialized the structure.
            let stat = unsafe { stat.assume_init() };
            usize::try_from(stat.st_size)
                .map_err(|_| ShmError::invalid_input("shared-memory size does not fit usize"))?
        } else {
            len
        };
        if actual_len == 0 {
            return Err(ShmError::invalid_input("shared-memory object is empty"));
        }
        let mapping = map(object._fd.as_raw_fd(), actual_len, writable)?;
        Ok(Self {
            mapping,
            _object: object,
            name: plain_name,
            writable,
        })
    }

    /// Unlink the POSIX shared-memory entry from `/dev/shm` immediately while keeping
    /// the open descriptor and `mmap` region alive until every attached process exits.
    pub fn unlink_now(&mut self) -> Result<()> {
        if !self._object.unlink_on_drop {
            return Ok(());
        }
        // SAFETY: `posix_name` is a valid NUL-terminated string owned by `self._object`.
        let rc = unsafe { libc::shm_unlink(self._object.posix_name.as_ptr()) };
        self._object.unlink_on_drop = false;
        if rc < 0 {
            let err = std::io::Error::last_os_error();
            if err.raw_os_error() != Some(libc::ENOENT) {
                return Err(ShmError::new("shm_unlink", err));
            }
        }
        Ok(())
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// Raw descriptor of the underlying POSIX shared-memory object (for explicit child FD inheritance).
    pub fn as_raw_fd(&self) -> std::os::fd::RawFd {
        self._object._fd.as_raw_fd()
    }

    pub fn is_writable(&self) -> bool {
        self.writable
    }

    pub fn len(&self) -> usize {
        self.mapping.len
    }

    pub fn is_empty(&self) -> bool {
        self.mapping.len == 0
    }

    pub fn bytes(&self) -> &[u8] {
        // SAFETY: the mapping is live for `self`, read-only access cannot mutate it, and its
        // length was accepted by mmap.
        unsafe { std::slice::from_raw_parts(self.mapping.ptr.as_ptr().cast::<u8>(), self.len()) }
    }

    pub fn try_bytes_mut(&mut self) -> Result<&mut [u8]> {
        self.checked_writable()?;
        // SAFETY: `self.writable` was verified (`PROT_WRITE`) and `&mut self` guarantees exclusive
        // Rust borrow access to the live mapping.
        Ok(unsafe {
            std::slice::from_raw_parts_mut(self.mapping.ptr.as_ptr().cast::<u8>(), self.len())
        })
    }

    pub fn bytes_mut(&mut self) -> &mut [u8] {
        assert!(
            self.writable,
            "cannot mutably borrow a read-only SharedMemoryArena"
        );
        // SAFETY: `self.writable` was asserted (`PROT_WRITE`) and `&mut self` guarantees exclusive
        // Rust access to the live mapping.
        unsafe {
            std::slice::from_raw_parts_mut(self.mapping.ptr.as_ptr().cast::<u8>(), self.len())
        }
    }

    pub(crate) fn checked_writable(&self) -> Result<()> {
        if self.writable {
            Ok(())
        } else {
            Err(ShmError::invalid_input(
                "shared-memory mapping was opened read-only",
            ))
        }
    }

    pub(crate) fn checked_range(&self, offset: usize, len: usize, align: usize) -> Result<()> {
        let end = offset
            .checked_add(len)
            .ok_or_else(|| ShmError::invalid_input("shared-memory range overflow"))?;
        if end > self.len() {
            return Err(ShmError::invalid_input(
                "shared-memory range is out of bounds",
            ));
        }
        let address = self.mapping.ptr.as_ptr() as usize + offset;
        if address % align != 0 {
            return Err(ShmError::invalid_input("shared-memory view is misaligned"));
        }
        Ok(())
    }

    pub(crate) fn pointer(&self, offset: usize) -> *mut u8 {
        // SAFETY: callers validate the complete range with `checked_range` first.
        unsafe { self.mapping.ptr.as_ptr().cast::<u8>().add(offset) }
    }
}

fn map(fd: std::os::fd::RawFd, len: usize, writable: bool) -> Result<Mapping> {
    let protection = libc::PROT_READ | if writable { libc::PROT_WRITE } else { 0 };
    // SAFETY: the descriptor is valid, offset zero is page-aligned, and no pointers are retained
    // by the kernel after this call beyond the returned mapping itself.
    let ptr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            len,
            protection,
            libc::MAP_SHARED,
            fd,
            0,
        )
    };
    if ptr == libc::MAP_FAILED {
        return Err(ShmError::last_os("mmap"));
    }
    let ptr = NonNull::new(ptr).ok_or_else(|| ShmError::invalid_input("mmap returned null"))?;
    Ok(Mapping { ptr, len })
}

fn validate_name(name: &str) -> Result<(String, CString)> {
    if name.is_empty() || name == "." || name == ".." || name.contains('/') {
        return Err(ShmError::invalid_input(
            "shared-memory name must be one nonempty path component",
        ));
    }
    let posix = format!("/{name}");
    let posix_name = CString::new(posix)
        .map_err(|_| ShmError::invalid_input("shared-memory name contains a NUL byte"))?;
    Ok((name.to_owned(), posix_name))
}
