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

use std::io;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::time::{Duration, Instant};

use crate::{Result, ShmError};

/// Owned Linux `eventfd` used as the sleeping control-plane wakeup channel.
#[derive(Debug)]
pub struct EventFd {
    fd: OwnedFd,
}

impl EventFd {
    /// Create a non-semaphore descriptor that is closed across `exec` by default.
    pub fn new() -> Result<Self> {
        // SAFETY: `eventfd` has no pointer arguments and returns a new owned fd.
        let raw = unsafe { libc::eventfd(0, libc::EFD_CLOEXEC) };
        if raw < 0 {
            return Err(ShmError::last_os("eventfd"));
        }
        // SAFETY: a successful `eventfd` call transfers one owned descriptor.
        let fd = unsafe { OwnedFd::from_raw_fd(raw) };
        Ok(Self { fd })
    }

    pub fn as_raw_fd(&self) -> RawFd {
        self.fd.as_raw_fd()
    }

    /// Add `value` to the kernel counter.
    pub fn write(&self, value: u64) -> Result<()> {
        loop {
            // SAFETY: the descriptor is valid and the value points to eight readable bytes.
            let written = unsafe {
                libc::write(
                    self.fd.as_raw_fd(),
                    (&value as *const u64).cast(),
                    size_of::<u64>(),
                )
            };
            if written == size_of::<u64>() as isize {
                return Ok(());
            }
            if written < 0 && io::Error::last_os_error().kind() == io::ErrorKind::Interrupted {
                continue;
            }
            return Err(ShmError::last_os("eventfd write"));
        }
    }

    /// Read and reset the kernel counter.
    pub fn read(&self) -> Result<u64> {
        let mut value = 0_u64;
        loop {
            // SAFETY: the descriptor is valid and the value points to eight writable bytes.
            let read = unsafe {
                libc::read(
                    self.fd.as_raw_fd(),
                    (&mut value as *mut u64).cast(),
                    size_of::<u64>(),
                )
            };
            if read == size_of::<u64>() as isize {
                return Ok(value);
            }
            if read < 0 && io::Error::last_os_error().kind() == io::ErrorKind::Interrupted {
                continue;
            }
            return Err(ShmError::last_os("eventfd read"));
        }
    }

    /// Sleep in `poll(2)` until readable or the deadline expires.
    pub fn wait_until(&self, deadline: Instant) -> Result<()> {
        loop {
            let now = Instant::now();
            if now >= deadline {
                return Err(ShmError::invalid_input("eventfd wait timed out"));
            }
            let remaining = deadline.saturating_duration_since(now);
            let timeout_ms = remaining.as_millis().clamp(1, i32::MAX as u128) as i32;
            let mut pollfd = libc::pollfd {
                fd: self.fd.as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            };
            // SAFETY: `pollfd` points to one initialized descriptor for the duration of the call.
            let ready = unsafe { libc::poll(&mut pollfd, 1, timeout_ms) };
            if ready > 0 {
                if pollfd.revents & (libc::POLLERR | libc::POLLHUP | libc::POLLNVAL) != 0 {
                    return Err(ShmError::invalid_input("eventfd poll returned an error"));
                }
                if pollfd.revents & libc::POLLIN != 0 {
                    return Ok(());
                }
                continue;
            }
            if ready == 0 {
                return Err(ShmError::invalid_input("eventfd wait timed out"));
            }
            let error = io::Error::last_os_error();
            if error.kind() != io::ErrorKind::Interrupted {
                return Err(ShmError::new("eventfd poll", error));
            }
        }
    }

    /// Sleep for at most `timeout` until the descriptor is readable.
    pub fn wait(&self, timeout: Duration) -> Result<()> {
        self.wait_until(Instant::now() + timeout)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eventfd_accumulates_and_drains() {
        let event = EventFd::new().expect("create eventfd");
        event.write(2).expect("write first value");
        event.write(3).expect("write second value");
        event.wait(Duration::from_secs(1)).expect("readable");
        assert_eq!(event.read().expect("read counter"), 5);
    }
}
