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

//! Linux POSIX shared-memory transport and safe typed Mode B protocol views.

mod arena;
mod eventfd;
mod protocol;

pub use arena::SharedMemoryArena;
pub use eventfd::EventFd;
pub use protocol::{ResultSlot, ResultSlotMut, ShmLayout, StepPlanSlot, StepPlanSlotMut};

use std::fmt;
use std::io;

/// Transport error with the failing system operation attached.
#[derive(Debug)]
pub struct ShmError {
    context: &'static str,
    source: io::Error,
}

impl ShmError {
    pub fn new(context: &'static str, source: io::Error) -> Self {
        Self { context, source }
    }

    pub fn last_os(context: &'static str) -> Self {
        Self::new(context, io::Error::last_os_error())
    }

    pub fn invalid_input(message: &'static str) -> Self {
        Self::new(
            message,
            io::Error::new(io::ErrorKind::InvalidInput, message),
        )
    }
}

impl fmt::Display for ShmError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.context, self.source)
    }
}

impl std::error::Error for ShmError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

pub type Result<T> = std::result::Result<T, ShmError>;
