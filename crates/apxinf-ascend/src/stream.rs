//! Stream and event wrappers.
//!
//! Lightweight handles that borrow the context's stream. Events are owned
//! and created on demand for graph capture synchronization.

use crate::ffi::acl::{self, aclrtEvent, aclrtStream};
use crate::context::AscendContext;

/// Borrowed view of the context's default stream.
pub struct AscendStream<'a> {
    raw: aclrtStream,
    _ctx: &'a AscendContext,
}

impl<'a> AscendStream<'a> {
    pub fn new(ctx: &'a AscendContext) -> Self {
        Self {
            raw: ctx.raw_stream(),
            _ctx: ctx,
        }
    }

    pub fn raw(&self) -> aclrtStream {
        self.raw
    }

    pub fn synchronize(&self) -> Result<(), String> {
        let ret = unsafe { acl::aclrtSynchronizeStream(self.raw) };
        acl::check(ret, "aclrtSynchronizeStream")
    }
}

/// Owned ACL event, used for stream synchronization during graph capture.
pub struct AscendEvent {
    raw: aclrtEvent,
}

impl AscendEvent {
    pub fn new() -> Result<Self, String> {
        let mut raw: aclrtEvent = std::ptr::null_mut();
        let ret = unsafe { acl::aclrtCreateEvent(&mut raw) };
        acl::check(ret, "aclrtCreateEvent")?;
        Ok(Self { raw })
    }

    pub fn raw(&self) -> aclrtEvent {
        self.raw
    }

    pub fn record(&self, stream: aclrtStream) -> Result<(), String> {
        let ret = unsafe { acl::aclrtRecordEvent(self.raw, stream) };
        acl::check(ret, "aclrtRecordEvent")
    }

    pub fn synchronize(&self) -> Result<(), String> {
        let ret = unsafe { acl::aclrtSynchronizeEvent(self.raw) };
        acl::check(ret, "aclrtSynchronizeEvent")
    }
}

impl Drop for AscendEvent {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { acl::aclrtDestroyEvent(self.raw) };
        }
    }
}
