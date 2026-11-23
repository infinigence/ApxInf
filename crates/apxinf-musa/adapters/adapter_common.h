// Shared surface of the adapters that allocate on first use.
//
// `mubin_fa_adapter.mu` (the MP22 assembly), `mutlass_fa_adapter.mu` (the
// vendored mp_31 FMHA) and `mudnn_adapter.mu` are mutually independent: the two
// FA adapters are mutually exclusive targets -- a build has at most one -- and
// their metadata tables are deliberately separate, because those kernels take
// different arguments (see `adapters/README.md`). What all of them share is the
// bookkeeping that has nothing to do with any of their ABIs: the per-thread
// failure message, the capture guard each one asks before its first allocation
// for a shape, and the two device questions below.
//
// `custom_kernels.mu` includes this for the device helpers: it builds the rope
// frequency table on first use, so it allocates like the rest. `mublas_adapter.mu`
// is the one that is not here: it allocates nothing on first use, and the state
// it does keep is already keyed by `(device, stream)`.

#pragma once

#include <cstdarg>
#include <cstdio>

#include <musa_runtime.h>

namespace apxinf_adapters {

/// The device `stream` belongs to, or `-1` when the runtime cannot name one.
///
/// Not `musaGetDevice`: that answers which device this *host thread* last
/// selected, and this backend changes that selection as it allocates, copies
/// and frees, so the two can disagree. What a cache keyed by this needs is the
/// device whose memory a launch on `stream` will read.
///
/// `APXINF_MUSA_HAS_STREAM_DEVICE` is set by `build.rs` from a compile probe,
/// not from `MUSART_VERSION`: the 5.1 toolkit this port runs on defines that
/// version as 50100 while declaring and exporting `musaStreamGetDevice`, so a
/// version test answers "no" on a toolkit that has it. On a toolkit without it,
/// the thread's own selection is the only answer available, and that is what
/// this falls back to — the answer this code gave before a stream could be
/// asked. `-1` is "unknown" and is a key like any other; it is never passed
/// to `musaSetDevice`.
inline int stream_device(void* stream) {
  int device = -1;
#if defined(APXINF_MUSA_HAS_STREAM_DEVICE)
  if (musaStreamGetDevice(static_cast<musaStream_t>(stream), &device) == musaSuccess) {
    return device;
  }
#else
  (void)stream;
#endif
  return musaGetDevice(&device) == musaSuccess ? device : -1;
}

/// Select `device` before allocating on it.
///
/// `musaMalloc` and a stream-less `musaMemcpy` resolve against the *current*
/// device, not against the stream the pointer will be read on, so an entry
/// keyed by a device it does not live on is worse than no entry at all. The
/// same discipline `MusaBuffer::alloc_with_stream` follows on the Rust side.
/// `-1` (see `stream_device`) is a no-op rather than a failure: there is
/// nothing to select, and the allocation that follows fails on its own if it
/// must.
inline bool select_device(int device) {
  return device < 0 || musaSetDevice(device) == musaSuccess;
}

/// The per-thread buffer a failed call leaves its message in.
///
/// One buffer per thread rather than one per adapter: no process loads two of
/// them for the same purpose, so a second copy would only be a second thing to
/// keep in step. Each adapter still exports its own `..._last_error()`
/// accessor name, because those are the C ABI the Rust side binds to.
inline char* error_buffer() {
  static thread_local char buffer[512];
  return buffer;
}

/// Record why the current call failed, in `printf` form.
inline void set_error(const char* format, ...) {
  char* buffer = error_buffer();
  va_list arguments;
  va_start(arguments, format);
  vsnprintf(buffer, 512, format, arguments);
  va_end(arguments);
}

/// Whether `stream` is recording a graph right now.
///
/// Asked before the first allocation for a shape, because a captured graph bakes
/// the pointers it was recorded with: an allocation that happens during capture
/// is either refused by the driver with no attribution, or -- worse -- succeeds
/// and leaves the graph reading a buffer that no longer exists.
///
/// A stream whose capture status cannot be read answers "capturing": that is the
/// case where guessing "not capturing" would let the allocation into a graph.
/// All three callers refuse on `true`; what differs is only how they say so --
/// the FA adapters through their `..._last_error()`, muDNN through its status
/// codes, since that is the channel each already has.
inline bool capturing(void* stream) {
  musaStreamCaptureStatus status = musaStreamCaptureStatusNone;
  if (musaStreamIsCapturing(static_cast<musaStream_t>(stream), &status) != musaSuccess) {
    return true;
  }
  return status != musaStreamCaptureStatusNone;
}

}  // namespace apxinf_adapters
