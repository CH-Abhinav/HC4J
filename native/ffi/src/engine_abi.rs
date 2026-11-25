//! C-ABI entry points for engine lifecycle, stream batching, stats, adapter metadata,
//! and thread-local last error retrieval (`hc4j_last_error`).

use hc4rs::engine;
use hc4rs::engine::error::{
    self, HC4J_SUCCESS, Hc4jError, ffi_guard, ffi_guard_preserve_error,
};
use hc4rs::{AdapterInfoC, EngineStats};

/// Initializes the global WebGPU compute engine, tiered memory manager, and
/// background matmul warm-up.
#[unsafe(no_mangle)]
pub extern "C" fn hc4j_init_gpu() -> i32 {
    ffi_guard("hc4j_init_gpu", engine::init)
}

/// Opens a command-batching scope: ops record into one command buffer until
/// the matching `hc4j_batch_end`.
#[unsafe(no_mangle)]
pub extern "C" fn hc4j_batch_begin() -> i32 {
    ffi_guard("hc4j_batch_begin", engine::batch_begin)
}

/// Closes a batching scope. Errors from submitting the batch surface here.
#[unsafe(no_mangle)]
pub extern "C" fn hc4j_batch_end() -> i32 {
    ffi_guard("hc4j_batch_end", engine::batch_end)
}

/// Submits all pending work and blocks until the GPU has finished executing it.
#[unsafe(no_mangle)]
pub extern "C" fn hc4j_synchronize() -> i32 {
    ffi_guard("hc4j_synchronize", engine::synchronize)
}

/// # Safety
/// `out` must be non-null and point to writable memory for one [`EngineStats`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hc4j_engine_stats(out: *mut EngineStats) -> i32 {
    ffi_guard("hc4j_engine_stats", || {
        if out.is_null() {
            return Err(Hc4jError::InvalidParam("null stats pointer"));
        }
        let stats = engine::stats()?;
        unsafe { out.write_unaligned(stats) };
        Ok(())
    })
}

/// # Safety
/// `out` must be non-null and point to writable memory for one [`AdapterInfoC`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hc4j_adapter_info(out: *mut AdapterInfoC) -> i32 {
    ffi_guard("hc4j_adapter_info", || {
        if out.is_null() {
            return Err(Hc4jError::InvalidParam("null adapter info pointer"));
        }
        let info = engine::query_adapter_info()?;
        unsafe { out.write_unaligned(info) };
        Ok(())
    })
}

/// Copies the thread-local diagnostic message from the most recent failed FFI call
/// into `buf` (up to `cap - 1` UTF-8 bytes + NUL terminator), returning the number
/// of bytes written (`0` if no error is pending).
///
/// When `buf` is null and `cap == 0`, returns the full UTF-8 byte length required
/// without modifying any buffer.
///
/// # Safety
/// If `buf` is non-null and `cap > 0`, `buf` must point to at least `cap` writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hc4j_last_error(buf: *mut u8, cap: u32) -> i32 {
    let mut written: i32 = 0;
    let status = ffi_guard_preserve_error("hc4j_last_error", || {
        let msg = error::last_error();
        let bytes = msg.as_bytes();
        if buf.is_null() {
            if cap > 0 {
                return Err(Hc4jError::InvalidParam("null error buffer with non-zero cap"));
            }
            written = bytes.len().min(i32::MAX as usize) as i32;
            return Ok(());
        }
        if cap == 0 {
            written = bytes.len().min(i32::MAX as usize) as i32;
            return Ok(());
        }
        let n = bytes.len().min((cap as usize).saturating_sub(1));
        if n > 0 {
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf, n) };
        }
        unsafe { *buf.add(n) = 0 };
        written = n.min(i32::MAX as usize) as i32;
        Ok(())
    });
    if status == HC4J_SUCCESS { written } else { status }
}
