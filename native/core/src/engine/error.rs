//! Status codes shared across the FFI boundary and the internal error type
//! every fallible path funnels into.

use std::fmt;

pub const HC4J_SUCCESS: i32 = 0;
pub const HC4J_ERR_NOT_FOUND: i32 = -1;
pub const HC4J_ERR_INVALID_PARAM: i32 = -2;
pub const HC4J_ERR_GPU_READBACK: i32 = -3;
/// VRAM, host RAM and the spill tier were all unable to satisfy a request.
pub const HC4J_ERR_OUT_OF_MEMORY: i32 = -4;
/// Engine unavailable, device lost, or a GPU wait timed out.
pub const HC4J_ERR_DEVICE: i32 = -5;
/// wgpu rejected a shader, pipeline, bind group or submission.
pub const HC4J_ERR_GPU_VALIDATION: i32 = -6;
/// Reading or writing a spill file failed.
pub const HC4J_ERR_HOST_IO: i32 = -7;
/// The request is valid but this code path cannot serve it (e.g. a strided
/// kernel on a tensor larger than the storage-binding limit).
pub const HC4J_ERR_UNSUPPORTED: i32 = -8;

#[derive(Debug)]
pub enum Hc4jError {
    NotFound,
    InvalidParam(&'static str),
    Readback(String),
    OutOfMemory,
    Device(String),
    GpuValidation(String),
    HostIo(std::io::Error),
    Unsupported(&'static str),
}

pub type Hc4jResult<T> = Result<T, Hc4jError>;

impl Hc4jError {
    pub fn code(&self) -> i32 {
        match self {
            Hc4jError::NotFound => HC4J_ERR_NOT_FOUND,
            Hc4jError::InvalidParam(_) => HC4J_ERR_INVALID_PARAM,
            Hc4jError::Readback(_) => HC4J_ERR_GPU_READBACK,
            Hc4jError::OutOfMemory => HC4J_ERR_OUT_OF_MEMORY,
            Hc4jError::Device(_) => HC4J_ERR_DEVICE,
            Hc4jError::GpuValidation(_) => HC4J_ERR_GPU_VALIDATION,
            Hc4jError::HostIo(_) => HC4J_ERR_HOST_IO,
            Hc4jError::Unsupported(_) => HC4J_ERR_UNSUPPORTED,
        }
    }

    /// Converts an error captured by a wgpu error scope.
    pub fn from_wgpu(err: wgpu::Error) -> Self {
        match err {
            wgpu::Error::OutOfMemory { .. } => Hc4jError::OutOfMemory,
            wgpu::Error::Validation { description, .. } => Hc4jError::GpuValidation(description),
            wgpu::Error::Internal { description, .. } => Hc4jError::Device(description),
        }
    }
}

impl Hc4jError {
    /// Appends diagnostic context to the message-carrying variants. Used to
    /// name the ops a failed command batch contained, since a batch is
    /// validated when it is submitted rather than when each op is recorded.
    pub fn with_context(self, context: &str) -> Self {
        match self {
            Hc4jError::Readback(msg) => Hc4jError::Readback(format!("{msg} ({context})")),
            Hc4jError::Device(msg) => Hc4jError::Device(format!("{msg} ({context})")),
            Hc4jError::GpuValidation(msg) => Hc4jError::GpuValidation(format!("{msg} ({context})")),
            other => other,
        }
    }
}

impl fmt::Display for Hc4jError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Hc4jError::NotFound => f.write_str("tensor handle not found"),
            Hc4jError::InvalidParam(why) => write!(f, "invalid parameter: {why}"),
            Hc4jError::Readback(why) => write!(f, "GPU readback failed: {why}"),
            Hc4jError::OutOfMemory => f.write_str("out of memory in every tier"),
            Hc4jError::Device(why) => write!(f, "GPU device error: {why}"),
            Hc4jError::GpuValidation(why) => write!(f, "GPU validation error: {why}"),
            Hc4jError::HostIo(err) => write!(f, "spill I/O error: {err}"),
            Hc4jError::Unsupported(why) => write!(f, "unsupported: {why}"),
        }
    }
}

impl From<std::io::Error> for Hc4jError {
    fn from(err: std::io::Error) -> Self {
        Hc4jError::HostIo(err)
    }
}

static TRACE: AtomicBool = AtomicBool::new(false);

pub fn init_trace_from_env() {
    let on = std::env::var("HC4J_TRACE").is_ok_and(|v| v != "0" && !v.is_empty());
    TRACE.store(on, Ordering::Relaxed);
}

pub fn trace_enabled() -> bool {
    TRACE.load(Ordering::Relaxed)
}

/// Verbose paging diagnostics, enabled with `HC4J_TRACE=1`.
#[macro_export]
macro_rules! hc4j_trace {
    ($($arg:tt)*) => {
        if $crate::engine::error::trace_enabled() {
            eprintln!("[HC4J trace] {}", format_args!($($arg)*));
        }
    };
}

use std::cell::RefCell;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, Ordering};

thread_local! {
    static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
}

/// Records the last diagnostic error message on the current thread.
pub fn set_last_error(msg: impl Into<String>) {
    let msg = msg.into();
    LAST_ERROR.with(|slot| *slot.borrow_mut() = msg);
}

/// Clears any stored thread-local error diagnostic.
pub fn clear_last_error() {
    LAST_ERROR.with(|slot| slot.borrow_mut().clear());
}

/// Returns the most recent thread-local error message recorded by [`ffi_guard`].
pub fn last_error() -> String {
    LAST_ERROR.with(|slot| slot.borrow().clone())
}

/// Runs an FFI body, mapping `Err` to its status code, recording the
/// diagnostic message in thread-local storage for `hc4j_last_error`, and
/// converting a panic into `HC4J_ERR_DEVICE`.
pub fn ffi_guard(op: &str, body: impl FnOnce() -> Hc4jResult<()>) -> i32 {
    ffi_guard_inner(op, true, body)
}

/// Variant of [`ffi_guard`] that preserves any previously recorded thread-local
/// error message when `body` succeeds (used by `hc4j_last_error` itself).
pub fn ffi_guard_preserve_error(op: &str, body: impl FnOnce() -> Hc4jResult<()>) -> i32 {
    ffi_guard_inner(op, false, body)
}

fn ffi_guard_inner(op: &str, clear_on_ok: bool, body: impl FnOnce() -> Hc4jResult<()>) -> i32 {
    match catch_unwind(AssertUnwindSafe(body)) {
        Ok(Ok(())) => {
            if clear_on_ok {
                clear_last_error();
            }
            HC4J_SUCCESS
        }
        Ok(Err(err)) => {
            set_last_error(format!("{op}: {err}"));
            // NotFound and InvalidParam are caller mistakes that callers turn into
            // exceptions themselves, so logging them to stderr would only be noise.
            if !matches!(err, Hc4jError::NotFound | Hc4jError::InvalidParam(_)) {
                eprintln!("[HC4J] {op} failed: {err}");
            }
            err.code()
        }
        Err(_) => {
            set_last_error(format!("{op}: internal panic contained at FFI boundary"));
            eprintln!("[HC4J] {op} panicked; the error was contained at the FFI boundary");
            HC4J_ERR_DEVICE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{HC4J_ERR_GPU_VALIDATION, HC4J_ERR_NOT_FOUND, Hc4jError};

    #[test]
    fn context_names_the_batch_that_failed() {
        let err = Hc4jError::GpuValidation("bad bind group".to_string())
            .with_context("batch of 3 ops: elementwise, matmul (gemm), transpose");
        assert_eq!(
            err.to_string(),
            "GPU validation error: bad bind group (batch of 3 ops: elementwise, matmul (gemm), transpose)"
        );
        assert_eq!(err.code(), HC4J_ERR_GPU_VALIDATION, "context must not change the status code");
    }

    #[test]
    fn context_leaves_message_free_variants_alone() {
        let err = Hc4jError::NotFound.with_context("batch of 1 ops: elementwise");
        assert_eq!(err.to_string(), "tensor handle not found");
        assert_eq!(err.code(), HC4J_ERR_NOT_FOUND);
    }

    #[test]
    fn ffi_guard_records_preserves_and_clears_last_error() {
        super::clear_last_error();
        let rc = super::ffi_guard("op_fail", || Err(Hc4jError::InvalidParam("bad arg")));
        assert_eq!(rc, super::HC4J_ERR_INVALID_PARAM);
        assert!(super::last_error().contains("op_fail"));
        assert!(super::last_error().contains("bad arg"));

        let rc_read = super::ffi_guard_preserve_error("hc4j_last_error", || Ok(()));
        assert_eq!(rc_read, super::HC4J_SUCCESS);
        assert!(super::last_error().contains("op_fail"), "preserve variant must not clear error");

        let rc_ok = super::ffi_guard("op_ok", || Ok(()));
        assert_eq!(rc_ok, super::HC4J_SUCCESS);
        assert!(super::last_error().is_empty());
    }
}
