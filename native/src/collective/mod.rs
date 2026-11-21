pub mod abi;
pub mod backend;
pub mod wgpu_backend;

pub use abi::*;
pub use backend::*;
pub use wgpu_backend::*;

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::error::{Hc4jError, Hc4jResult};

static CONTEXTS: OnceLock<Mutex<HashMap<u64, Arc<dyn CollectiveBackend>>>> = OnceLock::new();
static NEXT_CONTEXT: Mutex<u64> = Mutex::new(1);

fn contexts() -> &'static Mutex<HashMap<u64, Arc<dyn CollectiveBackend>>> {
    CONTEXTS.get_or_init(|| Mutex::new(HashMap::new()))
}

pub(crate) fn with_backend<R>(handle: u64, f: impl FnOnce(&dyn CollectiveBackend) -> Hc4jResult<R>) -> Hc4jResult<R> {
    let backend = {
        let guard = contexts().lock().unwrap_or_else(|p| p.into_inner());
        guard.get(&handle).cloned().ok_or(Hc4jError::NotFound)?
    };
    f(&*backend)
}

pub(crate) fn register_backend(backend: Arc<dyn CollectiveBackend>) -> u64 {
    let mut ids = NEXT_CONTEXT.lock().unwrap_or_else(|p| p.into_inner());
    let handle = *ids;
    *ids += 1;
    contexts().lock().unwrap_or_else(|p| p.into_inner()).insert(handle, backend);
    handle
}

pub(crate) fn destroy_backend(handle: u64) -> Hc4jResult<()> {
    contexts()
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .remove(&handle)
        .map(|_| ())
        .ok_or(Hc4jError::NotFound)
}

/// Creates a collective backend instance according to `HC4J_COLLECTIVE_BACKEND`.
pub(crate) fn create_backend(ranks: u32) -> Hc4jResult<Arc<dyn CollectiveBackend>> {
    let backend_var = std::env::var("HC4J_COLLECTIVE_BACKEND")
        .map(|v| v.to_ascii_lowercase())
        .unwrap_or_else(|_| "auto".to_string());
    match backend_var.as_str() {
        "auto" | "wgpu" | "" => {
            let backend = WgpuStagedBackend::create(ranks as usize)?;
            crate::hc4j_trace!("created collective backend: {}", backend.backend_name());
            Ok(Arc::new(backend))
        }
        "vulkan" => Err(Hc4jError::Unsupported("vulkan-p2p backend not yet enabled")),
        other => {
            eprintln!("[HC4J] unknown collective backend '{other}'");
            Err(Hc4jError::InvalidParam("unknown collective backend"))
        }
    }
}
