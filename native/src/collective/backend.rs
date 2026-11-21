use crate::error::Hc4jResult;

// ============================================================================
// Transport constants matching Java hc4j.collective.Transport
// ============================================================================

pub const TRANSPORT_HOST_STAGED: u32 = 0;
pub const TRANSPORT_PEER_TO_PEER: u32 = 1;
pub const TRANSPORT_HOST_SHARED_DMA: u32 = 2; // Reserved for Step 3B Windows Tier 1.5

/// How a directed rank-to-rank hop moves its bytes.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Transport {
    HostStaged,
    PeerToPeer,
    HostSharedDma,
}

impl Transport {
    pub fn code(self) -> u32 {
        match self {
            Transport::HostStaged => TRANSPORT_HOST_STAGED,
            Transport::PeerToPeer => TRANSPORT_PEER_TO_PEER,
            Transport::HostSharedDma => TRANSPORT_HOST_SHARED_DMA,
        }
    }
}

// ============================================================================
// LinkInfo matching the exact 24-byte C-ABI Link layout
// ============================================================================

/// One directed edge of the topology graph (exact 24-byte C-ABI layout).
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct LinkInfo {
    pub src_rank: u32,
    pub dst_rank: u32,
    /// Measured throughput of the hop, in GB/s.
    pub bandwidth_gbps: f64,
    /// [`Transport::code`] / `TRANSPORT_*`.
    pub transport: u32,
    /// 1 when both ranks sit on the same physical adapter (loopback), so the
    /// figure is an intra-GPU copy rather than a link between two devices.
    pub loopback: u32,
}

pub type Link = LinkInfo;

/// Above this, a link is reported as fast enough to treat as peer-class.
pub const PEER_CLASS_GBPS: f64 = 8.0;

/// Whether a measured hop reaches peer-class bandwidth.
pub fn is_peer_class(gbps: f64) -> bool {
    gbps >= PEER_CLASS_GBPS
}

// ============================================================================
// CollectiveBackend trait
// ============================================================================

/// Backend-agnostic abstraction for multi-GPU collective communications.
pub(crate) trait CollectiveBackend: Send + Sync {
    #[allow(dead_code)]
    fn backend_name(&self) -> &'static str;
    fn ranks(&self) -> usize;
    fn device_name(&self, rank: usize) -> Hc4jResult<String>;
    fn links(&self) -> &[LinkInfo];
    fn alloc(&self, rank: usize, elements: u64) -> Hc4jResult<u64>;
    fn free(&self, buffer_id: u64) -> Hc4jResult<()>;
    fn write(&self, buffer_id: u64, data: &[f32]) -> Hc4jResult<()>;
    fn read(&self, buffer_id: u64, out: &mut [f32]) -> Hc4jResult<()>;
    fn allreduce_sum(&self, buffers: &[u64]) -> Hc4jResult<()>;
}
