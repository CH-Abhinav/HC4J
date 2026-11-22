//! Zero-dependency persistent disk topology and profile cache.
//!
//! Keyed by deterministic hardware/driver fingerprint so repeat startups load in < 1 ms.
//! Bypassed by HC4J_COLLECTIVE_REPROBE=1 or non-auto HC4J_COLLECTIVE_FORCE_TIER or HC4J_COLLECTIVE_CACHE=0.

use std::path::{Path, PathBuf};

/// Versioned plain-text header.
pub const CACHE_HEADER: &str = "HC4J_TOPO_V2";

/// A cached device entry.
#[derive(Debug, Clone, PartialEq)]
pub struct CachedDevice {
    pub phys_idx: usize,
    pub vram_bytes: u64,
    pub compute_gflops: f64,
}

/// A cached edge entry.
#[derive(Debug, Clone, PartialEq)]
pub struct CachedEdge {
    pub src_phys_idx: usize,
    pub dst_phys_idx: usize,
    pub transport: u32,
    pub bandwidth_gbps: f64,
    pub latency_us: f64,
}

/// Loaded cached topology.
#[derive(Debug, Clone, PartialEq)]
pub struct CachedTopology {
    pub devices: Vec<CachedDevice>,
    pub edges: Vec<CachedEdge>,
}

/// Physical device summary for fingerprint calculation.
#[derive(Debug, Clone)]
pub struct PhysDeviceSummary<'a> {
    pub vendor_id: u32,
    pub device_id: u32,
    pub driver_version: u32,
    pub device_type: u32,
    pub device_name: &'a str,
}

/// Computes a deterministic 64-bit FNV-1a hex key from hardware/driver properties.
pub fn compute_fingerprint(include_cpu: bool, devices: &[PhysDeviceSummary]) -> String {
    let mut h = 0xcbf29ce484222325u64;
    let prime = 0x100000001b3u64;

    let mut feed_bytes = |bytes: &[u8]| {
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(prime);
        }
    };

    feed_bytes(std::env::consts::OS.as_bytes());
    feed_bytes(std::env::consts::ARCH.as_bytes());
    feed_bytes(env!("CARGO_PKG_VERSION").as_bytes());
    feed_bytes(&[if include_cpu { 1 } else { 0 }]);

    for dev in devices {
        feed_bytes(&dev.vendor_id.to_le_bytes());
        feed_bytes(&dev.device_id.to_le_bytes());
        feed_bytes(&dev.driver_version.to_le_bytes());
        feed_bytes(&dev.device_type.to_le_bytes());
        feed_bytes(dev.device_name.as_bytes());
    }

    format!("{h:016x}")
}

/// Checks whether caching is bypassed by environment variables.
pub fn should_bypass_cache() -> bool {
    if let Ok(val) = std::env::var("HC4J_COLLECTIVE_REPROBE") {
        let trimmed = val.trim();
        if trimmed == "1" || trimmed.eq_ignore_ascii_case("true") {
            return true;
        }
    }
    if let Ok(val) = std::env::var("HC4J_COLLECTIVE_CACHE") {
        let trimmed = val.trim();
        if trimmed == "0" || trimmed.eq_ignore_ascii_case("false") {
            return true;
        }
    }
    if let Ok(val) = std::env::var("HC4J_COLLECTIVE_FORCE_TIER") {
        let trimmed = val.trim().to_ascii_lowercase();
        if !trimmed.is_empty() && trimmed != "auto" {
            return true;
        }
    }
    false
}

/// Resolves the cache file path for the given fingerprint.
pub fn get_cache_path(fingerprint: &str) -> PathBuf {
    let dir = if let Ok(custom) = std::env::var("HC4J_CACHE_DIR") {
        PathBuf::from(custom).join("topology")
    } else {
        #[cfg(target_os = "windows")]
        {
            if let Ok(local_appdata) = std::env::var("LOCALAPPDATA") {
                PathBuf::from(local_appdata).join("hc4j").join("topology")
            } else {
                std::env::temp_dir().join("hc4j-topology")
            }
        }
        #[cfg(target_os = "macos")]
        {
            if let Ok(home) = std::env::var("HOME") {
                PathBuf::from(home)
                    .join("Library")
                    .join("Caches")
                    .join("hc4j")
                    .join("topology")
            } else {
                std::env::temp_dir().join("hc4j-topology")
            }
        }
        #[cfg(not(any(target_os = "windows", target_os = "macos")))]
        {
            if let Ok(xdg) = std::env::var("XDG_CACHE_HOME") {
                PathBuf::from(xdg).join("hc4j").join("topology")
            } else if let Ok(home) = std::env::var("HOME") {
                PathBuf::from(home).join(".cache").join("hc4j").join("topology")
            } else {
                std::env::temp_dir().join("hc4j-topology")
            }
        }
    };
    dir.join(format!("topo_{fingerprint}.cache"))
}

/// Formats cached topology into the versioned plain-text representation.
pub fn serialize_cache(devices: &[CachedDevice], edges: &[CachedEdge]) -> String {
    let mut out = String::with_capacity(128 + devices.len() * 40 + edges.len() * 40);
    out.push_str(CACHE_HEADER);
    out.push('\n');

    for dev in devices {
        out.push_str(&format!(
            "DEV\t{}\t{}\t{:.4}\n",
            dev.phys_idx, dev.vram_bytes, dev.compute_gflops
        ));
    }
    for edge in edges {
        out.push_str(&format!(
            "EDGE\t{}\t{}\t{}\t{:.4}\t{:.2}\n",
            edge.src_phys_idx, edge.dst_phys_idx, edge.transport, edge.bandwidth_gbps, edge.latency_us
        ));
    }
    out
}

/// Parses the versioned plain-text cache, validating device count and completeness.
pub fn parse_cache(content: &str, expected_phys_devices: usize) -> Option<CachedTopology> {
    let mut lines = content.lines();
    let header = lines.next()?.trim();
    if header != CACHE_HEADER {
        return None;
    }

    let mut devices = Vec::with_capacity(expected_phys_devices);
    let mut edges = Vec::new();

    for line in lines {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let parts: Vec<&str> = trimmed.split('\t').collect();
        match parts.first().copied() {
            Some("DEV") if parts.len() == 4 => {
                let phys_idx: usize = parts[1].parse().ok()?;
                let vram_bytes: u64 = parts[2].parse().ok()?;
                let compute_gflops: f64 = parts[3].parse().ok()?;
                devices.push(CachedDevice {
                    phys_idx,
                    vram_bytes,
                    compute_gflops,
                });
            }
            Some("EDGE") if parts.len() == 6 => {
                let src_phys_idx: usize = parts[1].parse().ok()?;
                let dst_phys_idx: usize = parts[2].parse().ok()?;
                let transport: u32 = parts[3].parse().ok()?;
                let bandwidth_gbps: f64 = parts[4].parse().ok()?;
                let latency_us: f64 = parts[5].parse().ok()?;
                edges.push(CachedEdge {
                    src_phys_idx,
                    dst_phys_idx,
                    transport,
                    bandwidth_gbps,
                    latency_us,
                });
            }
            _ => return None, // malformed line
        }
    }

    // Verify all expected devices are present
    if devices.len() != expected_phys_devices {
        return None;
    }
    devices.sort_by_key(|d| d.phys_idx);
    for (i, dev) in devices.iter().enumerate() {
        if dev.phys_idx != i {
            return None;
        }
    }

    // Verify all expected directed edges are present when expected_phys_devices > 1
    if expected_phys_devices > 1 {
        let expected_edges = expected_phys_devices * (expected_phys_devices - 1);
        if edges.len() != expected_edges {
            return None;
        }
    }

    Some(CachedTopology { devices, edges })
}

/// Atomically writes content to path using a temp file.
pub fn write_cache_atomic(path: &Path, content: &str) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let pid = std::process::id();
    let tmp_path = path.with_extension(format!("tmp.{pid}"));
    std::fs::write(&tmp_path, content)?;
    let _ = std::fs::rename(&tmp_path, path);
    Ok(())
}

/// Attempts to load cached topology from disk.
pub fn load_cache(path: &Path, expected_phys_devices: usize) -> Option<CachedTopology> {
    let content = std::fs::read_to_string(path).ok()?;
    parse_cache(&content, expected_phys_devices)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_roundtrip() {
        let temp_dir = std::env::temp_dir().join("hc4j_test_cache_roundtrip");
        let path = temp_dir.join("topo_test123.cache");

        let devices = vec![
            CachedDevice {
                phys_idx: 0,
                vram_bytes: 4 * 1024 * 1024 * 1024,
                compute_gflops: 185.5,
            },
            CachedDevice {
                phys_idx: 1,
                vram_bytes: 8 * 1024 * 1024 * 1024,
                compute_gflops: 500.0,
            },
        ];
        let edges = vec![
            CachedEdge {
                src_phys_idx: 0,
                dst_phys_idx: 1,
                transport: 1,
                bandwidth_gbps: 22.5,
                latency_us: 150.0,
            },
            CachedEdge {
                src_phys_idx: 1,
                dst_phys_idx: 0,
                transport: 1,
                bandwidth_gbps: 22.0,
                latency_us: 180.0,
            },
        ];

        let serialized = serialize_cache(&devices, &edges);
        write_cache_atomic(&path, &serialized).expect("write should succeed");

        let loaded = load_cache(&path, 2).expect("cache load should succeed");
        assert_eq!(loaded.devices, devices);
        assert_eq!(loaded.edges, edges);

        // Verify that old V1 cache fails to load
        let v1_content = "HC4J_TOPO_V1\nDEV\t0\t8589934592\t1000.0000\nDEV\t1\t8589934592\t800.0000\nEDGE\t0\t1\t1\t25.0000\nEDGE\t1\t0\t1\t22.0000\n";
        assert!(parse_cache(v1_content, 2).is_none());

        // Cleanup
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&temp_dir);
    }

    #[test]
    fn test_corrupt_cache_returns_none() {
        let temp_dir = std::env::temp_dir().join("hc4j_test_cache_corrupt");
        let path = temp_dir.join("corrupt.cache");

        // Malformed header
        let _ = write_cache_atomic(&path, "NOT_HC4J\nDEV\t0\t100\t100");
        assert!(load_cache(&path, 1).is_none());

        // Malformed lines
        let _ = write_cache_atomic(&path, "HC4J_TOPO_V2\nDEV\tinvalid\t100\t100");
        assert!(load_cache(&path, 1).is_none());

        // Incomplete device count
        let _ = write_cache_atomic(&path, "HC4J_TOPO_V2\nDEV\t0\t100\t100\n");
        assert!(load_cache(&path, 2).is_none());

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&temp_dir);
    }

    #[test]
    fn test_fingerprint_difference() {
        let dev1 = PhysDeviceSummary {
            vendor_id: 0x8086,
            device_id: 0x5917,
            driver_version: 1,
            device_type: 1,
            device_name: "Intel UHD 620",
        };
        let dev2 = PhysDeviceSummary {
            vendor_id: 0x10de,
            device_id: 0x2684,
            driver_version: 1,
            device_type: 2,
            device_name: "RTX 4090",
        };

        let fp1 = compute_fingerprint(false, std::slice::from_ref(&dev1));
        let fp2 = compute_fingerprint(false, std::slice::from_ref(&dev2));
        let fp1_cpu = compute_fingerprint(true, &[dev1]);

        assert_ne!(fp1, fp2);
        assert_ne!(fp1, fp1_cpu);
        assert_eq!(fp1.len(), 16);
    }
}
