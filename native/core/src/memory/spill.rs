//! The disk tier: anonymous, self-deleting spill files.
//!
//! A spill file is deleted by the OS when its last handle closes: Windows
//! opens it with `FILE_FLAG_DELETE_ON_CLOSE`, Unix unlinks it right after
//! creation. Spilled bytes therefore can never outlive the process, even if
//! the JVM is killed. All I/O is positional, so concurrent readers can share
//! one handle without a cursor.

use std::fs::{File, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

static SPILL_SEQ: AtomicU64 = AtomicU64::new(0);

/// Chunk size for file-to-file copies, independent of the GPU transfer chunk.
const COPY_CHUNK: usize = 8 << 20;

pub fn default_spill_dir() -> PathBuf {
    std::env::var_os("HC4J_SPILL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("hc4j-spill"))
}

#[derive(Debug)]
pub struct SpillFile {
    file: Arc<File>,
    len: u64,
}

impl SpillFile {
    /// Creates an empty spill file of `len` bytes (reads as zeros).
    pub fn create(dir: &Path, len: u64) -> io::Result<Self> {
        std::fs::create_dir_all(dir)?;
        let name = format!(
            "hc4j-{}-{}.spill",
            std::process::id(),
            SPILL_SEQ.fetch_add(1, Ordering::Relaxed)
        );
        let file = open_anonymous(&dir.join(name))?;
        file.set_len(len)?;
        Ok(Self {
            file: Arc::new(file),
            len,
        })
    }

    /// A shared handle for lock-free snapshot reads. The file stays alive (and
    /// on disk) while any handle exists.
    pub fn handle(&self) -> Arc<File> {
        Arc::clone(&self.file)
    }

    pub fn write_all_at(&self, data: &[u8], offset: u64) -> io::Result<()> {
        write_all_at(&self.file, data, offset)
    }

    /// Copy-on-write rewrite: a new file holding `prefix` followed by this
    /// file's bytes from `prefix.len()` on. Readers holding the old handle
    /// keep seeing the old contents.
    pub fn rewrite_prefix(&self, dir: &Path, prefix: &[u8]) -> io::Result<SpillFile> {
        let next = SpillFile::create(dir, self.len)?;
        next.write_all_at(prefix, 0)?;
        let mut offset = prefix.len() as u64;
        let mut buf = vec![0u8; COPY_CHUNK.min((self.len - offset.min(self.len)) as usize)];
        while offset < self.len {
            let n = ((self.len - offset) as usize).min(buf.len());
            read_exact_at(&self.file, &mut buf[..n], offset)?;
            next.write_all_at(&buf[..n], offset)?;
            offset += n as u64;
        }
        Ok(next)
    }
}

#[cfg(windows)]
fn open_anonymous(path: &Path) -> io::Result<File> {
    use std::os::windows::fs::OpenOptionsExt;
    const FILE_ATTRIBUTE_TEMPORARY: u32 = 0x0000_0100;
    const FILE_FLAG_DELETE_ON_CLOSE: u32 = 0x0400_0000;
    OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .attributes(FILE_ATTRIBUTE_TEMPORARY)
        .custom_flags(FILE_FLAG_DELETE_ON_CLOSE)
        .open(path)
}

#[cfg(not(windows))]
fn open_anonymous(path: &Path) -> io::Result<File> {
    let file = OpenOptions::new().read(true).write(true).create_new(true).open(path)?;
    std::fs::remove_file(path)?;
    Ok(file)
}

#[cfg(windows)]
fn read_at(file: &File, buf: &mut [u8], offset: u64) -> io::Result<usize> {
    std::os::windows::fs::FileExt::seek_read(file, buf, offset)
}

#[cfg(not(windows))]
fn read_at(file: &File, buf: &mut [u8], offset: u64) -> io::Result<usize> {
    std::os::unix::fs::FileExt::read_at(file, buf, offset)
}

#[cfg(windows)]
fn write_at(file: &File, buf: &[u8], offset: u64) -> io::Result<usize> {
    std::os::windows::fs::FileExt::seek_write(file, buf, offset)
}

#[cfg(not(windows))]
fn write_at(file: &File, buf: &[u8], offset: u64) -> io::Result<usize> {
    std::os::unix::fs::FileExt::write_at(file, buf, offset)
}

pub fn read_exact_at(file: &File, mut buf: &mut [u8], mut offset: u64) -> io::Result<()> {
    while !buf.is_empty() {
        match read_at(file, buf, offset) {
            Ok(0) => return Err(io::Error::from(io::ErrorKind::UnexpectedEof)),
            Ok(n) => {
                buf = &mut buf[n..];
                offset += n as u64;
            }
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

pub fn write_all_at(file: &File, mut buf: &[u8], mut offset: u64) -> io::Result<()> {
    while !buf.is_empty() {
        match write_at(file, buf, offset) {
            Ok(0) => return Err(io::Error::from(io::ErrorKind::WriteZero)),
            Ok(n) => {
                buf = &buf[n..];
                offset += n as u64;
            }
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_dir() -> PathBuf {
        std::env::temp_dir().join("hc4j-spill-tests")
    }

    #[test]
    fn roundtrip_and_zero_fill() {
        let spill = SpillFile::create(&test_dir(), 16).expect("create");
        spill.write_all_at(&[1, 2, 3, 4], 4).expect("write");
        let mut out = [9u8; 16];
        read_exact_at(&spill.handle(), &mut out, 0).expect("read");
        assert_eq!(out, [0, 0, 0, 0, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn rewrite_prefix_is_copy_on_write() {
        let old = SpillFile::create(&test_dir(), 8).expect("create");
        old.write_all_at(&[1, 1, 1, 1, 2, 2, 2, 2], 0).expect("write");
        let reader = old.handle();
        let new = old.rewrite_prefix(&test_dir(), &[7, 7, 7, 7]).expect("rewrite");

        let mut via_old = [0u8; 8];
        read_exact_at(&reader, &mut via_old, 0).expect("read old");
        assert_eq!(via_old, [1, 1, 1, 1, 2, 2, 2, 2]);

        let mut via_new = [0u8; 8];
        read_exact_at(&new.handle(), &mut via_new, 0).expect("read new");
        assert_eq!(via_new, [7, 7, 7, 7, 2, 2, 2, 2]);
    }

    #[test]
    fn short_file_read_is_an_error_not_a_panic() {
        let spill = SpillFile::create(&test_dir(), 4).expect("create");
        let mut out = [0u8; 8];
        assert!(read_exact_at(&spill.handle(), &mut out, 0).is_err());
    }
}
