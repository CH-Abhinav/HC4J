//! Lock-free byte accounting for one memory tier.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Tracks bytes committed to a tier against an adjustable limit. Reservations
/// are RAII: dropping a [`Reservation`] returns its bytes, so an allocation's
/// accounting can never outlive or leak past the allocation itself.
#[derive(Debug)]
pub struct Budget {
    limit: AtomicU64,
    used: AtomicU64,
}

impl Budget {
    pub fn new(limit: u64) -> Arc<Self> {
        Arc::new(Self {
            limit: AtomicU64::new(limit),
            used: AtomicU64::new(0),
        })
    }

    pub fn limit(&self) -> u64 {
        self.limit.load(Ordering::Acquire)
    }

    pub fn used(&self) -> u64 {
        self.used.load(Ordering::Acquire)
    }

    pub fn set_limit(&self, limit: u64) {
        self.limit.store(limit, Ordering::Release);
    }

    /// Lowers the limit to `ceiling` if it is currently higher; returns the
    /// resulting limit. Used when the driver reports OOM before the software
    /// budget is exhausted, i.e. the configured budget overestimates reality.
    pub fn clamp_limit(&self, ceiling: u64) -> u64 {
        let previous = self.limit.fetch_min(ceiling, Ordering::AcqRel);
        previous.min(ceiling)
    }

    /// Commits `bytes` if they fit under the limit.
    pub fn try_reserve(self: &Arc<Self>, bytes: u64) -> Option<Reservation> {
        let mut current = self.used.load(Ordering::Acquire);
        loop {
            let next = current.checked_add(bytes)?;
            if next > self.limit() {
                return None;
            }
            match self.used.compare_exchange_weak(current, next, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => {
                    return Some(Reservation {
                        budget: Arc::clone(self),
                        bytes,
                    });
                }
                Err(actual) => current = actual,
            }
        }
    }

    fn release(&self, bytes: u64) {
        // Saturating so a bookkeeping bug can never wrap `used` to ~u64::MAX
        // and wedge every future allocation.
        let _ = self.used.fetch_update(Ordering::AcqRel, Ordering::Acquire, |used| {
            Some(used.saturating_sub(bytes))
        });
    }
}

#[derive(Debug)]
pub struct Reservation {
    budget: Arc<Budget>,
    bytes: u64,
}

impl Reservation {
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        self.budget.release(self.bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserves_up_to_limit_and_releases_on_drop() {
        let budget = Budget::new(100);
        let a = budget.try_reserve(60).expect("fits");
        assert!(budget.try_reserve(41).is_none());
        let b = budget.try_reserve(40).expect("exactly fills");
        assert_eq!(budget.used(), 100);
        drop(a);
        assert_eq!(budget.used(), 40);
        drop(b);
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn clamp_only_lowers() {
        let budget = Budget::new(100);
        assert_eq!(budget.clamp_limit(150), 100);
        assert_eq!(budget.clamp_limit(70), 70);
        assert_eq!(budget.limit(), 70);
    }

    #[test]
    fn oversized_request_does_not_overflow() {
        let budget = Budget::new(u64::MAX);
        let _a = budget.try_reserve(u64::MAX - 1).expect("fits");
        assert!(budget.try_reserve(2).is_none());
    }

    #[test]
    fn lowering_limit_below_usage_blocks_new_reservations() {
        let budget = Budget::new(100);
        let _a = budget.try_reserve(80).expect("fits");
        budget.set_limit(50);
        assert!(budget.try_reserve(1).is_none());
    }
}
