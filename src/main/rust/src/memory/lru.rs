//! Recency ordering for eviction.

use std::collections::BTreeMap;

use super::TensorId;

/// Tensors ordered from least to most recently used. Ticks come from one
/// monotonic clock shared by every index in the manager, so each tick is
/// unique and the map doubles as an ordered set. An entry keeps its tick when
/// it moves between tiers, so relative age survives a demotion.
#[derive(Debug, Default)]
pub struct LruIndex {
    order: BTreeMap<u64, TensorId>,
}

impl LruIndex {
    pub fn insert(&mut self, tick: u64, id: TensorId) {
        self.order.insert(tick, id);
    }

    pub fn remove(&mut self, tick: u64) {
        self.order.remove(&tick);
    }

    /// The least recently used tensor for which `eligible` holds.
    pub fn oldest_where(&self, mut eligible: impl FnMut(TensorId) -> bool) -> Option<TensorId> {
        self.order.values().copied().find(|&id| eligible(id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oldest_skips_ineligible_entries() {
        let mut lru = LruIndex::default();
        lru.insert(1, 10);
        lru.insert(2, 20);
        lru.insert(3, 30);
        assert_eq!(lru.oldest_where(|_| true), Some(10));
        assert_eq!(lru.oldest_where(|id| id != 10), Some(20));
        assert_eq!(lru.oldest_where(|_| false), None);
    }

    #[test]
    fn touching_moves_entry_to_most_recent() {
        let mut lru = LruIndex::default();
        lru.insert(1, 10);
        lru.insert(2, 20);
        lru.remove(1);
        lru.insert(3, 10);
        assert_eq!(lru.oldest_where(|_| true), Some(20));
        assert_eq!(lru.oldest_where(|id| id != 20), Some(10));
    }
}
