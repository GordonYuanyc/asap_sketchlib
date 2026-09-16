//! Property tests for the Space-Saving error bounds.
//!
//! The oracle is an exact `HashMap` count of the generated stream, independent
//! of the summary's own bookkeeping. Every bound asserted here is
//! deterministic, so a counterexample is a defect rather than an unlucky draw.

use asap_sketchlib::{DataInput, HeapItem, SpaceSaving};
use proptest::prelude::*;
use std::collections::HashMap;

fn keys(max: usize) -> impl Strategy<Value = Vec<i64>> {
    prop::collection::vec(0i64..32, 0..max)
}

/// A capacity paired with two streams over a domain barely wider than it, so
/// both sides monitor the same keys and their merged union fits the capacity.
fn crowded_pair(max: usize) -> impl Strategy<Value = (usize, Vec<i64>, Vec<i64>)> {
    (1usize..12).prop_flat_map(move |capacity| {
        let domain = capacity as i64 + 3;
        (
            Just(capacity),
            prop::collection::vec(0..domain, 0..max),
            prop::collection::vec(0..domain, 0..max),
        )
    })
}

fn weighted_keys(max: usize) -> impl Strategy<Value = Vec<(i64, u64)>> {
    prop::collection::vec((0i64..32, 1u64..1_000), 0..max)
}

fn truth_of(stream: &[i64]) -> HashMap<i64, u64> {
    let mut counts: HashMap<i64, u64> = HashMap::new();
    for k in stream {
        *counts.entry(*k).or_default() += 1;
    }
    counts
}

fn weighted_truth_of(stream: &[(i64, u64)]) -> HashMap<i64, u64> {
    let mut counts: HashMap<i64, u64> = HashMap::new();
    for (k, w) in stream {
        *counts.entry(*k).or_default() += w;
    }
    counts
}

fn summary_of(capacity: usize, stream: &[i64]) -> SpaceSaving {
    let mut s = SpaceSaving::with_capacity(capacity);
    for k in stream {
        s.insert(&DataInput::I64(*k));
    }
    s
}

fn weighted_summary_of(capacity: usize, stream: &[(i64, u64)]) -> SpaceSaving {
    let mut s = SpaceSaving::with_capacity(capacity);
    for (k, w) in stream {
        s.insert_many(&DataInput::I64(*k), *w);
    }
    s
}

fn key_of(item: &HeapItem) -> i64 {
    match item {
        HeapItem::I64(v) => *v,
        other => panic!("unexpected key form {other:?}"),
    }
}

/// The monitored keys as `key -> (count, error)`.
fn monitored(s: &SpaceSaving) -> HashMap<i64, (u64, u64)> {
    s.entries()
        .iter()
        .map(|(item, count, error)| (key_of(item), (*count, *error)))
        .collect()
}

proptest! {
    // ===== Bounds against the exact counts =====

    #[test]
    fn monitored_counts_straddle_the_truth(
        capacity in 1usize..40,
        stream in keys(400),
    ) {
        let s = summary_of(capacity, &stream);
        let truth = truth_of(&stream);

        for (key, (count, error)) in monitored(&s) {
            let t = *truth.get(&key).expect("a monitored key the stream never carried");
            prop_assert!(error <= count, "key {}: error {} above count {}", key, error, count);
            prop_assert!(count >= t, "key {}: count {} below truth {}", key, count, t);
            prop_assert!(
                count - error <= t,
                "key {}: count {} less error {} above truth {}", key, count, error, t
            );
        }
    }

    #[test]
    fn weighted_counts_straddle_the_truth(
        capacity in 1usize..40,
        stream in weighted_keys(200),
    ) {
        let s = weighted_summary_of(capacity, &stream);
        let truth = weighted_truth_of(&stream);

        for (key, (count, error)) in monitored(&s) {
            let t = *truth.get(&key).expect("a monitored key the stream never carried");
            prop_assert!(error <= count, "key {}: error {} above count {}", key, error, count);
            prop_assert!(count >= t, "key {}: count {} below truth {}", key, count, t);
            prop_assert!(
                count - error <= t,
                "key {}: count {} less error {} above truth {}", key, count, error, t
            );
        }
    }

    #[test]
    fn upper_bound_never_falls_below_the_truth(
        capacity in 1usize..40,
        stream in keys(400),
    ) {
        let s = summary_of(capacity, &stream);

        for (key, t) in truth_of(&stream) {
            let bound = s.upper_bound(&DataInput::I64(key));
            prop_assert!(bound >= t, "key {}: ceiling {} below truth {}", key, bound, t);
        }
    }

    #[test]
    fn weighted_upper_bound_never_falls_below_the_truth(
        capacity in 1usize..40,
        stream in weighted_keys(200),
    ) {
        let s = weighted_summary_of(capacity, &stream);

        for (key, t) in weighted_truth_of(&stream) {
            let bound = s.upper_bound(&DataInput::I64(key));
            prop_assert!(bound >= t, "key {}: ceiling {} below truth {}", key, bound, t);
        }
    }

    #[test]
    fn a_key_outside_the_stream_reports_the_unmonitored_ceiling(
        capacity in 1usize..40,
        stream in keys(400),
    ) {
        let s = summary_of(capacity, &stream);
        let absent = DataInput::I64(1_000);

        prop_assert_eq!(s.estimate(&absent), 0);
        prop_assert_eq!(s.upper_bound(&absent), s.min_count());
        prop_assert_eq!(s.error(&absent), s.min_count());
        prop_assert!(!s.is_guaranteed(&absent));
    }

    // ===== Structural invariants =====

    #[test]
    fn the_summary_never_outgrows_its_capacity(
        capacity in 1usize..40,
        stream in keys(400),
    ) {
        let s = summary_of(capacity, &stream);
        let distinct = truth_of(&stream).len();

        prop_assert_eq!(s.capacity(), capacity);
        prop_assert!(s.len() <= s.capacity(), "{} counters over {}", s.len(), s.capacity());
        prop_assert_eq!(s.len(), distinct.min(capacity));
    }

    #[test]
    fn every_error_sits_under_the_unmonitored_ceiling(
        capacity in 1usize..40,
        stream in keys(400),
    ) {
        let s = summary_of(capacity, &stream);
        let ceiling = s.min_count();

        for (key, (_, error)) in monitored(&s) {
            prop_assert!(error <= ceiling, "key {}: error {} above ceiling {}", key, error, ceiling);
        }
    }

    #[test]
    fn unit_counters_sum_to_the_total_weight(
        capacity in 1usize..40,
        stream in keys(400),
    ) {
        let s = summary_of(capacity, &stream);
        let held: u64 = s.entries().iter().map(|(_, count, _)| count).sum();

        prop_assert_eq!(s.total(), stream.len() as u64);
        prop_assert_eq!(held, s.total());
    }

    #[test]
    fn a_full_summary_keeps_its_ceiling_under_the_average(
        capacity in 1usize..40,
        stream in keys(400),
    ) {
        let s = summary_of(capacity, &stream);
        prop_assume!(s.len() == s.capacity());

        let ceiling = u128::from(s.min_count()) * capacity as u128;
        prop_assert!(
            ceiling <= u128::from(s.total()),
            "ceiling {} over {} counters exceeds the total {}",
            s.min_count(), capacity, s.total()
        );
    }

    #[test]
    fn top_k_is_the_ranking_prefix(
        capacity in 1usize..40,
        k in 0usize..48,
        stream in keys(400),
    ) {
        let s = summary_of(capacity, &stream);
        let held = monitored(&s);
        let top = s.top_k(k);

        prop_assert_eq!(top.len(), k.min(s.len()));
        for pair in top.windows(2) {
            prop_assert!(pair[0].1 >= pair[1].1, "{} before {}", pair[0].1, pair[1].1);
        }
        for (item, count, error) in &top {
            let key = key_of(item);
            prop_assert_eq!(held.get(&key), Some(&(*count, *error)));
        }
        if let Some((_, lowest, _)) = top.last() {
            let cut = *lowest;
            prop_assert!(
                held.values().filter(|(c, _)| *c > cut).count() < top.len(),
                "a counter above the cut {} was left out of the ranking", cut
            );
        }
    }

    #[test]
    fn a_guaranteed_key_outranks_every_key_the_summary_dropped(
        capacity in 1usize..40,
        stream in keys(400),
    ) {
        let s = summary_of(capacity, &stream);
        let truth = truth_of(&stream);
        let held = monitored(&s);

        for (key, t) in &truth {
            if !s.is_guaranteed(&DataInput::I64(*key)) {
                continue;
            }
            for (other, other_t) in &truth {
                if held.contains_key(other) {
                    continue;
                }
                prop_assert!(
                    t > other_t,
                    "guaranteed key {} at {} does not outrank dropped key {} at {}",
                    key, t, other, other_t
                );
            }
        }
    }

    // ===== Bounds surviving a merge =====

    #[test]
    fn merged_bounds_hold_over_the_concatenated_stream(
        capacity in 1usize..24,
        other_capacity in 1usize..24,
        left in keys(200),
        right in keys(200),
    ) {
        let mut merged = summary_of(capacity, &left);
        merged.merge(&summary_of(other_capacity, &right));

        let concatenated: Vec<i64> = left.iter().chain(right.iter()).copied().collect();
        let truth = truth_of(&concatenated);
        let held = monitored(&merged);

        prop_assert!(merged.len() <= merged.capacity());
        prop_assert_eq!(merged.total(), concatenated.len() as u64);

        for (key, t) in &truth {
            let bound = merged.upper_bound(&DataInput::I64(*key));
            prop_assert!(bound >= *t, "key {}: ceiling {} below truth {}", key, bound, t);

            if let Some((count, error)) = held.get(key) {
                prop_assert!(count >= t, "key {}: count {} below truth {}", key, count, t);
                prop_assert!(
                    count - error <= *t,
                    "key {}: count {} less error {} above truth {}", key, count, error, t
                );
            }
        }
    }

    #[test]
    fn merged_bounds_hold_when_both_sides_monitor_the_same_keys(
        (capacity, left, right) in crowded_pair(120),
    ) {
        let mut merged = summary_of(capacity, &left);
        merged.merge(&summary_of(capacity, &right));

        let concatenated: Vec<i64> = left.iter().chain(right.iter()).copied().collect();
        let truth = truth_of(&concatenated);
        let held = monitored(&merged);

        for (key, t) in &truth {
            let bound = merged.upper_bound(&DataInput::I64(*key));
            prop_assert!(bound >= *t, "key {}: ceiling {} below truth {}", key, bound, t);

            if let Some((count, error)) = held.get(key) {
                prop_assert!(count >= t, "key {}: count {} below truth {}", key, count, t);
                prop_assert!(
                    count - error <= *t,
                    "key {}: count {} less error {} above truth {}", key, count, error, t
                );
            }
        }
    }

    #[test]
    fn merging_a_summary_with_its_own_shape_stays_bounded(
        capacity in 1usize..12,
        stream in keys(200),
    ) {
        let base = summary_of(capacity, &stream);
        let mut merged = base.clone();
        merged.merge(&base);

        for (key, t) in truth_of(&stream) {
            let doubled = t * 2;
            let bound = merged.upper_bound(&DataInput::I64(key));
            prop_assert!(bound >= doubled, "key {}: ceiling {} below truth {}", key, bound, doubled);
        }
    }

    #[test]
    fn an_empty_summary_leaves_a_merge_target_bounded(
        capacity in 1usize..24,
        stream in keys(200),
    ) {
        let base = summary_of(capacity, &stream);
        let mut merged = base.clone();
        merged.merge(&SpaceSaving::with_capacity(capacity));

        let truth = truth_of(&stream);
        prop_assert_eq!(merged.total(), base.total());
        for (key, t) in &truth {
            let bound = merged.upper_bound(&DataInput::I64(*key));
            prop_assert!(bound >= *t, "key {}: ceiling {} below truth {}", key, bound, t);
        }
    }
}
