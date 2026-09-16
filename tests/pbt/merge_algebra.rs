//! Property tests for the merge algebra of the mergeable sketches.
//!
//! Only laws that hold deterministically are asserted as equalities.
//! Where f64 counters or randomized compaction make exactness
//! unavailable, the law is stated as a bound instead.

use asap_sketchlib::{
    Bloom, CountMinSketch, CountSketch, DataInput, DdSketch, HllSketch, HllVariant, KllSketch,
};
use proptest::prelude::*;

/// Relative tolerance absorbing f64 summation-order differences
/// between a merge and the equivalent update stream.
const REL_TOL: f64 = 1e-12;

fn close(a: f64, b: f64) -> bool {
    let scale = a.abs().max(b.abs()).max(1.0);
    (a - b).abs() <= REL_TOL * scale
}

fn matrices_close(a: &[Vec<f64>], b: &[Vec<f64>]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(ra, rb)| ra.len() == rb.len() && ra.iter().zip(rb).all(|(x, y)| close(*x, *y)))
}

fn keyed_updates(max: usize) -> impl Strategy<Value = Vec<(String, f64)>> {
    prop::collection::vec(("[a-z]{1,8}", 1.0f64..10_000.0), 0..max)
}

fn byte_items(max: usize) -> impl Strategy<Value = Vec<Vec<u8>>> {
    prop::collection::vec(prop::collection::vec(any::<u8>(), 1..16), 0..max)
}

fn cms_of(rows: usize, cols: usize, updates: &[(String, f64)]) -> CountMinSketch {
    let mut s = CountMinSketch::new(rows, cols);
    for (k, v) in updates {
        s.update(k, *v);
    }
    s
}

fn cs_of(rows: usize, cols: usize, updates: &[(String, f64)]) -> CountSketch {
    let mut s = CountSketch::new(rows, cols);
    for (k, v) in updates {
        s.update(k, *v);
    }
    s
}

fn hll_of(precision: u32, items: &[Vec<u8>]) -> HllSketch {
    let mut s = HllSketch::new(HllVariant::Regular, precision);
    for i in items {
        s.update(i);
    }
    s
}

fn dd_of(alpha: f64, values: &[f64]) -> DdSketch {
    let mut s = DdSketch::new(alpha);
    for v in values {
        s.update(*v);
    }
    s
}

fn kll_of(k: u16, values: &[f64]) -> KllSketch {
    let mut s = KllSketch::with_seed(k, 0xC0FFEE);
    for v in values {
        s.update(*v);
    }
    s
}

fn bits_of(f: &Bloom) -> Vec<bool> {
    let m = f.as_bits();
    (0..m.rows())
        .flat_map(|r| (0..m.cols()).map(move |c| (r, c)))
        .map(|(r, c)| m.get(r, c))
        .collect()
}

fn bloom_of(rows: usize, cols: usize, values: &[u64]) -> Bloom {
    let mut f: Bloom = Bloom::with_dimensions(rows, cols);
    for v in values {
        f.insert(&DataInput::U64(*v));
    }
    f
}

proptest! {
    // ===== Count-Min =====

    #[test]
    fn count_min_merge_is_commutative(
        rows in 1usize..6,
        cols in 4usize..64,
        a in keyed_updates(30),
        b in keyed_updates(30),
    ) {
        let mut ab = cms_of(rows, cols, &a);
        ab.merge(&cms_of(rows, cols, &b)).expect("merge");
        let mut ba = cms_of(rows, cols, &b);
        ba.merge(&cms_of(rows, cols, &a)).expect("merge");

        prop_assert_eq!(ab.sketch(), ba.sketch());
    }

    #[test]
    fn count_min_merge_is_associative(
        rows in 1usize..6,
        cols in 4usize..64,
        a in keyed_updates(20),
        b in keyed_updates(20),
        c in keyed_updates(20),
    ) {
        let mut left = cms_of(rows, cols, &a);
        left.merge(&cms_of(rows, cols, &b)).expect("merge");
        left.merge(&cms_of(rows, cols, &c)).expect("merge");

        let mut right = cms_of(rows, cols, &b);
        right.merge(&cms_of(rows, cols, &c)).expect("merge");
        let mut right_full = cms_of(rows, cols, &a);
        right_full.merge(&right).expect("merge");

        prop_assert!(matrices_close(&left.sketch(), &right_full.sketch()));
    }

    #[test]
    fn count_min_empty_is_a_merge_identity(
        rows in 1usize..6,
        cols in 4usize..64,
        a in keyed_updates(30),
    ) {
        let base = cms_of(rows, cols, &a);
        let mut merged = base.clone();
        merged.merge(&CountMinSketch::new(rows, cols)).expect("merge");

        prop_assert_eq!(merged.sketch(), base.sketch());
    }

    #[test]
    fn count_min_merge_matches_streaming_the_concatenation(
        rows in 1usize..6,
        cols in 4usize..64,
        a in keyed_updates(30),
        b in keyed_updates(30),
    ) {
        let mut merged = cms_of(rows, cols, &a);
        merged.merge(&cms_of(rows, cols, &b)).expect("merge");

        let concatenated: Vec<_> = a.iter().chain(b.iter()).cloned().collect();
        let streamed = cms_of(rows, cols, &concatenated);

        prop_assert!(matrices_close(&merged.sketch(), &streamed.sketch()));
    }

    #[test]
    fn count_min_never_underestimates(
        rows in 1usize..6,
        cols in 4usize..64,
        updates in keyed_updates(40),
    ) {
        let s = cms_of(rows, cols, &updates);
        for (k, _) in &updates {
            let truth: f64 = updates.iter().filter(|(x, _)| x == k).map(|(_, v)| v).sum();
            let est = s.estimate(k);
            prop_assert!(
                est >= truth || close(est, truth),
                "key {}: estimate {} < truth {}", k, est, truth
            );
        }
    }

    // ===== Count Sketch =====

    #[test]
    fn count_sketch_merge_matches_streaming_the_concatenation(
        rows in 1usize..6,
        cols in 4usize..64,
        a in keyed_updates(30),
        b in keyed_updates(30),
    ) {
        let mut merged = cs_of(rows, cols, &a);
        merged.merge(&cs_of(rows, cols, &b)).expect("merge");

        let concatenated: Vec<_> = a.iter().chain(b.iter()).cloned().collect();
        let streamed = cs_of(rows, cols, &concatenated);

        prop_assert!(matrices_close(merged.sketch(), streamed.sketch()));
    }

    // ===== HyperLogLog =====

    #[test]
    fn hll_register_merge_is_commutative_and_idempotent(
        precision in 4u32..12,
        a in byte_items(50),
        b in byte_items(50),
    ) {
        let mut ab = hll_of(precision, &a);
        ab.merge(&hll_of(precision, &b)).expect("merge");
        let mut ba = hll_of(precision, &b);
        ba.merge(&hll_of(precision, &a)).expect("merge");

        prop_assert_eq!(&ab.registers, &ba.registers);

        let before = ab.registers.clone();
        let again = ab.clone();
        ab.merge(&again).expect("merge");
        prop_assert_eq!(&ab.registers, &before);
    }

    #[test]
    fn hll_merge_equals_streaming_the_concatenation(
        precision in 4u32..12,
        a in byte_items(50),
        b in byte_items(50),
    ) {
        let mut merged = hll_of(precision, &a);
        merged.merge(&hll_of(precision, &b)).expect("merge");

        let concatenated: Vec<_> = a.iter().chain(b.iter()).cloned().collect();
        let streamed = hll_of(precision, &concatenated);

        prop_assert_eq!(&merged.registers, &streamed.registers);
    }

    #[test]
    fn hll_empty_is_a_merge_identity(
        precision in 4u32..12,
        a in byte_items(50),
    ) {
        let base = hll_of(precision, &a);
        let mut merged = base.clone();
        merged.merge(&HllSketch::new(HllVariant::Regular, precision)).expect("merge");

        prop_assert_eq!(&merged.registers, &base.registers);
    }

    // ===== DDSketch =====

    #[test]
    fn ddsketch_merge_is_count_additive_and_commutative(
        alpha in 0.005f64..0.1,
        a in prop::collection::vec(1e-3f64..1e5, 0..80),
        b in prop::collection::vec(1e-3f64..1e5, 0..80),
    ) {
        let mut ab = dd_of(alpha, &a);
        ab.merge(&dd_of(alpha, &b)).expect("merge");
        let mut ba = dd_of(alpha, &b);
        ba.merge(&dd_of(alpha, &a)).expect("merge");

        prop_assert_eq!(ab.total_count(), (a.len() + b.len()) as u64);
        prop_assert_eq!(ab.total_count(), ba.total_count());
        prop_assert_eq!(&ab.store_counts, &ba.store_counts);
        prop_assert_eq!(ab.store_offset, ba.store_offset);
    }

    // ===== KLL =====

    #[test]
    fn kll_count_is_exact_below_the_compaction_threshold(
        k in 8u16..256,
        values in prop::collection::vec(-1e5f64..1e5, 0..8),
    ) {
        let s = kll_of(k, &values);
        prop_assert_eq!(s.count(), values.len() as u64);
    }

    #[test]
    fn kll_merged_count_stays_within_a_relative_band(
        k in 32u16..256,
        a in prop::collection::vec(-1e5f64..1e5, 0..400),
        b in prop::collection::vec(-1e5f64..1e5, 0..400),
    ) {
        let mut merged = kll_of(k, &a);
        merged.merge(&kll_of(k, &b)).expect("merge");

        let n = (a.len() + b.len()) as f64;
        let got = merged.count() as f64;
        let slack = (0.10 * n).max(4.0);
        prop_assert!(
            (got - n).abs() <= slack,
            "k={} n={} count()={} outside +/-{}", k, n, got, slack
        );
    }

    #[test]
    fn kll_quantiles_are_monotone_in_q(
        k in 32u16..256,
        values in prop::collection::vec(-1e5f64..1e5, 1..400),
    ) {
        let s = kll_of(k, &values);
        let mut prev = f64::NEG_INFINITY;
        for i in 0..=20 {
            let q = i as f64 / 20.0;
            let v = s.quantile(q);
            prop_assert!(v >= prev, "q={} gave {} after {}", q, v, prev);
            prev = v;
        }
    }

    #[test]
    fn kll_quantile_range_is_inside_the_observed_range(
        k in 32u16..256,
        values in prop::collection::vec(-1e5f64..1e5, 1..400),
    ) {
        let s = kll_of(k, &values);
        let lo = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        prop_assert!(s.quantile(0.0) >= lo, "min {} below observed {}", s.quantile(0.0), lo);
        prop_assert!(s.quantile(1.0) <= hi, "max {} above observed {}", s.quantile(1.0), hi);
    }

    // ===== Bloom =====

    #[test]
    fn bloom_has_no_false_negatives_before_or_after_merge(
        rows in 1usize..6,
        cols in 64usize..1024,
        a in prop::collection::vec(any::<u64>(), 0..40),
        b in prop::collection::vec(any::<u64>(), 0..40),
    ) {
        let mut fa = bloom_of(rows, cols, &a);
        for v in &a {
            prop_assert!(fa.contains(&DataInput::U64(*v)), "false negative on {}", v);
        }

        fa.merge(&bloom_of(rows, cols, &b));
        for v in a.iter().chain(b.iter()) {
            prop_assert!(
                fa.contains(&DataInput::U64(*v)),
                "false negative after merge on {}", v
            );
        }
    }

    #[test]
    fn bloom_merge_is_commutative_and_idempotent(
        rows in 1usize..6,
        cols in 64usize..1024,
        a in prop::collection::vec(any::<u64>(), 0..40),
        b in prop::collection::vec(any::<u64>(), 0..40),
    ) {
        let mut ab = bloom_of(rows, cols, &a);
        ab.merge(&bloom_of(rows, cols, &b));
        let mut ba = bloom_of(rows, cols, &b);
        ba.merge(&bloom_of(rows, cols, &a));

        prop_assert_eq!(bits_of(&ab), bits_of(&ba));

        let before = bits_of(&ab);
        let again = ab.clone();
        ab.merge(&again);
        prop_assert_eq!(bits_of(&ab), before);
    }
}
