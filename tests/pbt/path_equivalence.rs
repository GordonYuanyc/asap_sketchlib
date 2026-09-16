//! Property tests for the paths that must agree with one another.
//!
//! Both sides of every law here are real implementations, so none of it rests
//! on a model of the algorithm. Stream order must not reach the state of a
//! sketch whose updates commute, a weighted arrival must match repeating it,
//! the pre-hashed entry points must address the same cells as the value ones,
//! and a top-k wrapper must leave its inner matrix exactly as the bare sketch
//! would.

use asap_sketchlib::{
    Bloom, CMSHeap, CSHeap, Classic, Count, CountMin, DataInput, FastPath, HyperLogLog,
    RegularPath, Vector2D, hash_for_matrix,
};
use proptest::prelude::*;

type Cm = CountMin<Vector2D<i32>, RegularPath>;
type CmFast = CountMin<Vector2D<i32>, FastPath>;
type Cs = Count<Vector2D<i32>, RegularPath>;
type CmHeap = CMSHeap<Vector2D<i32>, RegularPath>;
type CsHeap = CSHeap<Vector2D<i32>, RegularPath>;

fn stream(max: usize) -> impl Strategy<Value = Vec<u64>> {
    prop::collection::vec(0u64..512, 0..max)
}

fn stream_and_permutation(max: usize) -> impl Strategy<Value = (Vec<u64>, Vec<u64>)> {
    prop::collection::vec(0u64..512, 0..max)
        .prop_flat_map(|v| (Just(v.clone()), Just(v).prop_shuffle()))
}

fn weighted(max: usize) -> impl Strategy<Value = Vec<(u64, i32)>> {
    prop::collection::vec((0u64..512, 1i32..8), 0..max)
}

fn cm_cells(sketch: &Cm) -> Vec<i32> {
    let (rows, cols) = (sketch.rows(), sketch.cols());
    (0..rows)
        .flat_map(|r| (0..cols).map(move |c| (r, c)))
        .map(|(r, c)| sketch.as_storage().query_one_counter(r, c))
        .collect()
}

fn cm_fast_cells(sketch: &CmFast) -> Vec<i32> {
    let (rows, cols) = (sketch.rows(), sketch.cols());
    (0..rows)
        .flat_map(|r| (0..cols).map(move |c| (r, c)))
        .map(|(r, c)| sketch.as_storage().query_one_counter(r, c))
        .collect()
}

fn cs_cells(sketch: &Cs) -> Vec<i32> {
    let (rows, cols) = (sketch.rows(), sketch.cols());
    (0..rows)
        .flat_map(|r| (0..cols).map(move |c| (r, c)))
        .map(|(r, c)| sketch.as_storage().query_one_counter(r, c))
        .collect()
}

fn bloom_bits(f: &Bloom) -> Vec<bool> {
    let m = f.as_bits();
    (0..m.rows())
        .flat_map(|r| (0..m.cols()).map(move |c| (r, c)))
        .map(|(r, c)| m.get(r, c))
        .collect()
}

proptest! {
    // ===== Stream order must not reach the state =====

    #[test]
    fn count_min_is_insensitive_to_stream_order(
        rows in 1usize..6,
        cols in 1usize..64,
        (a, b) in stream_and_permutation(300),
    ) {
        let mut first = Cm::with_dimensions(rows, cols);
        let mut second = Cm::with_dimensions(rows, cols);
        for k in &a {
            first.insert(&DataInput::U64(*k));
        }
        for k in &b {
            second.insert(&DataInput::U64(*k));
        }

        prop_assert_eq!(cm_cells(&first), cm_cells(&second));
    }

    #[test]
    fn count_sketch_is_insensitive_to_stream_order(
        rows in 1usize..6,
        cols in 1usize..64,
        (a, b) in stream_and_permutation(300),
    ) {
        let mut first = Cs::with_dimensions(rows, cols);
        let mut second = Cs::with_dimensions(rows, cols);
        for k in &a {
            first.insert(&DataInput::U64(*k));
        }
        for k in &b {
            second.insert(&DataInput::U64(*k));
        }

        prop_assert_eq!(cs_cells(&first), cs_cells(&second));
    }

    #[test]
    fn bloom_is_insensitive_to_stream_order(
        rows in 1usize..6,
        cols in 8usize..128,
        (a, b) in stream_and_permutation(200),
    ) {
        let mut first: Bloom = Bloom::with_dimensions(rows, cols);
        let mut second: Bloom = Bloom::with_dimensions(rows, cols);
        for k in &a {
            first.insert(&DataInput::U64(*k));
        }
        for k in &b {
            second.insert(&DataInput::U64(*k));
        }

        prop_assert_eq!(bloom_bits(&first), bloom_bits(&second));
    }

    #[test]
    fn hyperloglog_is_insensitive_to_stream_order(
        (a, b) in stream_and_permutation(400),
    ) {
        let mut first = HyperLogLog::<Classic>::default();
        let mut second = HyperLogLog::<Classic>::default();
        for k in &a {
            first.insert(&DataInput::U64(*k));
        }
        for k in &b {
            second.insert(&DataInput::U64(*k));
        }

        prop_assert_eq!(first.registers_as_slice(), second.registers_as_slice());
    }

    // ===== A weighted arrival must match repeating it =====

    #[test]
    fn count_min_weighted_matches_repeating_the_insert(
        rows in 1usize..6,
        cols in 1usize..64,
        updates in weighted(120),
    ) {
        let mut by_weight = Cm::with_dimensions(rows, cols);
        let mut by_repeat = Cm::with_dimensions(rows, cols);
        for (k, n) in &updates {
            by_weight.insert_many(&DataInput::U64(*k), *n);
            for _ in 0..*n {
                by_repeat.insert(&DataInput::U64(*k));
            }
        }

        prop_assert_eq!(cm_cells(&by_weight), cm_cells(&by_repeat));
    }

    #[test]
    fn count_sketch_weighted_matches_repeating_the_insert(
        rows in 1usize..6,
        cols in 1usize..64,
        updates in weighted(120),
    ) {
        let mut by_weight = Cs::with_dimensions(rows, cols);
        let mut by_repeat = Cs::with_dimensions(rows, cols);
        for (k, n) in &updates {
            by_weight.insert_many(&DataInput::U64(*k), *n);
            for _ in 0..*n {
                by_repeat.insert(&DataInput::U64(*k));
            }
        }

        prop_assert_eq!(cs_cells(&by_weight), cs_cells(&by_repeat));
    }

    // ===== Pre-hashed entry points must address the same cells =====

    #[test]
    fn count_min_weighted_entry_points_agree(
        rows in 1usize..6,
        cols in 1usize..64,
        updates in weighted(120),
    ) {
        let pairs: Vec<(DataInput, i32)> = updates
            .iter()
            .map(|(k, n)| (DataInput::U64(*k), *n))
            .collect();
        let hashed: Vec<_> = updates
            .iter()
            .map(|(k, n)| (hash_for_matrix(rows, cols, &DataInput::U64(*k)), *n))
            .collect();

        let mut by_loop = CmFast::with_dimensions(rows, cols);
        for (value, n) in &pairs {
            by_loop.insert_many(value, *n);
        }
        let mut by_bulk = CmFast::with_dimensions(rows, cols);
        by_bulk.bulk_insert_many(&pairs);
        let mut by_hashes = CmFast::with_dimensions(rows, cols);
        by_hashes.bulk_insert_many_with_hashes(&hashed);

        let expected = cm_fast_cells(&by_loop);
        prop_assert_eq!(cm_fast_cells(&by_bulk), expected.clone());
        prop_assert_eq!(cm_fast_cells(&by_hashes), expected);
    }

    #[test]
    fn hyperloglog_hashed_entry_point_matches_the_value_one(
        keys in stream(400),
    ) {
        let mut by_value = HyperLogLog::<Classic>::default();
        for k in &keys {
            by_value.insert(&DataInput::U64(*k));
        }

        let hashes: Vec<u64> = keys
            .iter()
            .map(|k| HyperLogLog::<Classic>::canonical_hash(&DataInput::U64(*k)))
            .collect();
        let mut by_hash = HyperLogLog::<Classic>::default();
        by_hash.insert_many_with_hashes(&hashes);

        prop_assert_eq!(by_value.registers_as_slice(), by_hash.registers_as_slice());
    }

    // ===== A top-k wrapper must not disturb its inner matrix =====

    #[test]
    fn cms_heap_leaves_the_matrix_a_bare_count_min_would_build(
        rows in 1usize..6,
        cols in 1usize..64,
        top_k in 1usize..16,
        keys in stream(300),
    ) {
        let mut wrapped = CmHeap::new(rows, cols, top_k);
        let mut bare = Cm::with_dimensions(rows, cols);
        for k in &keys {
            wrapped.insert(&DataInput::U64(*k));
            bare.insert(&DataInput::U64(*k));
        }

        prop_assert_eq!(cm_cells(wrapped.cms()), cm_cells(&bare));
    }

    #[test]
    fn cs_heap_leaves_the_matrix_a_bare_count_sketch_would_build(
        rows in 1usize..6,
        cols in 1usize..64,
        top_k in 1usize..16,
        keys in stream(300),
    ) {
        let mut wrapped = CsHeap::new(rows, cols, top_k);
        let mut bare = Cs::with_dimensions(rows, cols);
        for k in &keys {
            wrapped.insert(&DataInput::U64(*k));
            bare.insert(&DataInput::U64(*k));
        }

        prop_assert_eq!(cs_cells(wrapped.cs()), cs_cells(&bare));
    }
}
