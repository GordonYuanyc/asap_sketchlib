//! Property tests against models taken from the algorithms' own papers.
//!
//! The models here are derived from the published definitions, not from the
//! implementations they check, so a misreading in the code cannot be copied
//! into the oracle. Where a paper leaves a choice free - which bits of a hash
//! index a register, how a bucket maps to a rank - the model states only what
//! the paper fixes, because an implementation is entitled to the other option.
//!
//! `estimate` is a deterministic function of the registers, so no model of it
//! can be independent of the code it checks. Its laws here are relations
//! between estimates instead: a permutation, a uniform shift, a single rise.
//!
//! * HyperLogLog: Flajolet, Fusy, Gandouet, Meunier, AofA '07.
//! * DDSketch: Masson, Rim, Lee, VLDB '19.
//! * Bloom: Bloom, CACM '70.

use asap_sketchlib::sketches::hll::HyperLogLogImpl;
use asap_sketchlib::{Bloom, Classic, DDSketch, DataInput, HyperLogLog};
use proptest::prelude::*;

// 256 registers, small enough that a few hundred keys straddle the 5m/2 the
// small range correction switches at. The default P14 puts that boundary past
// forty thousand distinct keys.
asap_sketchlib::impl_hll_bucket_list!(HllBucketListP8, 8, 1_usize << 8);
type HllP8 = HyperLogLogImpl<Classic, HllBucketListP8>;

/// Registers in the default sketch, well above the 128 at which the paper's
/// alpha formula takes over from its three tabulated constants.
fn hll_of(keys: &[u64]) -> HyperLogLog<Classic> {
    let mut s = HyperLogLog::<Classic>::new();
    for k in keys {
        s.insert(&DataInput::U64(*k));
    }
    s
}

/// Ranks stay in this band so the sum of `2^-r` over 256 registers is exact in
/// f64: a 40-bit span plus 8 bits of count sits inside the 53-bit significand,
/// so the order the terms are added in cannot reach the result.
const MAX_RANK: u8 = 40;

/// The hash that drives `bucket` to `rank`: the top `PRECISION` bits select the
/// register, the highest set bit below them fixes the rank.
fn hash_placing(bucket: usize, rank: u8) -> u64 {
    let payload_bits = 64 - HllBucketListP8::PRECISION as u32;
    ((bucket as u64) << payload_bits) | (1u64 << (payload_bits - rank as u32))
}

/// A sketch whose registers are exactly `target`, built through the public
/// insertion path. The assertion holds the hash split in place: a different one
/// fails here rather than handing the laws below some other state.
fn sketch_with_registers(target: &[u8]) -> HllP8 {
    let mut sketch = HllP8::new();
    for (bucket, &rank) in target.iter().enumerate() {
        if rank != 0 {
            sketch.insert_with_hash(hash_placing(bucket, rank));
        }
    }
    assert_eq!(sketch.registers_as_slice(), target, "crafted registers");
    sketch
}

/// Register arrays with no zero entry: the estimator's raw branch, the one the
/// small range correction cannot fire in.
fn dense_registers() -> impl Strategy<Value = Vec<u8>> {
    prop::collection::vec(1u8..=MAX_RANK, HllBucketListP8::NUM_REGISTERS)
}

fn registers_and_permutation() -> impl Strategy<Value = (Vec<u8>, Vec<u8>)> {
    prop::collection::vec(0u8..=MAX_RANK, HllBucketListP8::NUM_REGISTERS)
        .prop_flat_map(|v| (Just(v.clone()), Just(v).prop_shuffle()))
}

// Zero registers put the estimator in linear counting, where `m * ln(m/m)` is
// zero at every precision.
#[test]
fn hll_the_empty_sketch_estimates_zero() {
    assert_eq!(HllP8::new().estimate(), 0);
    assert_eq!(HyperLogLog::<Classic>::new().estimate(), 0);
}

fn keys(max: usize) -> impl Strategy<Value = Vec<u64>> {
    prop::collection::vec(any::<u64>(), 0..max)
}

/// Values inside the band DDSketch indexes, so nothing is dropped before the
/// guarantee can apply.
fn positive_values(max: usize) -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec(1e-6f64..1e9, 1..max)
}

fn alpha() -> impl Strategy<Value = f64> {
    prop_oneof![
        Just(0.001f64),
        Just(0.005),
        Just(0.01),
        Just(0.05),
        Just(0.1),
        Just(0.2),
    ]
}

fn dd_of(alpha: f64, values: &[f64]) -> DDSketch {
    let mut s = DDSketch::new(alpha);
    for v in values {
        s.add(v);
    }
    s
}

fn sorted(values: &[f64]) -> Vec<f64> {
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).expect("the generator excludes NaN"));
    v
}

proptest! {
    // ===== HyperLogLog =====

    // The registers reach the estimate as a multiset: what the formula reads is
    // the sum of `2^-M[j]`, never the position a register sits at.
    #[test]
    fn hll_estimate_is_invariant_under_permuting_the_registers(
        (registers, permuted) in registers_and_permutation(),
    ) {
        prop_assert_eq!(
            sketch_with_registers(&permuted).estimate(),
            sketch_with_registers(&registers).estimate(),
        );
    }

    // Raising every register by one halves the sum of `2^-M[j]`, so the raw
    // indicator doubles. `estimate` truncates, which leaves one unit of slack.
    #[test]
    fn hll_raising_every_register_by_one_doubles_the_estimate(
        registers in dense_registers(),
    ) {
        let base = sketch_with_registers(&registers).estimate();
        let raised: Vec<u8> = registers.iter().map(|r| r + 1).collect();
        let doubled = sketch_with_registers(&raised).estimate();

        prop_assert!(
            doubled == 2 * base || doubled == 2 * base + 1,
            "every register raised by one: {} against a base of {}",
            doubled,
            base
        );
    }

    // A higher register contributes less to the sum the estimate divides by.
    #[test]
    fn hll_estimate_never_falls_when_a_register_rises(
        registers in dense_registers(),
        index in 0..HllBucketListP8::NUM_REGISTERS,
        lift in 1u8..8,
    ) {
        let before = sketch_with_registers(&registers).estimate();
        let mut raised = registers.clone();
        raised[index] = (raised[index] + lift).min(MAX_RANK);
        let after = sketch_with_registers(&raised).estimate();

        prop_assert!(
            after >= before,
            "register {} at {}: {} fell from {}",
            index,
            raised[index],
            after,
            before
        );
    }

    #[test]
    fn hll_registers_ignore_duplicates(stream in keys(600)) {
        let mut deduped = stream.clone();
        deduped.sort_unstable();
        deduped.dedup();

        let (full, unique) = (hll_of(&stream), hll_of(&deduped));
        prop_assert_eq!(full.registers_as_slice(), unique.registers_as_slice());
    }

    #[test]
    fn hll_registers_never_decrease(stream in keys(300)) {
        let mut sketch = HyperLogLog::<Classic>::new();
        let mut previous = sketch.registers_as_slice().to_vec();
        for k in &stream {
            sketch.insert(&DataInput::U64(*k));
            let current = sketch.registers_as_slice();
            prop_assert!(
                current.iter().zip(&previous).all(|(now, before)| now >= before),
                "a register fell after inserting {}", k
            );
            previous = current.to_vec();
        }
    }

    #[test]
    fn hll_one_arrival_raises_at_most_one_register(stream in keys(300)) {
        let mut sketch = HyperLogLog::<Classic>::new();
        let mut previous = sketch.registers_as_slice().to_vec();
        for k in &stream {
            sketch.insert(&DataInput::U64(*k));
            let changed = sketch
                .registers_as_slice()
                .iter()
                .zip(&previous)
                .filter(|(now, before)| now != before)
                .count();
            prop_assert!(changed <= 1, "inserting {} moved {} registers", k, changed);
            previous = sketch.registers_as_slice().to_vec();
        }
    }

    // ===== DDSketch =====

    #[test]
    fn ddsketch_records_every_value_in_its_indexable_band(
        alpha in alpha(),
        values in positive_values(300),
    ) {
        let sketch = dd_of(alpha, &values);
        prop_assert_eq!(sketch.get_count() as usize, values.len());
    }

    /// Sharper than the paper's bound: the sketch tracks the extremes outside
    /// the buckets, so a lone value comes back unrounded at every quantile.
    #[test]
    fn ddsketch_returns_a_lone_value_exactly(
        alpha in alpha(),
        value in 1e-6f64..1e9,
        q in 0.0f64..=1.0,
    ) {
        let sketch = dd_of(alpha, &[value]);
        let answer = sketch.get_value_at_quantile(q).expect("a non-empty sketch answers");

        prop_assert_eq!(answer.to_bits(), value.to_bits());
    }

    /// Also sharper than the bound: q=0 and q=1 read the tracked extremes
    /// rather than a bucket, so they are exact and say nothing about alpha.
    #[test]
    fn ddsketch_returns_the_extremes_exactly(
        alpha in alpha(),
        values in positive_values(300),
    ) {
        let sketch = dd_of(alpha, &values);
        let ordered = sorted(&values);

        let smallest = sketch.get_value_at_quantile(0.0).expect("a non-empty sketch answers");
        let largest = sketch.get_value_at_quantile(1.0).expect("a non-empty sketch answers");

        prop_assert_eq!(smallest.to_bits(), ordered[0].to_bits());
        prop_assert_eq!(largest.to_bits(), ordered[ordered.len() - 1].to_bits());
    }

    /// The paper's guarantee proper, on the quantiles that actually read a
    /// bucket: the answer sits within a relative alpha of a recorded value.
    #[test]
    fn ddsketch_interior_quantiles_stay_within_alpha(
        alpha in alpha(),
        values in positive_values(300),
        q in 0.05f64..0.95,
    ) {
        prop_assume!(values.len() >= 8);
        let sketch = dd_of(alpha, &values);
        let answer = sketch.get_value_at_quantile(q).expect("a non-empty sketch answers");

        prop_assert!(
            values.iter().any(|v| (answer - v).abs() <= alpha * v),
            "alpha={} q={}: {} is within alpha of nothing in the stream", alpha, q, answer
        );
    }

    #[test]
    fn ddsketch_answers_within_alpha_of_a_recorded_value(
        alpha in alpha(),
        values in positive_values(300),
        q in 0.0f64..=1.0,
    ) {
        let sketch = dd_of(alpha, &values);
        let answer = sketch.get_value_at_quantile(q).expect("a non-empty sketch answers");

        prop_assert!(
            values.iter().any(|v| (answer - v).abs() <= alpha * v),
            "alpha={} q={}: {} is within alpha of nothing in the stream", alpha, q, answer
        );
    }

    #[test]
    fn ddsketch_answers_are_ordered_by_quantile(
        alpha in alpha(),
        values in positive_values(300),
    ) {
        let sketch = dd_of(alpha, &values);
        let probes = [0.0f64, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0];

        let answers: Vec<f64> = probes
            .iter()
            .map(|q| sketch.get_value_at_quantile(*q).expect("a non-empty sketch answers"))
            .collect();
        for pair in answers.windows(2) {
            prop_assert!(pair[0] <= pair[1], "{} came before {}", pair[0], pair[1]);
        }
    }

    // ===== Bloom =====

    #[test]
    fn bloom_never_reports_a_false_negative(
        rows in 1usize..8,
        cols in 8usize..256,
        stream in keys(200),
    ) {
        let mut filter: Bloom = Bloom::with_dimensions(rows, cols);
        for k in &stream {
            filter.insert(&DataInput::U64(*k));
        }

        for k in &stream {
            prop_assert!(filter.contains(&DataInput::U64(*k)), "key {} was inserted", k);
        }
    }

    #[test]
    fn bloom_membership_only_ever_turns_on(
        rows in 1usize..8,
        cols in 8usize..256,
        stream in keys(150),
        probes in keys(40),
    ) {
        let mut filter: Bloom = Bloom::with_dimensions(rows, cols);
        let mut seen: Vec<bool> = probes
            .iter()
            .map(|p| filter.contains(&DataInput::U64(*p)))
            .collect();

        for k in &stream {
            filter.insert(&DataInput::U64(*k));
            for (i, p) in probes.iter().enumerate() {
                let now = filter.contains(&DataInput::U64(*p));
                prop_assert!(now || !seen[i], "probe {} stopped being a member", p);
                seen[i] = now;
            }
        }
    }
}
