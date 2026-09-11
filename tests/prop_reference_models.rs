//! Property tests against models taken from the algorithms' own papers.
//!
//! The models here are derived from the published definitions, not from the
//! implementations they check, so a misreading in the code cannot be copied
//! into the oracle. Where a paper leaves a choice free - which bits of a hash
//! index a register, how a bucket maps to a rank - the model states only what
//! the paper fixes, because an implementation is entitled to the other option.
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

/// The paper's estimator, §4: alpha_m * m^2 / sum 2^-M[j], with the small
/// range correction m * ln(m/V) when the raw estimate falls below 5m/2 and
/// some register is still zero. Ranks come from a 64-bit hash, which puts the
/// large range correction out of reach.
fn flajolet_estimate(registers: &[u8]) -> usize {
    let m = registers.len() as f64;
    let alpha_m = 0.7213 / (1.0 + 1.079 / m);
    let z: f64 = registers.iter().map(|r| 2f64.powi(-(*r as i32))).sum();
    let mut est = alpha_m * m * m / z;
    if est <= m * 5.0 / 2.0 {
        let zeros = registers.iter().filter(|r| **r == 0).count();
        if zeros != 0 {
            est = m * (m / zeros as f64).ln();
        }
    }
    est as usize
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

    #[test]
    fn hll_estimate_matches_the_papers_formula(stream in keys(2_000)) {
        let sketch = hll_of(&stream);
        prop_assert_eq!(sketch.estimate(), flajolet_estimate(sketch.registers_as_slice()));
    }

    #[test]
    fn hll_estimate_matches_the_papers_formula_across_the_correction_switchover(
        stream in keys(1_500),
    ) {
        let mut sketch = HllP8::new();
        for k in &stream {
            sketch.insert(&DataInput::U64(*k));
        }

        prop_assert_eq!(sketch.estimate(), flajolet_estimate(sketch.registers_as_slice()));
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
