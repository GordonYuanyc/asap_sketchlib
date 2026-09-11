//! Property tests for the ASAPv1 envelope under adversarial input.
//!
//! One law, stated once and applied to every type: an encode either refuses
//! the state outright or produces bytes that decode to the same observable
//! answers and re-encode byte for byte. A refusal is allowed because several
//! payloads cannot represent every state their constructor accepts; silent
//! corruption is not.
//!
//! What the generators aim at is the input the fixed streams in
//! `tests/e2e_wire.rs` never reach: empty sketches, NaN and the infinities,
//! the f64 extremes, and geometries at one, at a prime and at a power of two.

use asap_sketchlib::{
    Bloom, Count, CountMin, DDSketch, DataInput, DefaultXxHasher, ErtlMLE, FastPath, HyperLogLog,
    KLL, KLLDynamic, RegularPath, SpaceSaving, Vector2D,
};
use proptest::prelude::*;

/// Encode, decode, and hold the decoded sketch to the original's answers and
/// to a byte-identical re-encode. A refused encode passes.
macro_rules! round_trip {
    ($ty:ty, $sketch:expr, $($probe:expr),+ $(,)?) => {{
        let original = &$sketch;
        if let Ok(bytes) = original.serialize_to_bytes() {
            let decoded = <$ty>::deserialize_from_bytes(&bytes)
                .expect("bytes the encoder produced must decode");
            $(
                prop_assert_eq!($probe(&decoded), $probe(original));
            )+
            prop_assert_eq!(
                decoded.serialize_to_bytes().expect("a decoded sketch must re-encode"),
                bytes,
                "the re-encode is not byte-identical"
            );
        }
    }};
}

fn extreme_f64() -> impl Strategy<Value = f64> {
    prop_oneof![
        6 => any::<f64>(),
        1 => Just(f64::NAN),
        1 => Just(f64::INFINITY),
        1 => Just(f64::NEG_INFINITY),
        1 => Just(0.0f64),
        1 => Just(-0.0f64),
        1 => Just(f64::MIN_POSITIVE),
        1 => Just(f64::MAX),
        1 => Just(f64::MIN),
        1 => Just(f64::EPSILON),
    ]
}

fn extreme_values(max: usize) -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec(extreme_f64(), 0..max)
}

/// One, a prime, and a power of two, at both ends of the useful range.
fn edge_dimension() -> impl Strategy<Value = usize> {
    prop_oneof![
        Just(1usize),
        Just(2),
        Just(3),
        Just(7),
        Just(8),
        Just(64),
        Just(127),
        Just(128),
    ]
}

/// Slice counts a Bloom filter accepts: `with_dimensions` panics above
/// `BLOOM_MAX_SLICES`, so the geometry is refused at construction rather than
/// at serialization.
fn bloom_rows() -> impl Strategy<Value = usize> {
    prop_oneof![
        Just(1usize),
        Just(2),
        Just(3),
        Just(7),
        Just(8),
        Just(16),
        Just(20),
    ]
}

fn keys(max: usize) -> impl Strategy<Value = Vec<u64>> {
    prop::collection::vec(0u64..512, 0..max)
}

const QUANTILES: [f64; 5] = [0.0, 0.1, 0.5, 0.9, 1.0];
const PROBE_KEYS: usize = 64;

fn bloom_bits(f: &Bloom) -> Vec<bool> {
    let m = f.as_bits();
    (0..m.rows())
        .flat_map(|r| (0..m.cols()).map(move |c| (r, c)))
        .map(|(r, c)| m.get(r, c))
        .collect()
}

proptest! {
    // ===== Quantile payloads against NaN, the infinities and the f64 edges =====

    #[test]
    fn ddsketch_round_trips_under_extreme_values(
        alpha in prop_oneof![
            Just(0.001f64), Just(0.01), Just(0.1), Just(0.5), Just(0.9), Just(0.999)
        ],
        values in extreme_values(200),
    ) {
        let mut sketch = DDSketch::new(alpha);
        for v in &values {
            sketch.add(v);
        }

        round_trip!(
            DDSketch,
            sketch,
            |s: &DDSketch| s.get_count(),
            |s: &DDSketch| s.alpha().to_bits(),
            |s: &DDSketch| QUANTILES.iter()
                .map(|q| s.get_value_at_quantile(*q).map(f64::to_bits))
                .collect::<Vec<_>>(),
        );
    }

    #[test]
    fn kll_round_trips_under_extreme_values(
        k in prop_oneof![Just(1i32), Just(8), Just(64), Just(200), Just(1_024)],
        values in extreme_values(300),
    ) {
        let mut sketch: KLL<f64> = KLL::init_kll_with_seed(k, 0x5EED_0001);
        for v in &values {
            sketch.update(v);
        }

        round_trip!(
            KLL<f64>,
            sketch,
            |s: &KLL<f64>| s.count(),
            |s: &KLL<f64>| QUANTILES.iter().map(|q| s.quantile(*q).to_bits()).collect::<Vec<_>>(),
        );
    }

    #[test]
    fn kll_dynamic_round_trips_under_extreme_values(
        k in prop_oneof![Just(1i32), Just(8), Just(64), Just(200), Just(1_024)],
        values in extreme_values(300),
    ) {
        let mut sketch = KLLDynamic::<f64>::init_kll_with_seed(k, 0x5EED_0002);
        for v in &values {
            sketch.update(v);
        }

        round_trip!(
            KLLDynamic<f64>,
            sketch,
            |s: &KLLDynamic<f64>| s.count(),
            |s: &KLLDynamic<f64>| QUANTILES.iter().map(|q| s.quantile(*q).to_bits()).collect::<Vec<_>>(),
        );
    }

    // ===== Matrix payloads at edge geometries, empty streams included =====

    #[test]
    fn count_min_round_trips_at_edge_geometries(
        rows in edge_dimension(),
        cols in edge_dimension(),
        stream in keys(200),
    ) {
        type Cm = CountMin<Vector2D<i64>, FastPath>;
        let mut sketch = Cm::with_dimensions(rows, cols);
        for k in &stream {
            sketch.insert(&DataInput::U64(*k));
        }

        round_trip!(
            Cm,
            sketch,
            |s: &Cm| (0..PROBE_KEYS as u64)
                .map(|k| s.estimate(&DataInput::U64(k)))
                .collect::<Vec<_>>(),
        );
    }

    #[test]
    fn count_sketch_round_trips_at_edge_geometries(
        rows in edge_dimension(),
        cols in edge_dimension(),
        stream in keys(200),
    ) {
        type Cs = Count<Vector2D<i64>, RegularPath>;
        let mut sketch = Cs::with_dimensions(rows, cols);
        for k in &stream {
            sketch.insert(&DataInput::U64(*k));
        }

        round_trip!(
            Cs,
            sketch,
            |s: &Cs| (0..PROBE_KEYS as u64)
                .map(|k| s.estimate(&DataInput::U64(k)))
                .collect::<Vec<_>>(),
        );
    }

    #[test]
    fn bloom_round_trips_or_refuses_the_geometry(
        rows in bloom_rows(),
        cols in edge_dimension(),
        stream in keys(200),
    ) {
        let mut sketch: Bloom = Bloom::with_dimensions(rows, cols);
        for k in &stream {
            sketch.insert(&DataInput::U64(*k));
        }

        round_trip!(Bloom, sketch, bloom_bits);
    }

    // ===== Register and counter payloads, empty streams included =====

    #[test]
    fn hyperloglog_round_trips_including_the_empty_sketch(
        stream in keys(400),
    ) {
        type Hll = HyperLogLog<ErtlMLE>;
        let mut sketch = Hll::new();
        for k in &stream {
            sketch.insert(&DataInput::U64(*k));
        }

        round_trip!(
            Hll,
            sketch,
            |s: &Hll| s.registers_as_slice().to_vec(),
            |s: &Hll| s.estimate(),
        );
    }

    #[test]
    fn space_saving_round_trips_at_edge_capacities(
        capacity in edge_dimension(),
        stream in keys(400),
    ) {
        type Ss = SpaceSaving<DefaultXxHasher>;
        let mut sketch = Ss::with_capacity(capacity);
        for k in &stream {
            sketch.insert(&DataInput::U64(*k));
        }

        round_trip!(
            Ss,
            sketch,
            |s: &Ss| s.total(),
            |s: &Ss| s.min_count(),
            |s: &Ss| (0..PROBE_KEYS as u64)
                .map(|k| (s.estimate(&DataInput::U64(k)), s.error(&DataInput::U64(k))))
                .collect::<Vec<_>>(),
        );
    }
}
