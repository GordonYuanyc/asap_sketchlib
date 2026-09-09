//! Property tests for the `MessagePackCodec` round-trip contract.
//!
//! Each property asserts that decoding an encoding reproduces the
//! observable state of the original — matrix cells, registers, bucket
//! counts, quantiles — not just the shape fields.

use asap_sketchlib::message_pack_format::MessagePackCodec;
use asap_sketchlib::{CountMinSketch, CountSketch, DdSketch, HllSketch, HllVariant, KllSketch};
use proptest::prelude::*;

const QUANTILE_PROBES: [f64; 7] = [0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0];

fn keyed_updates() -> impl Strategy<Value = Vec<(String, f64)>> {
    prop::collection::vec(("[a-z]{1,8}", 1.0f64..1_000_000.0), 0..40)
}

fn hll_variant() -> impl Strategy<Value = HllVariant> {
    prop_oneof![
        Just(HllVariant::Regular),
        Just(HllVariant::Datafusion),
        Just(HllVariant::Hip),
    ]
}

proptest! {
    #[test]
    fn count_min_round_trip_preserves_every_cell(
        rows in 1usize..8,
        cols in 1usize..128,
        updates in keyed_updates(),
    ) {
        let mut s = CountMinSketch::new(rows, cols);
        for (k, v) in &updates {
            s.update(k, *v);
        }

        let bytes = s.to_msgpack().expect("encode");
        let restored = CountMinSketch::from_msgpack(&bytes).expect("decode");

        prop_assert_eq!(restored.rows(), s.rows());
        prop_assert_eq!(restored.cols(), s.cols());
        prop_assert_eq!(restored.sketch(), s.sketch());
        for (k, _) in &updates {
            prop_assert_eq!(restored.estimate(k), s.estimate(k), "key {}", k);
        }
    }

    #[test]
    fn count_sketch_round_trip_preserves_every_cell(
        rows in 1usize..8,
        cols in 1usize..128,
        updates in keyed_updates(),
    ) {
        let mut s = CountSketch::new(rows, cols);
        for (k, v) in &updates {
            s.update(k, *v);
        }

        let bytes = s.to_msgpack().expect("encode");
        let restored = CountSketch::from_msgpack(&bytes).expect("decode");

        prop_assert_eq!(restored.sketch(), s.sketch());
        for (k, _) in &updates {
            prop_assert_eq!(restored.estimate(k), s.estimate(k), "key {}", k);
        }
    }

    #[test]
    fn hll_round_trip_preserves_registers(
        variant in hll_variant(),
        precision in 4u32..14,
        items in prop::collection::vec(prop::collection::vec(any::<u8>(), 1..16), 0..64),
    ) {
        let mut s = HllSketch::new(variant, precision);
        for item in &items {
            s.update(item);
        }

        let bytes = s.to_msgpack().expect("encode");
        let restored = HllSketch::from_msgpack(&bytes).expect("decode");

        prop_assert_eq!(restored.variant, s.variant);
        prop_assert_eq!(restored.precision, s.precision);
        prop_assert_eq!(&restored.registers, &s.registers);
        prop_assert_eq!(restored.estimate(), s.estimate());
    }

    #[test]
    fn kll_round_trip_preserves_count_and_quantiles(
        k in 8u16..256,
        values in prop::collection::vec(-1e6f64..1e6, 0..300),
    ) {
        let mut s = KllSketch::with_seed(k, 0x5EED);
        for v in &values {
            s.update(*v);
        }

        let bytes = s.to_msgpack().expect("encode");
        let restored = KllSketch::from_msgpack(&bytes).expect("decode");

        prop_assert_eq!(restored.k(), s.k());
        prop_assert_eq!(restored.count(), s.count());
        for q in QUANTILE_PROBES {
            prop_assert_eq!(restored.quantile(q), s.quantile(q), "q={}", q);
        }
    }

    #[test]
    fn ddsketch_round_trip_preserves_buckets_and_quantiles(
        alpha in 0.001f64..0.2,
        values in prop::collection::vec(1e-3f64..1e6, 0..300),
    ) {
        let mut s = DdSketch::new(alpha);
        for v in &values {
            s.update(*v);
        }

        let bytes = s.to_msgpack().expect("encode");
        let restored = DdSketch::from_msgpack(&bytes).expect("decode");

        prop_assert_eq!(restored.total_count(), s.total_count());
        prop_assert_eq!(&restored.store_counts, &s.store_counts);
        prop_assert_eq!(restored.store_offset, s.store_offset);
        for q in QUANTILE_PROBES {
            prop_assert_eq!(restored.quantile(q), s.quantile(q), "q={}", q);
        }
    }
}
