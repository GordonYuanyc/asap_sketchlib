//! Property tests for the OctoSketch delta protocol.
//!
//! The oracle is a single-threaded sketch over the same stream: whatever the
//! workers promoted plus whatever they still hold must reconstruct it cell for
//! cell, and a flush must close the gap exactly. The counter sketches apply
//! deltas by addition, so every law here is an exact equality.

use asap_sketchlib::{
    CmDelta, CmOctoAggregator, CmWorkerSketch, Count, CountDelta, CountMin, CountOctoAggregator,
    CountWorkerSketch, DataInput, MAX_PROMASK, OctoAggregator, RegularPath, Vector2D,
};
use proptest::prelude::*;

type Cm = CountMin<Vector2D<i32>, RegularPath>;
type Cs = Count<Vector2D<i32>, RegularPath>;

fn stream(max: usize) -> impl Strategy<Value = Vec<u64>> {
    prop::collection::vec(0u64..256, 0..max)
}

fn cm_cells(sketch: &Cm) -> Vec<i32> {
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

fn cm_reference(rows: usize, cols: usize, keys: &[u64]) -> Cm {
    let mut s = Cm::with_dimensions(rows, cols);
    for k in keys {
        s.insert(&DataInput::U64(*k));
    }
    s
}

fn cs_reference(rows: usize, cols: usize, keys: &[u64]) -> Cs {
    let mut s = Cs::with_dimensions(rows, cols);
    for k in keys {
        s.insert(&DataInput::U64(*k));
    }
    s
}

proptest! {
    // ===== Count-Min, full-precision worker =====

    #[test]
    fn cm_promotions_and_residual_reconstruct_a_single_pass(
        rows in 1usize..6,
        cols in 1usize..64,
        tau in 1u32..=MAX_PROMASK,
        keys in stream(400),
    ) {
        let mut child = Cm::with_dimensions(rows, cols);
        let mut parent = Cm::with_dimensions(rows, cols);
        for k in &keys {
            child.insert_emit_delta_with_threshold(&DataInput::U64(*k), tau, &mut |d| {
                parent.apply_delta(d)
            });
        }

        let reference = cm_reference(rows, cols, &keys);
        for (i, ((c, p), r)) in cm_cells(&child)
            .into_iter()
            .zip(cm_cells(&parent))
            .zip(cm_cells(&reference))
            .enumerate()
        {
            prop_assert_eq!(p + c, r, "cell {}: promoted {} plus residual {} != {}", i, p, c, r);
            prop_assert!(c < tau as i32, "cell {}: residual {} reached the threshold", i, c);
        }
    }

    #[test]
    fn cm_every_delta_carries_one_whole_promotion(
        rows in 1usize..6,
        cols in 1usize..64,
        tau in 1u32..=MAX_PROMASK,
        keys in stream(400),
    ) {
        let mut child = Cm::with_dimensions(rows, cols);
        let mut deltas: Vec<CmDelta> = Vec::new();
        for k in &keys {
            child.insert_emit_delta_with_threshold(&DataInput::U64(*k), tau, &mut |d| deltas.push(d));
        }

        for d in &deltas {
            prop_assert_eq!(d.value, tau, "a promotion carried {} rather than tau", d.value);
            prop_assert!((d.row as usize) < rows, "row {} outside {}", d.row, rows);
            prop_assert!((d.col as usize) < cols, "col {} outside {}", d.col, cols);
        }
        prop_assert_eq!(
            deltas.len() as u64 * u64::from(tau) + cm_cells(&child).iter().map(|c| *c as u64).sum::<u64>(),
            keys.len() as u64 * rows as u64
        );
    }

    #[test]
    fn cm_delta_application_is_order_independent(
        rows in 1usize..6,
        cols in 1usize..64,
        tau in 1u32..=MAX_PROMASK,
        keys in stream(300),
        seed in any::<u64>(),
    ) {
        let mut child = Cm::with_dimensions(rows, cols);
        let mut deltas: Vec<CmDelta> = Vec::new();
        for k in &keys {
            child.insert_emit_delta_with_threshold(&DataInput::U64(*k), tau, &mut |d| deltas.push(d));
        }

        let mut ordered = Cm::with_dimensions(rows, cols);
        for d in &deltas {
            ordered.apply_delta(*d);
        }

        let mut shuffled = deltas.clone();
        let mut state = seed | 1;
        for i in (1..shuffled.len()).rev() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            shuffled.swap(i, (state % (i as u64 + 1)) as usize);
        }
        let mut reordered = Cm::with_dimensions(rows, cols);
        for d in &shuffled {
            reordered.apply_delta(*d);
        }

        prop_assert_eq!(cm_cells(&ordered), cm_cells(&reordered));
    }

    // ===== Count-Min, compact worker against the full-precision parent =====

    #[test]
    fn cm_worker_promotions_and_residual_reconstruct_a_single_pass(
        rows in 1usize..6,
        cols in 1usize..64,
        tau in 1u32..=MAX_PROMASK,
        keys in stream(400),
    ) {
        let mut worker = CmWorkerSketch::new(rows, cols);
        let mut parent = CmOctoAggregator::new(rows, cols);
        for k in &keys {
            worker.insert_emit_delta(&DataInput::U64(*k), tau, &mut |d| parent.apply(d));
        }

        let residual = worker.residual().to_vec();
        let reference = cm_reference(rows, cols, &keys);
        for (i, (p, r)) in cm_cells(&parent.sketch).into_iter().zip(cm_cells(&reference)).enumerate() {
            prop_assert_eq!(
                p + i32::from(residual[i]), r,
                "cell {}: promoted {} plus residual {} != {}", i, p, residual[i], r
            );
        }
    }

    #[test]
    fn cm_worker_flush_leaves_the_parent_exact(
        rows in 1usize..6,
        cols in 1usize..64,
        tau in 1u32..=MAX_PROMASK,
        keys in stream(400),
    ) {
        let mut worker = CmWorkerSketch::new(rows, cols);
        let mut parent = CmOctoAggregator::new(rows, cols);
        for k in &keys {
            worker.insert_emit_delta(&DataInput::U64(*k), tau, &mut |d| parent.apply(d));
        }
        worker.flush(&mut |d| parent.apply(d));

        prop_assert_eq!(cm_cells(&parent.sketch), cm_cells(&cm_reference(rows, cols, &keys)));
        prop_assert!(worker.residual().iter().all(|c| *c == 0), "a flush left a counter held back");
    }

    #[test]
    fn cm_sharded_workers_flush_to_a_single_pass(
        rows in 1usize..6,
        cols in 1usize..64,
        tau in 1u32..=MAX_PROMASK,
        shards in 1usize..5,
        keys in stream(400),
    ) {
        let mut workers: Vec<CmWorkerSketch> = (0..shards).map(|_| CmWorkerSketch::new(rows, cols)).collect();
        let mut parent = CmOctoAggregator::new(rows, cols);
        for (i, k) in keys.iter().enumerate() {
            workers[i % shards].insert_emit_delta(&DataInput::U64(*k), tau, &mut |d| parent.apply(d));
        }
        for w in &mut workers {
            w.flush(&mut |d| parent.apply(d));
        }

        prop_assert_eq!(cm_cells(&parent.sketch), cm_cells(&cm_reference(rows, cols, &keys)));
    }

    // ===== Count sketch =====

    #[test]
    fn count_promotions_and_residual_reconstruct_a_single_pass(
        rows in 1usize..6,
        cols in 1usize..64,
        tau in 1u32..=MAX_PROMASK,
        keys in stream(400),
    ) {
        let mut child = Cs::with_dimensions(rows, cols);
        let mut parent = Cs::with_dimensions(rows, cols);
        for k in &keys {
            child.insert_emit_delta_with_threshold(&DataInput::U64(*k), tau, &mut |d| {
                parent.apply_delta(d)
            });
        }

        let reference = cs_reference(rows, cols, &keys);
        for (i, ((c, p), r)) in cs_cells(&child)
            .into_iter()
            .zip(cs_cells(&parent))
            .zip(cs_cells(&reference))
            .enumerate()
        {
            prop_assert_eq!(p + c, r, "cell {}: promoted {} plus residual {} != {}", i, p, c, r);
            prop_assert!(c.abs() < tau as i32, "cell {}: residual {} reached the threshold", i, c);
        }
    }

    #[test]
    fn count_deltas_carry_the_signed_threshold(
        rows in 1usize..6,
        cols in 1usize..64,
        tau in 1u32..=MAX_PROMASK,
        keys in stream(400),
    ) {
        let mut child = Cs::with_dimensions(rows, cols);
        let mut deltas: Vec<CountDelta> = Vec::new();
        for k in &keys {
            child.insert_emit_delta_with_threshold(&DataInput::U64(*k), tau, &mut |d| deltas.push(d));
        }

        for d in &deltas {
            prop_assert_eq!(d.value.abs(), tau as i32, "a promotion carried {}", d.value);
            prop_assert!((d.row as usize) < rows, "row {} outside {}", d.row, rows);
            prop_assert!((d.col as usize) < cols, "col {} outside {}", d.col, cols);
        }
    }

    #[test]
    fn count_worker_flush_leaves_the_parent_exact(
        rows in 1usize..6,
        cols in 1usize..64,
        tau in 1u32..=MAX_PROMASK,
        keys in stream(400),
    ) {
        let mut worker = CountWorkerSketch::new(rows, cols);
        let mut parent = CountOctoAggregator::new(rows, cols);
        for k in &keys {
            worker.insert_emit_delta(&DataInput::U64(*k), tau, &mut |d| parent.apply(d));
        }
        worker.flush(&mut |d| parent.apply(d));

        prop_assert_eq!(cs_cells(&parent.sketch), cs_cells(&cs_reference(rows, cols, &keys)));
        prop_assert!(worker.residual().iter().all(|c| *c == 0), "a flush left a counter held back");
    }

    #[test]
    fn count_sharded_workers_flush_to_a_single_pass(
        rows in 1usize..6,
        cols in 1usize..64,
        tau in 1u32..=MAX_PROMASK,
        shards in 1usize..5,
        keys in stream(400),
    ) {
        let mut workers: Vec<CountWorkerSketch> =
            (0..shards).map(|_| CountWorkerSketch::new(rows, cols)).collect();
        let mut parent = CountOctoAggregator::new(rows, cols);
        for (i, k) in keys.iter().enumerate() {
            workers[i % shards].insert_emit_delta(&DataInput::U64(*k), tau, &mut |d| parent.apply(d));
        }
        for w in &mut workers {
            w.flush(&mut |d| parent.apply(d));
        }

        prop_assert_eq!(cs_cells(&parent.sketch), cs_cells(&cs_reference(rows, cols, &keys)));
    }
}
