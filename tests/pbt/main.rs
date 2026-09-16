//! Property tests: the laws each sketch family claims, checked with
//! `proptest` against oracles derived independently of the implementation.
//!
//! One target rather than one per file, so `cargo test --test pbt` runs
//! the whole suite and the library is linked once.

mod asapv1_roundtrip;
mod heap_invariants;
mod merge_algebra;
mod octo_delta;
mod path_equivalence;
mod reference_models;
mod space_saving_bounds;
mod wire_roundtrip;
