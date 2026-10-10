//! The deterministic numeric substrate that width replay runs on, behind a trait.
//!
//! Replay evaluates the envelope's program once on the supplied inputs and then
//! `K` more times on dither-resampled copies of them. Everything about that
//! except three numeric primitives is in this crate: the program grammar, its
//! classification, alignment, the operators, the resource bounds and the
//! verdict. The three primitives are the substrate: a compensated sum, a mean,
//! and the seeded dither stream. They are this trait.
//!
//! The specified substrate is a private deterministic kernel and is not linked
//! here. A program that carries no [`ReplayKernel`] still checks
//! the signature, the program hash, the inputs, transparency, witness, provider
//! and key lifecycle; it does not replay, so `scalar`, `tier`, `budget` and
//! `width` stay null (not performed) and the verdict's `ok` stays false. No
//! fallback arithmetic stands in for the substrate.
//!
//! An implementation's [`ReplayKernel::id`] is what the verifier compares with
//! the envelope's `kernel` to decide whether a nonzero width can be checked
//! exactly. Report it honestly: a kernel that claims the specified substrate's
//! id and computes something else makes this program's width verdicts wrong.

/// The substrate width replay runs on.
pub trait ReplayKernel: Send + Sync {
    /// This substrate's identity, compared with an envelope's `kernel`.
    fn id(&self) -> &str;
    /// The compensated sum of `values` (which carry no NaN).
    fn sum(&self, values: &[f64]) -> f64;
    /// The mean of `values` (which carry no NaN); NaN for none.
    fn mean(&self, values: &[f64]) -> f64;
    /// The dither stream of resample `k` under `seed`. One stream serves every
    /// input of that resample, in the program's reference order.
    fn dither(&self, seed: u64, k: u64) -> Box<dyn DitherStream + '_>;
}

/// The offsets of one resample, drawn input by input.
pub trait DitherStream {
    /// One offset per delta, or why the deltas cannot be resampled.
    fn resample(&mut self, deltas: &[f64]) -> Result<Vec<f64>, String>;
}

/// A kernel for this crate's own tests: a plain sum, and a dither of zero for a
/// zero delta and of half the delta otherwise. Its id names it as a fixture, so
/// no envelope's `kernel` ever matches it and it can never check a nonzero width
/// exactly. Tests that need the specified substrate's numbers live with that
/// substrate, outside this crate.
#[cfg(test)]
pub(crate) struct TestKernel;

#[cfg(test)]
impl ReplayKernel for TestKernel {
    fn id(&self) -> &str {
        "test-fixture/0"
    }

    fn sum(&self, values: &[f64]) -> f64 {
        values.iter().sum()
    }

    fn mean(&self, values: &[f64]) -> f64 {
        if values.is_empty() {
            f64::NAN
        } else {
            self.sum(values) / values.len() as f64
        }
    }

    fn dither(&self, _seed: u64, _k: u64) -> Box<dyn DitherStream + '_> {
        Box::new(HalfDelta)
    }
}

#[cfg(test)]
struct HalfDelta;

#[cfg(test)]
impl DitherStream for HalfDelta {
    fn resample(&mut self, deltas: &[f64]) -> Result<Vec<f64>, String> {
        deltas
            .iter()
            .map(|delta| {
                if delta.is_finite() && *delta >= 0.0 {
                    Ok(delta / 2.0)
                } else {
                    Err(format!("delta {delta} is not a finite non-negative number"))
                }
            })
            .collect()
    }
}
