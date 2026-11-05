# Secondary-proton crossing partitions (2026-10-02)

This closes the specific missed-partition defect found during the previous
Stage-4 review. It is a numerical correction to the existing diagnostic,
radial, single-generation CSDA operator, not an additional physical model
prescribed by OST and not qualification of the complete secondary spectrum.
CM-to-laboratory conversion, angular path lengths, higher-generation cascades,
data-domain completeness and general convergence remain separate requirements.

## Derivation and implementation

For shield thickness t, production depth d and range function R,

`R(E_birth) = R(E_exit) + t - d`.

Consequently a fixed birth-energy knot b crosses an exit-energy knot e at

`d* = t - [R(b) - R(e)]`.

Both R values have units g/cm2, so d* has units g/cm2. Only events strictly
inside `(0,t)` are added; 0 and the active end depth already bound integration.
The implementation evaluates neither R nor its inverse outside the supplied
range table. Unsupported output bins retain their own domain blockers.

The event union includes:

- every retained incident slice's outgoing histogram edges, not just a sampled
  or first/last histogram;
- range/stopping and nonelastic MT3 interpolation knots;
- explicit MT3 support endpoints, even if absent from its sampled grid;
- the maximum supported birth range, where a full output-bin mapping enters
  the supported domain;
- supported output-bin boundaries and interior stopping/attenuation knots.

The usual primary-transport events remain `d = R(E0) - R(k)`. The order of a
primary event and a fixed output event changes when

`E0* = R^-1(R(k) + d*)`.

Those coupled events now augment the opt-in incident-energy partition. The
lowest range knot also covers changes in the clipped active primary path.
Both independent MF6 incident grids, source nodes, reaction thresholds,
one-sided discontinuities, and direct/full-thickness mapped knots are retained.
No open GL4 rule crosses a retained jump merely because the uniform depth grid
did not happen to land on it.

Code: `radar.shielding.proton_al_secondary`, helpers
`_secondary_exit_depth_knots`, `_depth_quadrature_intervals` and
`_point_source_quadrature_boundaries`. The default centre-bin operator now has
transformation version `4_depth_crossing_partitions`; the opt-in source
integral with the global-union reference has
`5_diagnostic_incident_and_depth_crossings`. The opt-in bin-local variant below
has `6_diagnostic_per_output_bin_crossings`. Zero thickness retains
its exact identity and does not acquire fictitious transport warnings.

## Bounded work and truthful status

`DepthCrossingQuadrature` defaults to 128 unique depth knots and 4096 additional
coupled incident knots. These are resource limits, not accuracy tolerances.
The immutable options can be passed through `depth_quadrature=`. Zero limits
explicitly exercise the incomplete diagnostic fallback.

If a partition exceeds its limit, no arbitrary prefix is retained. The whole
affected addition is discarded and the prior diagnostic partition is used,
with an explicit budget reason, retained-node counts and declared limits.
If only the outer partition exceeds its budget, complete depth partitions
remain active. The outer status cannot be `converged` when its required
partitions are incomplete, regardless of agreement between sampled estimates.

Missing crossing partitions and unestablished depth accuracy are now distinct
reasons. A complete event set removes the former, but does not by itself prove
accuracy for arbitrary data. The general depth-refinement blocker remains.
The previous false-zero protection remains: two zero GL estimates do not
prove a zero response without an independent support/reaction proof. The
global reference retains its reaction-only proof; the bin-local option adds
the strictly supported histogram-envelope proof described below. Either proof
is deliberately conservative, so a numerically correct total can still carry
unresolved outer intervals.

This also means that `converged` is an estimator result for the outer integral
over the declared operator, not an endorsement of absent transport physics.

## Independent analytic counterexample, now resolved

Tests use an exactly consistent synthetic `R(E)=E`, `S=1`, MT5=2 barn above
5 MeV, MT3=3 barn, yield=1, a uniform birth PDF on 0.39..0.3901 MeV, t=3 g/cm2,
and a linear source from `(7,0)` to `(8,1)`. The first exit edge is `1/sqrt(7)`
MeV. With `u=8-E0` and `w=E_birth-E_exit_min`, the source/depth integral is

`integral_0^w (1-u)(w-u) du = w^2/2 - w^3/6`.

Averaging over the birth PDF and multiplying by the independently known
reaction and total-path attenuation factors gives `2.6559853528781e-6` counts
in the source's temporal convention. The previous rule could miss the entire
positive interval; after collinear source-node insertion, even 128 uniform
depth steps still differed by about 2.48%.

| Depth steps | Corrected count | Relative error against analytic result |
|---:|---:|---:|
| 1 | 2.655985352878617e-6 | 8.51e-14 |
| 2 | 2.655985352878617e-6 | 8.51e-14 |
| 8 | 2.655985352878617e-6 | 8.51e-14 |
| 128 | 2.655985352878617e-6 | 8.51e-14 |

Each used two depth events and 84 response evaluations. The result remains
diagnostic/incomplete for other reasons; an accurate scalar is not promoted
to normative readiness. The table compares two floating-point computations of
the analytical identity; the regression tolerance is 1e-7, not a claimed
physical precision of 1e-13. Independent tests cover both peak and fluence units,
collinear insertion, `R(E)=2E` dimensional scaling, support clipping, every
incident PDF slice, typed support boundaries and bounded fallbacks. The old
missed-zero regression explicitly tests the fallback, instead of being deleted.

## Cost of real tables and next work

The retained TENDL data contain 10 products, 106 histograms and 635 distinct
birth edges (622 inside the Al range support). Counting only birth-PDF versus
exit-bin crossings, before the additional attenuation/stopping events, gives:

| Al, g/cm2 | Nine-node narrow-source grid | Eleven-node grid | GCR 81-node grid |
|---:|---:|---:|---:|
| 0.01 | 157 | 157 | 512 |
| 0.1 | 421 | 432 | 1759 |
| 1 | 885 | 955 | 4177 |
| 3 | 1268 | 1427 | 6126 |
| 10 | 1822 | 2118 | 8795 |

The full coupled Cartesian product is much larger. Even exact restriction to
incident-active PDF pairs still requires tens of thousands of outer response
evaluations for the full GCR grid. It must not be enabled as an unbounded
global union in ordinary source calculations. The real narrow 30-40 MeV test
therefore explicitly records its default-budget fallback; agreement of its
two source grids is no longer described as completed crossing quadrature.

## Opt-in per-output-bin operator

`IncidentEnergyQuadrature(partition_strategy="per_output_bin")` now constructs
each original output bin's depth and coupled incident partitions separately.
It evaluates `K_j(E0)` only on that bin's partition, using the original centre
and both original edges. The incident spectrum remains the full piecewise
linear source on its true support. Neither its half-bin tails nor a one-point
reconstruction of output-bin boundaries are introduced.

Every retained PDF slice is still included in crossing construction. No event
is removed based on sampled activity. Incident range, MF3, yield/PDF grids and
one-sided jumps are unchanged. The default `global_union` remains available as
a numerical reference, and default source pipelines are not switched to the
new diagnostic option.

The crossing-knot limits apply per partition: the full exit grid for the
reference, or one exit bin for the new option. An affected partition falls
back as a whole and records `INCOMPLETE`; other bins keep their full partitions.
By contrast, `max_response_evaluations` is one strict **global** work limit.
It counts vector responses for `global_union` and scalar bin responses for
`per_output_bin`. Metadata states the unit and scope, per-partition response
counts, unresolved intervals and budget failures. A cache key includes both
partition identity and entrance energy. Numerical work counts with different
units must not be presented as directly comparable timing measurements.

All partitions receive their coarse pass before adaptive refinement starts,
subject to the shared budget. If that budget is exhausted even during the
coarse pass, omitted bin/incident intervals are explicitly unresolved; retained
counts are partial, not a complete spectrum. Absolute coarse/fine errors are
summed in exit-bin counts and only then divided by total counts. An unknown
error in any interval makes the global error estimate unavailable. Physical
and data-domain blockers are never cleared by estimator agreement.

### Supported zero-response proof

For a fixed output bin `[a,b]` and `0 <= d <= t`, all possible birth energies
lie in `[a, R^-1(R(b)+t)]`. If this interval has no positive-width intersection
with any possible positive histogram segment, that bin's integral is zero.
The final histogram ordinate has no following interval; zero-density segments
and boundary-only contacts have zero measure.

The proof requires the entire primary path, output bin and birth upper bound
to stay inside range-table support, and the entire reachable incident interval
to stay inside declared kernel, MF3 and yield support. For each channel,
MF3 times yield is first tested for identically zero on every knot-free cell,
including one-sided jumps. Only a proven inactive channel needs no PDF.
Otherwise its PDF grid must cover the whole reachable incident interval.
Declared linear interpolation then permits exactly all bracketing and interior
PDF slices of that interval; it does not require unrelated high-energy slices.
An intermediate positive slice or either bracketing neighbour cannot be skipped.

This proof is used only after coarse and fine samples are zero. Those samples
and their coverage evidence are retained; it is not a shortcut that suppresses
range/domain warnings. Missing PDFs are not interpreted as a physical zero.

### Independent numerical checks

Both strategies reproduce the narrow analytic case above. Separate tests
integrate a uniform birth distribution over each original exit bin using an
independent antiderivative, including at least three positive bins. Both
PEAK and CUMULATIVE semantics are preserved without an extra time multiplier.

Identical mathematical transport does **not** imply bitwise equality of finite
GL4 quadratures: the global union adds other bins' unnecessary depth splits,
which still affect numerical integration of nonlinear attenuation. A separate
synthetic test uses `R(E)=E`, a triangular source on 8..12 MeV, uniform births
on 2..7 MeV and MT3 cross sections `(100,3,100)` barn on `(0.1,5,20)` MeV.
Depth refinement 1 -> 8 -> 16 checks every output bin, with agreement between
strategies at 8 and 16 steps and within-strategy refinement at relative 1e-8.
The general depth-accuracy blocker remains; one test is not a full-source proof.

The subsequent exact active-PDF-cell pruning of the **crossing partitions**
is now implemented as a third opt-in strategy; see `secondary_active_cells.md`.
The original two reference strategies remain available. Full-source/depth
refinement at 0.01, 0.1, 1, 3 and 10 g/cm2 remains open. The small cases here
must not be described as closure of the representative GCR matrix or normative
secondary transport.

### Full-table bounded timing comparison

The reproducible helper `scripts/benchmark_secondary_bin_quadrature.py` uses
all ten retained TENDL products, with an explicitly synthetic triangle on
33..33.02 MeV and five exit bins. Its separate preflight rejects any crossing
partition fallback before timing. The measured case uses t=0.0001 g/cm2,
one uniform depth step (in addition to all event splits), relative outer
tolerance 0.001, response budget 4096, depth-knot budget 8192 and coupled-knot
budget 65536. This thin numerical test is **not** one of the required Stage-4
shield thicknesses and is not a GCR environment reference.

| Strategy | Measured calculation time | Response work | Outer estimator |
|---|---:|---:|---|
| global_union | 68.0142 s | 24 vector responses, 120 bin components | converged |
| per_output_bin | 8.30912 s | 120 scalar bin responses | converged |

The full five-bin count vectors are identical in the recorded floating-point
results. Both event partitions are complete; neither uses fallback or exceeds
the response budget. The measured time ratio is 8.19 only for this local case,
not a general performance guarantee. Both output spectra still carry the
physical-model and unestablished general depth-accuracy blockers.

Inputs, actual output vectors, options, statuses, local runtime information and
SHA256 hashes are retained in
`secondary_bin_quadrature_benchmark_2026-10-02.json`. The production-code hash
was independently checked after the measurements. Each measured strategy was
run under an external 110-second process limit. Reproduce with the configured
environment:

```text
.venv\Scripts\python.exe scripts\benchmark_secondary_bin_quadrature.py --thickness 0.0001
```

The helper writes JSON to stdout and progress to stderr. For a strict runtime
cap, invoke one strategy at a time under an external process timeout; the
response budget is a work limit, not a wall-clock timeout.

An independent repeat through the final helper at t=0.00001 g/cm2 also kept
both complete partitions and converged outer estimators: 7.096 s versus
2.378 s, with maximum relative nonzero-bin difference 3.48e-16. The different
local time ratio (2.98 rather than 8.19) further illustrates why neither
measurement should be extrapolated to the full production matrix.

## Verification

The combined secondary, analytic crossing, zero-estimator and magnetic-runner
tests passed **123 tests in 43.06 s**. An additional integration/regression run
covering GCR/SEP/ERB wrappers, spectrum operations, Stage-4 release checks and
magnetic coordinates passed **273 tests in 2.92 s**. These are scoped runs,
not a final complete-repository run. Ruff and mypy passed. Tests requiring
temporary files were run with the execution permission needed to avoid a
Windows sandbox ACL failure; no test assertions or filesystem permissions
were weakened to bypass it.

After the bin-local implementation and final independent review, the expanded
secondary/zero-proof/magnetic-runner suite passed **190 tests in 45.20 s**.
The separate source-chain, spectrum, Stage-4 gate and magnetic-coordinate
regression suite passed **273 tests in 3.34 s**. The two new test files contain
64 cases, including actual call-count budget checks, bin-local fallback,
unsupported domains, nonlinear depth refinement and absent-physics blockers
even when the outer estimator converges. Ruff and mypy (133 source files)
passed. The tree collects 1677 tests; no full 1677-test run is claimed here.
