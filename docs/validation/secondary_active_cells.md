# Secondary-proton active-cell partitions (2026-10-02)

This is a further numerical optimization of the diagnostic single-generation
radial CSDA operator, not a new OST prescription or a completed normative
transport model. The default production path is unchanged. The two earlier
opt-in strategies, `global_union` and `per_output_bin`, remain references.

The new option is
`IncidentEnergyQuadrature(partition_strategy="per_output_bin_active_cells")`,
recorded as transformation version `7_diagnostic_active_pdf_cell_crossings`.

## Exact selection, not sampled activity

The primary energy is `Ep(E0,d)=R^-1(R(E0)-d)`. Split its supported domain into
cells `[a,b]` at every retained range, MF3, yield, PDF and MT3 knot, one-sided
jump and declared support endpoint. On a cell, the declared incident PDF
interpolation uses the same bracketing histogram slices. Include all those
slices, even when a bracketing slice's own energy lies outside the reachable
interval. A middle slice or narrow yield peak cannot be discarded by checking
only a few entrance energies or depth samples.

A channel is excluded only when its nonnegative, piecewise-linear MF3 times
yield is identically zero on the whole cell. If required PDF coverage is
unknown, retain all that channel's stored histogram edges instead of pruning
them, and record an explicit missing-PDF-support reason. Unknown data are not
represented as a physical zero.

The PDF birth/exit crossing retains its original depth

`d* = t - [R(E_birth_edge) - R(E_exit_knot)]`.

Attach it to the primary cell in which its PDF may participate. For a given
entrance energy, it is needed only when

`R(a) <= R(E0) - d* <= R(b)`.

The corresponding outer incident-energy events are

`E0_left = R^-1(R(a)+d*)`, `E0_right = R^-1(R(b)+d*)`.

Both endpoints of **every original cell** are retained. Adjacent cells are not
merged just because their PDF pairs happen to match: the shared MF3/yield/range
knot can still change the incident integrand. Range functions are evaluated
only inside their supported domains.

Stopping/MT3/domain birth events remain unconditional and are still coupled
to all primary knots. Conditional PDF selection does not remove the maximum
supported birth-range event, MT3 support endpoints, output-bin boundaries or
interior attenuation knots.

## Source reachability

The source remains piecewise linear on its original support. A segment is
omitted only if both endpoint ordinates are zero. A positive-to-zero segment
is retained in full. Intersect these source intervals with the range-table
domain; omitted out-of-range input keeps its existing blocker.

For supported source mass ranges `[r0,r1]`, reachable primary ranges lie in
`[r0-t,r1]`. A conditional event for cell `[R(a),R(b)]` can participate only
if `[R(a)+d*,R(b)+d*]` intersects a positive source interval. These are closed,
conservative interval tests, not estimates based on source samples. Necessary
interpolation neighbours can lie outside these intervals; selection is based
on interpolation support, not on whether a table row is itself reachable.

## Budgets and status

Per output bin, `max_depth_knots` limits unique geometric depths, while
`max_active_cell_records` (default 32768) separately limits attached
`(primary cell, depth)` records. These counts are different. The coupled outer
event budget remains separate, and `max_response_evaluations` remains one
global scalar-response budget across all bins and cells.

If either depth limit is exceeded, discard the entire added depth partition,
not an arbitrary prefix. Retain the prior bounded diagnostic rule and mark
`INCOMPLETE`. If the coupled outer budget alone is exceeded, retain the complete
depth partition but explicitly report the incomplete outer partition.

Metadata records strategy, source support, active cells, attached records,
unique depths, work counts, missing PDF support and unresolved intervals.
Unchanged physical-model, depth-accuracy and incoming-source blockers remain.
Outer estimator agreement cannot authorize normative export.

The retained-event limits are not a wall-clock or total temporary-memory cap:
candidate crossings are constructed before filtering and counting retained
records. Expensive validation runs therefore also use an external process
timeout. This remaining preparation-cost limitation is not a numerical
completeness claim.

## Floating-point boundary checks

Independent review exposed two endpoint cases, both retained as regressions:

- Subtracting nearly equal large ranges can lose a very small conditional
  depth event. The activation test now expands the mass-range operands
  conservatively before subtraction; it may retain an extra split, but never
  moves the original physical event depth.
- The arithmetic midpoint of adjacent floating-point endpoints can equal an
  endpoint. Its zero ordinate cannot prove a nonnegative linear factor zero
  inside the mathematical interval. The channel-zero proof now declines to
  certify such a cell unless an actual interior midpoint exists.

These corrections do not change the CSDA or attenuation formulas, supply
missing tables or weaken a convergence tolerance.

## Validation scope

Independent tests compare against complete global/per-bin references, the
narrow analytic crossing example and a separate narrow triangular yield
integral. They cover both time units, collinear source refinement, independent
yield/PDF grids, nonconstant production and attenuation, global work limits,
explicit fallbacks and preservation of input incompleteness. Nonlinear depth
tests use refinement rather than demanding identical finite GL4 errors.

Full-source work must additionally distinguish a represented numerical zero
from an unavailable TENDL contribution. The latter remains `INCOMPLETE`.
Likewise, doubling source/output centres changes the derived outer log-bin
edges; an integral comparison must state its domain instead of attributing
domain movement to numerical quadrature error.

This optimization alone does not certify the full orbital/energy/depth matrix.
CM-to-LAB energy-angle conversion, angular path lengths, higher generations,
range-data provenance and general depth convergence remain unresolved.

## Measured pilot and full-source work forecast

The full actual TENDL kernel was evaluated for a synthetic triangular source
on 33..33.02 MeV, five original exit bins and 0.01 g/cm2 Al. This is a required
shield thickness, but the narrow synthetic source is **not** the complete GCR
source. Both the 1-step and 2-step depth runs included all crossing events and
used 120 scalar responses, with converged outer estimators. The maximum
nonzero-bin relative difference was 1.70e-16; the total difference was 1.14e-16.
This is floating-point agreement in one example, not a general accuracy bound.

The two initial runs used the code before the adjacent-float proof guard. A
further 2-step run on final code SHA `5e08bbc7...` exactly reproduced the prior
2-step count vector. The final total is `1.861342874726708e-6` in the synthetic
source's fluence normalization. Full physical/depth blockers remain. Recorded
times were approximately 105, 98 and 116 seconds, respectively; concurrent
local load differs, so these are not comparative speedup measurements.

The inputs, limits, hashes and actual output arrays are retained in:

- `secondary_active_cells_pilot_2026-10-02.json` (initial depth=1);
- `secondary_active_cells_pilot_depth2_2026-10-02.json` (initial depth=2);
- `secondary_active_cells_pilot_final_2026-10-02.json` (final-code depth=2).

A separate real-source preflight uses the OST GCR proton component on GEO,
2026 launch, five years and an explicit 60-month solar timeline. Both PEAK and
CUMULATIVE sources are retained with their original units and time semantics.
Only H is generated for this proton-operator study; the inherited incomplete
ion-roster warning remains. All ten TENDL products and the full incident energy
domain are retained, including the unsupported high-energy part.

Complete partitions were built for both 81- and 161-node source grids at all
five thicknesses, with no partition fallback at the explicitly enlarged
limits (8192 unique depths, 65536 coupled knots, 32768 attached records per
output bin). This run performs **zero secondary-response evaluations**. It
counts actual distinct GL4 abscissae to forecast a complete coarse pass:

| Al, g/cm2 | 81-node coarse scalar calls | 161-node coarse scalar calls |
|---:|---:|---:|
| 0.01 | 368604 | 662172 |
| 0.1 | 1096491 | 1773859 |
| 1 | 2567952 | 4037214 |
| 3 | 3273649 | 5106884 |
| 10 | 4163616 | 6462521 |

These counts already exceed the default global response budget of 4096,
before error estimation/refinement. Thus a full spectrum cannot be called
converged under that budget. Merely increasing the limit would not establish
accuracy or repair unsupported high-energy data. Next work is pre-quadrature
proof and explicit accounting of represented zero/unsupported contributions,
followed by a full source/depth refinement ladder. Unknown physical flux must
never be described as zero to reduce cost.

The reproducible runner is `scripts/validate_secondary_active_cells.py`; use
`--output` to save its complete large JSON report directly. The retained
`secondary_active_cells_preflight_2026-10-02.json` was verified to contain
exactly all ten grid/thickness pairs, complete ordered output-bin records,
matching sums and stable source/input/code hashes. Truncated terminal output
from an earlier attempt was rejected, not reconstructed as evidence.

## Final scoped verification

On final code, 253 secondary/boundary/preflight/magnetic-runner tests passed
in 69.62 seconds. The separate integration, release-gate and L/B regression
suite passed 273 tests in 3.35 seconds. Ruff and mypy (133 source modules plus
both validation helpers) passed. The complete tree collects 1740 tests;
no full 1740-test run or normative qualification is claimed.

## Pre-quadrature classification, 2026-10-29

The optional `IncidentEnergyQuadrature(prequadrature_screening=True)` now
classifies whole base intervals **before** building their expensive coupled
crossing partitions. It requires `per_output_bin_active_cells`; the default
remains false and production pipelines are unchanged. Its diagnostic operator
version is `8_diagnostic_prequadrature_screening`. This is a numerical change
to the existing provisional operator, not a new physical transport model or
an additional formula attributed to OST.

Each original output bin has its own classification of the same incident
source. Base boundaries include source nodes, direct model knots and their
full-depth range-shifted events. A parent classification is inherited only by
subintervals wholly contained in that parent; a crossing interval must be
evaluated. There are four distinct categories:

- `source_zero`: both ends of the piecewise-linear source panel are zero;
- `proven_zero`: the fully supported reaction is identically zero, or every
  required positive PDF segment is disjoint from the reachable birth window;
- `omitted_unsupported`: the present operator cannot represent any complete
  contribution on this interval; the physical contribution is **unknown**;
- `evaluate`: all other intervals, including partially supported paths and
  missing active PDFs. These retain the existing quadrature and work limits.

The proof checks do not sample the secondary response. In particular, two
zero Gauss estimates are still not proof of a zero integral. A reaction-zero
proof checks all factor knots; a geometry proof checks all required PDF
interpolation neighbours and interior slices. No missing table is filled by
extrapolation or used to establish zero.

### Support inequalities and units

Let `R(E)` be the current monotone Al range table in g/cm2, `t` the shielding
in g/cm2 and `[a,b]` a supported incident-energy interval in MeV. The complete
primary-energy envelope used for domain checks is

`[R^-1(max(Rmin, R(a)-t)), b]`.

Checking only `[a,b]` would be wrong: an entrance energy above TENDL support
can lose energy inside the shield and reach a supported reaction region.
Only an envelope wholly outside the reaction support is omitted; an envelope
which straddles a support boundary remains evaluable. The below-table
remainder when `R(a)-t < Rmin` remains explicitly unsupported, not repaired.

For output upper edge `u`, the latest represented primary depth is bounded by
`dmax = min(t, R(b)-Rmin)`. A complete output-bin birth mapping is unavailable
at every represented depth if

`R(u) + max(0, t-dmax) > Rmax`.

This is a strict sufficient condition, not permission to discard a partial
supported depth strip. Equality and neighbouring floating-point values are
retained for evaluation. Range operands are expanded with directed rounding
before subtraction/addition, avoiding cancellation-based false omissions.
Independent `R(E)=2E` tests distinguish range in g/cm2 from energy in MeV.

The existing geometry-zero proof is allowed only with complete primary,
reaction and birth-range support. Unsupported output edges cannot be clipped
silently. A missing active PDF remains evaluable and incomplete, even when
the stored PDF segments appear disjoint.

### Accounting and uncertainty

Only evaluable incident-range intervals generate active/coupled crossings;
proven zeros and unsupported omissions spend no response calls. The original
source/direct boundaries remain, and any work-budget failure still rejects
the entire added partition rather than retaining an arbitrary prefix.

For each original output bin, the four source weights sum to the integral
of the declared piecewise-linear incident spectrum. These weights describe
the input regions, **not** secondary yields or bounds on missing output.
They are recorded separately in `prequadrature_source_integrals_by_output_bin`
and must never be summed across bins as physical flux/fluence. Neither the
screening nor these weights introduce a multiplication by mission duration.

Domain checks retain primary-path endpoints even when screening leaves the
response cache empty. Thus reaction/MT3/range warnings do not disappear as a
side effect of doing less work. `incident_quadrature_estimated_*_error` becomes
`unavailable` whenever the screened calculation has an unknown domain
contribution, including a partially supported reaction path with no wholly
omitted panels. In that case the full quadrature status is `incomplete`.
Separate `incident_quadrature_supported_*` fields describe only the retained
numerical contributions; they exclude the unknown physical domains and are
not a complete-result error bound or normative acceptance.

The independent public and boundary tests are
`test_proton_secondary_prequadrature.py` and
`test_proton_secondary_screening_boundaries.py`. They compare against analytic
integrals and the unscreened reference, including narrow positive responses,
partial support, missing PDFs, zero reaction, directed-rounding boundaries,
work-budget exhaustion and both PEAK/CUMULATIVE time semantics.

### Full-source work forecast after classification

`secondary_prequadrature_preflight_2026-10-29.json` retains a fresh comparison
of separately constructed v7/v8 partitions on the same GEO GCR case as the
previous preflight. All 25 pinned input/code hashes match before, after and
on independent recheck; all 10 grid/thickness cases and 1210 ordered output
bin records are present. There are no reference or screened partition-budget
fallbacks. The unscreened counts reproduce the previous artifact exactly.
The forecast performs **zero secondary-response evaluations**.

| Al, g/cm2 | 81 nodes: unscreened -> retained | 161 nodes: unscreened -> retained |
|---:|---:|---:|
| 0.01 | 368604 -> 225248 | 662172 -> 360020 |
| 0.1 | 1096491 -> 821139 | 1773859 -> 1283607 |
| 1 | 2567952 -> 1920239 | 4037214 -> 2979210 |
| 3 | 3273649 -> 2434688 | 5106884 -> 3763133 |
| 10 | 4163616 -> 3087202 | 6462521 -> 4758519 |

The 25.1-45.6% reduction is in forecast coarse response-call counts, **not**
a measured wall-clock speedup. The remaining 225248-4758519 calls still
exceed the default 4096 budget in every case, even before error estimation
and refinement. No full-source convergence result follows from this table.
The report separates base/final work counts and all classification reasons;
unsupported-panel counts are numerical work records repeated across bins,
not physical event counts. Per-bin classified source weights reproduce the
source integral with maximum relative difference 2.78e-16. They use PEAK
units explicitly; CUMULATIVE source values and time semantics are separately
retained. The shared work geometry follows from identical positive support,
not from converting peak flux into fluence.

Reproduce the geometry/work forecast with:

```text
python scripts/validate_secondary_active_cells.py --prequadrature-screening --output <new-report.json>
```

This progress closes the implementation and bounded validation of the
pre-quadrature accounting step, not Stage 4. The next numerical task is to
reduce the cost of retained nonzero responses and then run justified
source/depth refinement ladders. Raising a budget alone, treating unsupported
panels as physical zero, or exporting this forecast as a calculated secondary
spectrum would not satisfy the acceptance criteria. Physical transport
omissions, data provenance/coverage and COSRAD gates remain separate.

### Calculated reference comparisons and final scoped checks

The benchmark runner now accepts `--compare-screening` for the active-cell
strategy and `--output` for a complete, non-overwriting JSON artifact. It
refuses to publish a result if its listed input/code hashes differ between
the start and end of calculation. This guards those listed dependencies,
not unlisted files or an edit reverted between checks.

Two actual integrations compare v7 and v8 with the synthetic triangular
source on 33-33.02 MeV and all ten retained TENDL products:

- `secondary_prequadrature_pilot_2026-10-29.json`: 0.0001 g/cm2, one depth
  step, total retained count `1.9485662767020681e-8` in the declared source
  normalization;
- `secondary_prequadrature_pilot_0p01_2026-10-29.json`: 0.01 g/cm2, two depth
  steps, total retained count `1.8613428747267081e-6`, also reproducing the
  earlier narrow-source depth-two checkpoint.

Every output-bin count is exactly equal between the screened and reference
calculation in both pilots. All crossing partitions are included; response
calls decrease from 120 to 72. The reported relative outer estimators are
approximately 1.78e-13 in both variants. The calculations remain `INCOMPLETE`
because the physical transport and general depth-accuracy blockers remain.
They are not full GCR spectra or the required orbit/thickness validation.

Observed calculation times are 2.714/2.736 seconds and 89.983/94.077 seconds
(unscreened/screened), respectively. No wall-clock speedup is demonstrated by
these pilots; local concurrent load also differed. Removing cheap zero-bin
calls is not sufficient to accelerate the expensive positive responses.

The final scoped run passed 319 numerical/boundary/runner tests in 57.72 s.
The separate source-chain, release-gate and L/B regression run passed 273
tests in 6.62 s. Ruff and mypy (135 source/helper modules) passed. The tree
collects 1806 tests, including 66 new tests; no full 1806-test run is claimed.
The validated production-module SHA-256 is
`5b800e23702ed6497df29e670a3ba9e244d9a03a0bf8ed9baacf1b3f521f24dc`.
OST formula coefficients, L/B algorithms and production defaults were not
changed by this pre-quadrature implementation.

## Exact nonzero-response work reuse, 2026-10-30

Profiling the retained nonzero work identified two repeated computations,
without requiring a change to the physical model or to quadrature accuracy:

1. The shared proton range/stopping interpolator linearly scanned its sorted
   table for every query. It now uses `bisect_left` to select the same left
   interval. Endpoints, domain errors and the arithmetic expression are
   unchanged. At an exact interior knot it deliberately does not substitute
   the stored ordinate: that would change rounding relative to the old scan.
   Independent tests compare hexadecimal float representations at all actual
   table knots, adjacent floats, deterministic dense samples and adversarial
   synthetic tables for range, inverse range and stopping power.
2. For one fixed birth-to-exit mapping, all direct and mapped attenuation
   knots depend only on its output bin, residual thickness and tables. They
   do not depend on the selected PDF histogram segment. The new explicit
   `reuse_birth_survival_partition=True` option builds those transport knots
   once, then selects their strict interior subset for each original overlap
   using binary searches. The old per-overlap construction remains available
   and is still the default. The option works with all three incident
   quadrature strategies, independently of pre-quadrature screening.

For any histogram overlap `[l,u]` contained in the mapping's full birth
interval, restricting the full transport-knot union to `(l,u)` and adjoining
`l,u` gives exactly the same ordered partition as the reference construction.
Consequently the GL4 segments, nodes, weights and accumulation order are
unchanged. The existing segment-integral cache remains local to **one**
mapping; neither knots nor probabilities are shared between depths, output
bins or datasets. No CDF subtraction, exponential-ratio rearrangement,
PDF coarsening, tolerance change or interpolation-law change is introduced.

The new numerical option records version
`9_diagnostic_birth_survival_partition_reuse` and the exact reuse scope.
All missing-domain, frame, angular, cascade and general depth-accuracy
restrictions remain. The binary search is a general implementation
improvement; production strategy and reuse-option defaults are unchanged.

### Independent checks and measured comparisons

`test_proton_range_binary_interpolation.py` has 39 independent cases against
the legacy scan. `test_secondary_birth_partition_reuse.py` has 30 cases,
including both time modes and all strategies, nonlinear attenuation with an
MT3 jump, independent PDF grids, exact GL4-segment equality around adjacent
floats, a positive `5e-21` histogram tail, missing-data warnings and a shared
response budget. The tests also change depth and MT3 tables successively to
detect accidental cross-mapping cache reuse.

The real-data pilot `secondary_birth_partition_reuse_2026-10-30.json` compares
reuse off/on with screening held enabled, all ten TENDL products retained,
two depth steps and the same synthetic incident triangle on 33-33.02 MeV.
All 17 listed dependency hashes match at start/end and on recheck.

| Al, g/cm2 | Per-overlap, s | Per-mapping reuse, s | Local ratio |
|---:|---:|---:|---:|
| 0.0001 | 1.265 | 0.655 | 1.93 |
| 0.01 | 28.884 | 16.198 | 1.78 |

Both sides already use binary range lookup, so these paired ratios measure
the additional reuse option under the local run conditions, not the combined
gain or a guarantee for other spectra. Every bin ordinate and count agrees
bit for bit, along with warnings, error estimators and numerical statuses.
Both variants use 72 responses and remain `INCOMPLETE` as physical results.

The separate local cProfile runs (`outputs/secondary_nonzero_before_20261030`
and `outputs/secondary_nonzero_after_20261030`, `.json`/`.prof`) use one depth
step at 0.0001 g/cm2. Their output vectors are identical. They show the
mechanism rather than just a timing difference:

| Operation | Before calls | After calls |
|---|---:|---:|
| Build birth-survival boundaries | 29952 | 864 |
| Map output energy to birth energy | 468480 | 14112 |
| Strict range/stopping interpolation | 1293578 | 384842 |
| Integrate a survival segment | 41280 | 41280 |
| Single-incident scalar response | 72 | 72 |

No required segment integration was removed. The profiled calculation times
5.160/2.120 seconds are descriptive only: two implementation changes and
different concurrent load prevent treating that ratio as an isolated effect.

At 0.01 g/cm2, additional records
`secondary_birth_partition_depth4_2026-10-30.json` and
`secondary_birth_partition_depth8_2026-10-30.json` complete a 2 -> 4 -> 8 depth
ladder for this narrow source. Depths 2 and 4 give exactly the same bin counts
and total `1.8613428747267081e-6`. At depth 8 the first bin differs
by one float step (`2.6469779601696886e-23`, relative `1.7471727604139175e-16`);
the other bins and the rounded total are unchanged. This is scoped evidence
at machine-rounding scale for that source and thickness, not a substitute
for full GCR 81/161-grid or orbital validation.

### Bounded execution of the complete GCR source

`scripts/validate_secondary_full_source.py` now executes the real GEO GCR H
source with its full original energy domain and all ten retained TENDL
products. It uses the existing 2026/five-year/60-month source builder and
records source/output dataclasses, per-bin response work, all warnings and
start/end hashes. Its 25 independent runner tests check both grids/modes,
including positive unsupported source energies, no extra time factor,
budget accounting, restored instrumentation and exclusive/hash-checked output.

The following actual 0.01-g/cm2, depth-one runs are retained (suffix
`_2026-10-30.json`):

| Report stem | Grid | Mode | Responses | Calculation seconds |
|---|---:|---|---:|---:|
| `secondary_full_source_peak_128` | 81 | PEAK | 128 | 4.501 |
| `secondary_full_source_cumulative_128` | 81 | CUMULATIVE | 128 | 6.181 |
| `secondary_full_source_peak_161_128` | 161 | PEAK | 128 | 8.399 |
| `secondary_full_source_peak_4096` | 81 | PEAK | 4096 | 91.523 |

These are intentionally partial execution diagnostics, **not** complete
secondary spectra or grid-convergence comparisons. The global budget is
exhausted in the first output bin; zero retained values elsewhere do not mean
zero physical contribution. All outputs retain `INCOMPLETE`, unresolved
intervals and `unavailable` full-result error. The 4096-call run records 56312
unresolved intervals. Increasing its budget did not establish convergence.
The complete-coarse forecast remains 225248 calls for this 81-node case,
before error estimation/refinement; these partial timings alone cannot
reliably forecast the cost of all other energies/bins.

PEAK outputs retain `cm^-2 s^-1 MeV^-1` and no mission-duration multiplier;
CUMULATIVE retains `cm^-2 MeV^-1` with the original 157788000-s mission.
The reported `retained_diagnostic_count_sum` excludes unknown and unresolved
contributions and must not be exported as a completed environment result.
All four artifacts were checked against current pinned dependencies and
their complete 81/161-node input/output arrays.

### Final repository verification

The first full run recorded 1886 passes and three historical AE8/AP8 fixture
failures in 1051.47 s. Their old-v1-versus-current-v2 geometry mismatch was
isolated without changing field physics, spectral reference numbers or
tolerances; see `ost_appendix_d_magnetic_coordinates_audit.md`.

After the explicit fixture-scope correction and new runner tests, a second
complete run passed **all 1917 tests**, with zero failures, errors or skips,
in 1054.81 s. Its retained JUnit file is
`outputs/pytest_full_stage4_final_20261030.xml` (SHA-256
`c75cf6d7585ed22d9b95dc09d89749b97dcc21f8f430de0ee2ea09367c0ce790`).
The first failing report was preserved separately. Ruff and mypy passed;
mypy checked 136 source/helper modules. Final production hashes are:

- `proton_al_secondary.py`:
  `a5cca0111a6d72f29c1aafc6befe55733c5088521a3a12d080e2fef169812a13`;
- `proton_al.py`:
  `161b8c418ee9bd3dcc1cd651a7e000a8b1d199730f9f890bbdac32fd124ca227`.

This is a complete repository regression pass, not closure of Stage 4,
independent COSRAD agreement, full-source secondary numerical accuracy or
physical/normative qualification.
