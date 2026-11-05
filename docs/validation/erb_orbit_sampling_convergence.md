# OST ERB orbit-sampling convergence

**Historical results:** the numerical tables below predate the 2026-10-02
`ost_appendix_a_v3` magnetic-coordinate correction. They used IGRF/D.32 for
all L and the earlier tracer. They are not acceptance evidence for v3, which
uses the required outer shifted-dipole branch. See
`ost_appendix_d_magnetic_coordinates_audit.md`; regenerate the ladders once
the magnetic-region selection and orbit-trajectory conventions are resolved.

This note records the Stage-4 diagnostic comparison of successive nested OST
ERB orbit grids and the localization of the observed failures.  It is a
convergence measurement, not a COSRAD gold baseline and not evidence that the
OST Appendix A source tables are valid outside their tabulated domain.

## Fixed calculation contract

- source model: `OstErbModel`, strict Appendix A interpolation;
- scenarios: the Stage-4 LEO, GEO and HEO definitions;
- mission: launch in 2026, five years, SEP probability 0.1 (the SEP input does
  not affect the ERB model);
- measured orbit-grid ladder: 48x36, 96x72 and 192x144, with every coarser
  grid exactly embedded in the next grid;
- products: proton and electron `MISSION_FLUENCE` and `MAXIMUM_FLUX` on-orbit
  differential spectra;
- integral comparison: `Spectrum1D.integral_conserved()`.  The ERB products are
  point-sample spectra, so this is trapezoidal integration over the unchanged
  normative Appendix A energy nodes;
- node comparison: the maximum relative change over all ten unchanged
  Appendix A energy nodes, reported together with the controlling energy and
  both ordinate values;
- relative change: `abs(refined - production) / abs(refined)`;
- Stage-4 acceptance threshold: strictly less than 1% for every required
  scalar result.

One cumulative-mode configuration was sufficient for each orbit because one
OST ERB model result contains mean, maximum and mission-total products.  No
second calculation was made merely to relabel the same source result as peak.

## First ladder result: 48x36 to 96x72 (2026-08-23)

| Orbit | Particle | Product | 48x36 | 96x72 | Relative change |
|---|---|---:|---:|---:|---:|
| LEO | proton | maximum flux, cm^-2 s^-1 | 7.996496682e5 | 8.011433461e5 | 0.186443% |
| LEO | proton | mission fluence, cm^-2 | 4.891673518e11 | 4.539859483e11 | **7.749448%** |
| LEO | electron | maximum flux, cm^-2 s^-1 | 4.179745341e6 | 4.525724707e6 | **7.644729%** |
| LEO | electron | mission fluence, cm^-2 | 9.977119373e12 | 1.004725737e13 | 0.698081% |
| GEO | proton | maximum flux, cm^-2 s^-1 | 4.687245210e6 | 4.687534634e6 | 0.006174% |
| GEO | proton | mission fluence, cm^-2 | 3.696620276e14 | 3.696723020e14 | 0.002779% |
| GEO | electron | maximum flux, cm^-2 s^-1 | 2.332148690e7 | 2.332287583e7 | 0.005955% |
| GEO | electron | mission fluence, cm^-2 | 3.185690121e15 | 3.185696839e15 | 0.000211% |
| HEO | proton | maximum flux, cm^-2 s^-1 | 1.354450369e7 | 1.375735394e7 | **1.547174%** |
| HEO | proton | mission fluence, cm^-2 | 3.039846362e14 | 3.022446769e14 | 0.575679% |
| HEO | electron | maximum flux, cm^-2 s^-1 | 3.572926319e7 | 4.009126895e7 | **10.880189%** |
| HEO | electron | mission fluence, cm^-2 | 1.906704542e15 | 1.912458392e15 | 0.300861% |

The production 48x36 grid therefore does **not** satisfy the Stage-4 1%
criterion.  The worst result is the HEO electron maximum-flux integral at
10.880189%.  GEO alone satisfies the criterion for all four required
products.

## Dimension-isolation diagnostic

Two mixed grids were also evaluated for the failed orbits.  These runs are
diagnostic only; the acceptance comparison remains 48x36 versus 96x72.

| Orbit/product | 48x36 | 96x36 | 48x72 | 96x72 |
|---|---:|---:|---:|---:|
| LEO proton mission fluence | 4.891673518e11 | 4.796652083e11 | 4.634360724e11 | 4.539859483e11 |
| LEO electron maximum flux | 4.179745341e6 | 4.221023203e6 | 4.503775819e6 | 4.525724707e6 |
| HEO proton maximum flux | 1.354450369e7 | 1.372492405e7 | 1.354477518e7 | 1.375735394e7 |
| HEO electron maximum flux | 3.572926319e7 | 3.584941204e7 | 3.574663276e7 | 4.009126895e7 |

For LEO proton fluence, both dimensions contribute materially.  The LEO
electron maximum is primarily sensitive to RAAN sampling.  The HEO proton
maximum is primarily sensitive to mean anomaly.  Most importantly, the HEO
electron maximum is found only when both dimensions are refined: neither
96x36 nor 48x72 exposes the 96x72 value.  Since the grids are nested, this
means a new odd-anomaly/odd-RAAN sample locates a stronger peak.

## Discontinuous-sampling risks

The strict OST table evaluator returns zero outside its L range and, for each
selected shell, above that shell's tabulated B/B0 range.  Uniform-grid means
can therefore change abruptly as samples cross a table boundary.  The
following diagnostic coverage fractions changed when the grid was doubled:

- LEO proton: L below table 20.6597% -> 21.1661%; B/B0 above the nearest shell
  69.0394% -> 68.9381%;
- LEO electron: L below table 28.2986% -> 28.0816%; B/B0 above the nearest
  shell 53.7037% -> 53.8918%;
- HEO proton: L below/above table 1.7940%/6.8866% -> 0.8970%/7.1904%, and
  B/B0 above the nearest shell 0.8102% -> 1.9965%;
- HEO electron: L below/above table 2.0833%/3.6458% -> 1.0272%/4.0075%, and
  B/B0 above the nearest shell 0% -> 1.1863%.

`MAXIMUM_FLUX` adds another non-smooth operation: it is the elementwise maximum
over sampled orbit points.  A nested refinement can only retain or increase
that spectrum, but it can reveal a previously missed narrow maximum.  The HEO
electron result demonstrates that risk directly.  Consequently, convergence
of each dimension in isolation is not sufficient.

## Required second ladder: 96x72 to 192x144

The second ladder was run sequentially for LEO and HEO, without competing
heavy processes.  Relative change is always
`abs(value_192x144 - value_96x72) / abs(value_192x144)`.  A value equal to 1%
would fail because the acceptance threshold is strictly less than 1%.

### Conserved integrals

| Orbit | Particle | Product | 96x72 | 192x144 | Relative change | Result |
|---|---|---|---:|---:|---:|---|
| LEO | proton | maximum flux, cm^-2 s^-1 | 8.011433461e5 | 8.095020111e5 | **1.032569%** | FAIL |
| LEO | proton | mission fluence, cm^-2 | 4.539859483e11 | 4.432816637e11 | **2.414782%** | FAIL |
| LEO | electron | maximum flux, cm^-2 s^-1 | 4.525724707e6 | 4.560787988e6 | 0.768799% | PASS |
| LEO | electron | mission fluence, cm^-2 | 1.004725737e13 | 1.010363335e13 | 0.557977% | PASS |
| HEO | proton | maximum flux, cm^-2 s^-1 | 1.375735394e7 | 1.427748646e7 | **3.643026%** | FAIL |
| HEO | proton | mission fluence, cm^-2 | 3.022446769e14 | 3.027356387e14 | 0.162175% | PASS |
| HEO | electron | maximum flux, cm^-2 s^-1 | 4.009126895e7 | 4.010120395e7 | 0.024775% | PASS |
| HEO | electron | mission fluence, cm^-2 | 1.912458392e15 | 1.911976419e15 | 0.025208% | PASS |

The integral criterion alone would pass six of eight spectra.  It is not a
sufficient ERB convergence criterion.

### Maximum change at an energy node

The ordinate units are cm^-2 s^-1 MeV^-1 for maximum flux and cm^-2 MeV^-1
for mission fluence.

| Orbit | Particle | Product | Energy, MeV | 96x72 ordinate | 192x144 ordinate | Maximum node change | Result |
|---|---|---|---:|---:|---:|---:|---|
| LEO | proton | maximum flux | 300 | 7.503306243e-1 | 8.414133080e-1 | **10.824964%** | FAIL |
| LEO | proton | mission fluence | 0.2 | 1.507007172e12 | 1.458406662e12 | **3.332439%** | FAIL |
| LEO | electron | maximum flux | 4.5 | 5.005737801e2 | 5.117973767e2 | **2.192977%** | FAIL |
| LEO | electron | mission fluence | 1.5 | 1.179416281e11 | 1.199276077e11 | **1.655982%** | FAIL |
| HEO | proton | maximum flux | 5 | 5.697855801e3 | 9.214623445e3 | **38.165072%** | FAIL |
| HEO | proton | mission fluence | 300 | 1.162009007e6 | 6.434191718e5 | **80.599065%** | FAIL |
| HEO | electron | maximum flux | 5.5 | 1.421181330e2 | 1.896391874e2 | **25.058668%** | FAIL |
| HEO | electron | mission fluence | 6.5 | 1.219262223e4 | 1.675711913e5 | **92.723915%** | FAIL |

All eight required spectra fail when every energy node is checked.  The large
HEO relative changes occur in the high-energy tail for the mission spectra,
but they are not numerical division by exact zero: both compared ordinates are
finite and positive.  The strict all-node criterion must therefore not be
replaced by the integral criterion.

## Failure localization on the 192x144 grid

For a nested grid, the 96x72 samples are precisely the even-anomaly,
even-RAAN (`EE`) subset of the 192x144 samples.  `EO`, `OE` and `OO` denote
the three inserted midpoint classes.  Reconstructing the model from the four
classes reproduced both preceding calculations.

### Mission-mean branch

The table below uses the energy node with the largest nodewise change.  Each
parity class contains 6912 valid magnetic-field samples.  `Nonzero EE/EO/OE/OO`
counts samples for which the strict OST table evaluation is nonzero.  Support
edges count periodic neighbouring pairs that change between zero and nonzero.

| Orbit | Particle | Energy, MeV | Mean node change | Nonzero EE/EO/OE/OO | Support edges M/RAAN |
|---|---|---:|---:|---:|---:|
| LEO | proton | 0.2 | 3.332439% | 608 / 614 / 621 / 614 | 990 / 730 |
| LEO | electron | 1.5 | 1.655982% | 1326 / 1305 / 1332 / 1301 | 1220 / 628 |
| HEO | proton | 300 | 80.599065% | 26 / 27 / 35 / 35 | 118 / 16 |
| HEO | electron | 6.5 | 92.723915% | 1 / 2 / 10 / 10 | 46 / 4 |

The HEO proton parity means at 300 MeV are respectively
7.364368691e-3, 7.361630042e-3, 8.192249280e-4 and 7.657549125e-4
cm^-2 s^-1 MeV^-1.  The HEO electron parity means at 6.5 MeV are
7.727217675e-5, 8.925424202e-5, 2.038999175e-3 and 2.042482908e-3
cm^-2 s^-1 MeV^-1.  These class-to-class differences show aliasing at narrow
table-support regions; simply averaging more unweighted points is not a
reliable error estimator.

### Maximum branch

| Orbit | Particle | Energy, MeV | 192x144 peak M/RAAN, deg | New-point class | Samples >=90% of peak | Locally refined peak | Change from 192x144 |
|---|---|---:|---:|---:|---:|---:|---:|
| LEO | proton | 300 | 320.625 / 347.5 | OO | 5 | 8.868796401e-1 | **5.126550%** |
| LEO | electron | 4.5 | 266.25 / 137.5 | EO | 14 | 5.196511634e2 | **1.511357%** |
| HEO | proton | 5 | 354.375 / 355.0 | OE | 4 | 1.004186673e4 | **8.237943%** |
| HEO | electron | 5.5 | 350.625 / 20.0 | OE | 6 | 1.955420260e2 | **3.018706%** |

The local check evaluated a 33x33 patch around each 192x144 maximum.  Its
virtual spacing was 0.234375 degrees in mean anomaly and 0.3125 degrees in
RAAN (one eighth of the 192x144 spacing), with half-widths 3.75 and 5 degrees.
The reported local change uses `abs(local - value_192x144) / abs(local)`.
The locally refined peak coordinates were 317.8125/345.3125 degrees (LEO
proton), 268.59375/132.8125 degrees (LEO electron),
354.140625/351.25 degrees (HEO proton), and 350.390625/16.5625 degrees (HEO
electron).  These patches are lower-bound failure diagnostics, not claims that
the local maxima themselves are converged.  Thus 192x144 is not a justified
production replacement: all four locally inspected maximum nodes still
increase by more than 1%.

## Recommended adaptive orbit algorithm

Mean spectra and maximum spectra require different numerical operators.  They
must no longer be produced by one undifferentiated flat-sample reduction.

### Mission mean: periodic adaptive cell quadrature

1. Treat mean anomaly and RAAN as two periodic integration axes on
   `[0, 2*pi) x [0, 2*pi)`.  Mean anomaly remains the time-uniform coordinate;
   every leaf-cell weight is its angular area divided by `4*pi^2`, and all leaf
   weights must sum to one.
2. Seed a 96x72 periodic rectangular cell mesh whose vertices are the current
   uniform samples, and cache each unique orbit point, magnetic trace,
   table-support signature and flux vector.  Use the tensor trapezoidal cell
   estimate `cell_area * sum(four_corner_vectors) / 4`.  Evaluate the nested
   mid-edge and centre points needed to split a parent into four children; the
   four-child estimate is the sum of the same area-weighted rule on them.
3. Refine a cell when any particle/energy node changes support status, changes
   the selected OST interpolation stencil, or when the vector difference
   between the parent estimate and the sum of its four child estimates exceeds
   its allocated error budget.  Strict zero outside table support is retained;
   no interpolation or extrapolation is performed across that boundary.
4. Sum absolute cell error estimates per particle and energy node.  Continue
   until the estimated relative error is below 0.25% at every nonzero node and
   a complete successive adaptive level changes every spectral node and
   `integral_conserved()` by strictly less than 1%.  Require two consecutive
   passing levels to guard against the parity cancellation observed here.
5. If the maximum depth or point budget is reached first, return an explicit
   `INCOMPLETE` result.  Exact zero on both levels is a pass; zero on only one
   level is a failure and must not be hidden by an absolute tolerance.

### Maximum flux: periodic adaptive extremum search

`MAXIMUM_FLUX` is not an integral and must not use the mean-spectrum
quadrature weights.

1. Use the global nested 96x72 to 192x144 evaluation as the discovery level.
   For every particle and every Appendix A energy node, retain all periodic
   cells containing a local maximum, a support/stencil transition, or a large
   centre/edge-to-corner rise.  Include neighbouring cells so a peak on a cell
   boundary cannot be discarded.
2. Recursively quarter those candidate cells, reusing the same magnetic-trace
   cache.  Track the componentwise maximum and its mean anomaly, RAAN, L shell,
   B/B0 and refinement depth separately for every energy node.
3. Stop only after the maximum at every node changes by strictly less than 1%
   for two consecutive refinement levels and the conserved integral of the
   resulting maximum spectrum also changes by strictly less than 1%.
4. Before normative release, validate candidate selection against at least one
   global 384x288 reference run for LEO, GEO and HEO.  Until that guard shows
   that no off-candidate peak is missed, adaptive maximum results remain
   diagnostic or `INCOMPLETE`.

This split algorithm is safer and usually cheaper than selecting a fixed
global 384x288 production grid.  The present data do not authorize a new fixed
sampling constant.

## Required implementation metadata and tests

Each result must record, at minimum:

- algorithm/version separately for mean quadrature and maximum search;
- seed grid, periodic-axis policy, coordinate weighting and strict OST
  interpolation mode;
- unique traced points, accepted leaf cells, refinement levels, maximum depth
  and minimum angular cell sizes;
- counts of support-boundary, interpolation-stencil and peak-candidate cells;
- per-particle/per-energy error estimates and successive-level changes;
- for every maximum node, peak M, RAAN, L, B/B0 and search depth;
- convergence threshold, the fact that it is strict, final convergence status,
  point/depth budget and any `INCOMPLETE` reason.

Focused tests must cover periodic wrapping, leaf-weight conservation, trace
cache identity, exact strict-support zeros, a synthetic discontinuous mean
integrand, a synthetic narrow off-grid maximum, and all-node plus conserved-
integral convergence.  The Stage-4 LEO/GEO/HEO matrix remains a mandatory
numerical validation, not an ordinary fast unit test.

## Runtime and disposition

The earlier 48x36 plus 96x72 pair took about 49 seconds for GEO and 66 seconds
for LEO.  Its HEO process ran under CPU contention and reported approximately
160 seconds, so those older numbers are retained only as historical context.

The sequential 96x72 plus 192x144 measurements used for the present decision
took:

- LEO: 48.783964 s plus 193.596556 s = 242.380519 s;
- HEO: 122.998920 s plus 494.400209 s = 617.399129 s;
- required pair total: 859.779649 s (14 min 19.780 s).

The separate 192x144 localization took 198.930643 s for LEO and 483.903016 s
for HEO.  The four local 33x33 peak patches took another 23.704446 s and
26.670217 s.  All runs were sequential; the total diagnostic wall time was
1592.987971 s (26 min 32.988 s).

No production sampling constant or ERB physics code is changed by this
measurement.  The 96x72 seed fails, 192x144 is still locally unconverged for
maximum flux, and no passing regression is added for the violated criterion.
The next production change should implement and validate the two adaptive
operators above; until then the ERB Stage-4 spectral convergence gate remains
open.
