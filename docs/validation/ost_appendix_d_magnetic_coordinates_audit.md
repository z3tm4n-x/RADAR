# OST Appendix D magnetic-coordinate audit (2026-10-02)

Source: supplied `ОСТ 134-1044-2007 изм. 2_хор.кач..pdf`, PDF pages
119-121, printed pages 113-115, visually checked in full. This record separates
printed requirements, mathematical checks and implementation conventions.
It is not an official errata sheet or a COSRAD comparison.

## Required two-region model

Appendix D.2.6 explicitly assigns **L <= 5.5 to the inner region** and
**L > 5.5 to the outer region**. Before this change RADAR applied IGRF tracing
and D.32 everywhere: the outer shifted-dipole branch was missing.

`OstErbModel` version `ost_appendix_a_v3` now dispatches through
`ost_l_shell_b_over_b0_for_orbit`. The experimental AE8/AP8 dispatcher is not
changed to this OST-specific two-region prescription. Its common IGRF field
and tracer do receive the numerical corrections described below; its version
is therefore incremented to `ae8_ap8_radbelt_asc_v2`, with the solver recorded
in method metadata. That does not qualify the experimental profile for release.

| Formula | Checked implementation | Qualification |
|---|---|---|
| D.21 | `L = rm / (R_E cos(lambda_m)^2)` | Geometrical shifted-dipole L, not IGRF D.32. |
| D.22 | `B = g M0 / rm^3 sqrt(1 + 3 sin(lambda_m)^2)` | `M0=8.06e25 G cm^3`; convert km to cm before cubing, then G to nT. |
| D.23 | `rm = norm(r_geographic - r_offset)` | Offset 430 km at latitude 15.5 degrees, longitude 150.5 degrees E. |
| D.24 | `sin(lambda_m) = dot(r_geographic-r_offset, magnetic_axis)/rm` | Axis latitude 79 degrees, longitude 291 degrees E. Geographic latitude and longitude are explicit, not ambiguous Greek variable names. |
| D.25-D.28 | IGRF spherical harmonic synthesis from `B = -gradient(V)` | Printed component expressions are inconsistent with this identity; see below. |
| D.29-D.31 | Cartesian `dr/ds = +/- B/abs(B)` | Equivalent field-line equations, without a geographic-pole coordinate singularity. |
| D.32 | `L = (0.311653 / B0_gauss)^(1/3)` | Applied only to the selected inner IGRF branch. B0 is the traced minimum, not the field at the initial point or the last unfinished step. |

For the outer branch `B0 = g M0/(L R_E)^3`, using the same moment and units
as local B. Thus `B/B0 = sqrt(1+3 sin(lambda_m)^2)/cos(lambda_m)^6`.
The scale g cancels from both coordinates consumed by the radiation tables.
The standard describes g as a time correction but does not give g(t) or a
reference epoch here. RADAR uses explicitly labelled `g=1` reference absolute
fields; this is not a claim about the measured field at launch. Changing g
in the helper changes both absolute fields proportionally, but neither L nor
B/B0. `M0/R_E^3=0.311653010733 G`, consistent with D.32 to its precision.

### Boundary selection is a declared convention, not a resolved ambiguity

D.2.6 gives the inequalities but not a pilot classifier when the two candidate
L values fall on different sides of 5.5. RADAR uses the analytical
shifted-dipole L to select the branch. It preserves exact `>5.5` versus `<=5.5`
comparison of that computed number; it applies no empirical transition band,
blending, clamping or artificial continuity correction.

Every inner point has both L candidates available. For outer points with
`5 <= L_dipole <= 6`, the IGRF candidate is also computed for diagnosis;
`compare_outer_igrf=True` extends the comparison elsewhere. This diagnostic
band is an engineering choice, not a region boundary or normative tolerance.
Candidate disagreement is recorded as an incomplete reason. A failed inner
trace is never replaced by a successful outer value. A failed optional outer
cross-check retains the valid outer value but adds its own diagnostic reason.

All OST products retain the separate boundary-policy validation blocker.
Successful numerical tests alone cannot approve the pilot classifier.
Model metadata records the selection rule, branch counts, diagnostic counts,
number of candidate comparisons and maximum absolute L-candidate difference.
All point-specific reasons propagate to mean, peak and cumulative spectra.

The justification in D.2.5 for an approximately constant L along the field
line (within 1%) is worded for L<5, although D.2.6 extends the inner algorithm
through 5.5. RADAR does not interpret that statement as a proven 1% bound in
the whole 5..5.5 interval. The historical configured IGRF epoch remains 1985;
its suitability for a normative release is not newly established by this
audit. Neither launch-year substitution nor a fabricated secular moment law
was introduced. When coefficients are explicitly injected, metadata records
their actual epoch separately from the configured loader epoch; a custom
2026 set is not falsely labelled 1985.

## IGRF component identities and polar correction

The [official NOAA description](https://www.ncei.noaa.gov/products/international-geomagnetic-reference-field)
defines B as the negative gradient of the harmonic potential. The
[official IGRF-14 synthesis program](https://www.ngdc.noaa.gov/IAGA/vmod/igrf14.f),
particularly lines 1126-1177, provides an independent primary implementation.
RADAR's Schmidt semi-normalization and ordinary-point component synthesis
agree with that identity. Literal copying of printed D.25-D.27 would instead
lose the radial `(n+1)` factor and introduce inconsistent angular derivatives.
No official OST errata is supplied as evidence for these inconsistencies.

The actual code defect was setting the azimuthal component to zero on the
geographic axis. For m=1, `P(n,1)/sin(theta)` has a finite nonzero limit.
It is now evaluated as `dP(n,1)/dtheta * cos(theta)` on that axis, as in the
official synthesis. At other points `sin(theta)=hypot(x,y)/r` and `cos(theta)=z/r`
feed the Schmidt recurrence directly, avoiding cancellation near both poles
and the loss of a tiny angle on converting through `theta ~= pi`.

Independent Cartesian dipole example at the north pole, r=R_E:
`g10=-30000`, `g11=-2000`, `h11=5000` nT require
`B=(2000,-5000,-60000)` nT. The previous code produced `(2000,0,-60000)`.
Regression expectations are derived analytically, not copied from production.

## Tracing and numerical convergence

Previously, exhaustion of `max_steps` returned `valid=True` without finding
the minimum. On the axial reference dipole at r=2 R_E, colatitude 60 degrees,
one step returned L=1.87193158 while the analytic value is 8/3. That state now
returns an invalid sample with a specific reason and blocks all affected
spectral products. Zero field, sub-domain starting points, entry into Earth,
and failed minimum refinement are also explicit failures.

The midpoint step was replaced by fourth-order Runge-Kutta at the existing
default step 0.06 R_E. After the first increase of B brackets a local minimum,
bounded scalar minimization refines the position along the field line.
Near an initial minimum the bracket includes both directions. No claim is
made that an arbitrary non-dipolar field cannot have multiple local minima;
COSRAD comparison and magnetic-coordinate applicability remain separate work.

Independent tests use analytical Cartesian dipoles/quadrupoles, degree
3/7/13 potential gradients evaluated with SciPy's separate associated-Legendre
implementation, exact polar limits through degree 13, and analytical dipole
field-line identities. At r=1,2,4 R_E and latitudes +/-20,40,60 degrees,
the default-step relative B0 error is below 1e-6; halving the step reduces
it by approximately the expected fourth-order factor. Tests also cover
near-initial minima, both field directions, max-step failure, and below-domain
input. These are numerical validations, not full ERB spectrum convergence.

Inverse construction of shifted-dipole points verifies D.21-D.24 independently
at L=2,5.5,6.6, latitudes -40,0,35 degrees and three magnetic azimuths.
Separate dispatcher tests explicitly cover 5.5-epsilon, exactly 5.5 and
5.5+epsilon, including candidate disagreement and failure propagation.

## Consequences for existing validation records

Earlier ERB orbit-grid convergence numbers used the old all-IGRF mapping and
are historical evidence of the former implementation, not acceptance results
for version v3. They must be regenerated after the magnetic-coordinate and
trajectory conventions are settled. Orbit-grid convergence, energy-integral
convergence, source-table support and magnetic-coordinate method validity are
separate checks; passing one must not erase the other release blockers.

## Real-coefficient control sample

The bundled IGRF-14 coefficients for epoch 1985 were evaluated on an 8x6
mean-anomaly/RAAN grid for each planned orbit. This deliberately small sample
is a coordinate diagnostic, not an orbit-flux convergence result.

| Orbit (km, degrees) | Inner / outer points | Invalid points | L range |
|---|---:|---:|---:|
| LEO 400/400, 51.6 | 48 / 0 | 0 | 1.010625-5.435144 |
| GEO 35786/35786, 0 | 0 / 48 | 0 | 6.573622-6.911198 |
| HEO 500/39700, 63.4 | 10 / 38 | 0 | 1.027228-9.192727 |

For all 58 inner points, the step was successively halved from 0.06 to 0.03
to 0.015 R_E. Maximum relative changes (fractions, not percentages):

| Sample | L, first halving | L, second halving | B/B0, first halving | B/B0, second halving |
|---|---:|---:|---:|---:|
| LEO | 3.36857e-7 | 2.16390e-8 | 1.01057e-6 | 6.49169e-8 |
| HEO | 1.86126e-7 | 1.13291e-8 | 5.58379e-7 | 3.39873e-8 |

The worst LEO point is r=6771.2 km, geographic latitude 51.6 degrees,
longitude -30 degrees: L=3.136456393, 3.136455337, 3.136455269 respectively.
The worst HEO point is r=6871.2 km, latitude 0, longitude 120 degrees:
L=1.027227736, 1.027227927, 1.027227939. Eight additional epoch-2026 points
(four LEO, two HEO, two GEO) completed without failure; maximum inner B/B0
relative changes were 7.05052e-7 and 4.42778e-8 on successive halvings.

### Actual discontinuity between the two models

Points constructed at shifted-dipole `L=5.5 +/- 1e-8` isolate branch switching
from machine rounding. The changes below are `100*(outer-inner)/inner`, not
tracer-error estimates. The magnetic azimuth is measured in the independently
constructed frame in `tests/test_erb_ost_magnetic_coordinates.py`.

| Magnetic latitude | Azimuth (rad) | Inner IGRF L | Inner IGRF B/B0 | Change in L | Change in B/B0 |
|---|---:|---:|---:|---:|---:|
| 0 deg | 0 | 5.551186854 | 1.000692672 | -0.922088% | -0.069219% |
| 0 deg | 0.9 | 5.557352833 | 1.001678241 | -1.032017% | -0.167543% |
| 0 deg | pi | 5.527226554 | 1.000482878 | -0.492590% | -0.048265% |
| 30 deg | 0 | 5.626817224 | 3.324663874 | -2.253800% | -5.683540% |
| 30 deg | 0.9 | 5.392732639 | 2.848985590 | +1.989110% | +10.063921% |
| 30 deg | pi | 5.617093264 | 3.325883641 | -2.084588% | -5.718131% |

The outer values are L=5.50000001 and B/B0=1 or 3.135705258 for the two
latitudes. Model differences greatly exceed the measured integration error.
No COSRAD agreement, guaranteed continuity, or official correction is claimed.

Inverse construction at nominal L=5.5 produces a floating-point pilot L
between 5.499999999999998 and 5.500000000000004 in these orientations.
Consequently an exact geometrical boundary is also sensitive to rounding.
The exact comparator is unit-tested separately; no unreported tolerance or
rounding-based smoothing is added. The unresolved boundary-method blocker
continues to apply, including to nominal-boundary points.

## Verification record

The expanded regression over all ERB test modules, calculation integration
and Stage-4 acceptance/runner checks passed 404 tests in 677.72 s. After final
metadata/version updates, the targeted model/dispatch/epoch tests passed
98 tests in 5.87 s; the independent magnetic-coordinate set was also rerun.
These are overlapping checks, not additive counts or a claim that the whole
repository suite was rerun on the final tree. Numerical and normative gates
remain distinct, as described above.

## Reproducible COSRAD-comparison preparation

`radar.validation.erb_magnetic_boundary` now emits a deterministic diagnostic
JSON matrix for explicitly selected IGRF epochs 1985 and/or 2026:

```powershell
.venv\Scripts\python.exe -m radar.validation.erb_magnetic_boundary `
  --epoch 1985 --epoch 2026 `
  --output outputs\stage4_erb_magnetic_boundary_1985_2026.json
```

There are 18 points per epoch: nominal dipole L=5.49, 5.5, 5.51; magnetic
latitude 0 and 30 degrees; azimuth 0, 0.9 and pi radians. Each record retains
the geocentric radius/latitude/longitude, both IGRF steps, shifted-dipole
candidate, actual production selection, incomplete reasons and the rounding
of nominal equality. Input-file and loaded-coefficient SHA-256 fingerprints
make the coefficient set explicit. Failed candidates retain reasons and JSON
null values, not substituted zeros.

No COSRAD data were supplied: its values are null, comparison_performed=false,
numerical_gold=false and normative_export_allowed=false. The file is a set of
inputs and RADAR results for a later independent comparison; it does not claim
that a particular COSRAD interface can import it directly.

The generated two-epoch matrix has 36 points, no failed numerical candidates,
14 IGRF selections and 22 dipole selections. Seventeen selections have
point-specific diagnostics. Repeated runs were byte-identical. In this matrix
the largest model difference in B/B0 is **24.045589%** at epoch 2026,
L_dipole=5.49, magnetic latitude 30 degrees, azimuth 0.9 rad. Geographic input:
r=25901.172658 km, geocentric latitude 23.4226204402 degrees, longitude
-21.4645927995 degrees. IGRF gives B/B0=2.527865181882445; the fixed reference
dipole gives 3.1357052575580324. This is a same-point model difference, not an
error against COSRAD and not the earlier infinitesimal boundary-jump test.

The 2026 run is diagnostic only: it does not change the model's configured
1985 epoch. It illustrates why epoch choice and model junction cannot be
resolved by reducing the tracing step or forcing the two values to agree.

## 2026-10-30: historical AE8/AP8 regression fixture correction

The broad test run exposed three failures in
`tests/test_ae8_ap8_external_regression.py`: proton differential flux,
proton integral flux, and fluence calculated from the frozen proton mean.
The electron comparison passed. The fixture had continued to apply historical
v1 spectral numbers to the current v2 RK4/refined-minimum geometry.

The six `EXTERNAL_REFERENCE_*` arrays entered Git in commit
`44cd50e57f5b6671ccece84922c1e1db11a80383` on 2026-06-27. Its original test blob
is `9043b4436f0b104e553ae1727555b84e760a93cd`; the contemporaneous
`src/radar/erb/field.py` blob is `8552a17b00733ad527bddda0838fb58fe9b26ac9`.
That commit retains no external program version, original exported result,
or reference-generation procedure. The arrays are therefore described as a
**historical v1 regression fixture**, not independently verified external or
normative validation. Their original numbers and tolerances remain unchanged.

The configured comparison orbit is circular, 2000 km altitude, 0 degrees
inclination, sampled at 2 anomaly nodes and 2 RAAN nodes, with IGRF epoch 1985.
Executing the historical tracer gives these two distinct coordinate pairs,
each appearing twice in the original four-node order:

| Longitude | Historical L | Historical B/B0 | Current v2 L | Current v2 B/B0 |
|---|---:|---:|---:|---:|
| 0 | 1.3674757917475509 | 1.0315326949341908 | 1.3675753675364946 | 1.0317580513804068 |
| pi | 1.2922934613873873 | 1.0372708039953007 | 1.2922459623185105 | 1.0371564315529824 |

RADBELT discretizes its coordinates as `nl=int(L*Lscale)` and
`nb=int((B/B0-1)*Bscale)`. At longitude zero the AP8 `Bscale=2048` changes
`nb` from 64 to 65. The AE8 `Bscale=1024` leaves `nb=32` in both versions.
The other integer coordinates are unchanged. This explains why the proton
outputs change while the electron outputs still match. The maximum relative
differences from the historical proton arrays are 0.003660077978418989 for
mean differential flux, 0.004655331522269428 for peak differential flux,
and 0.002163624596198122 for integral flux. These are implementation-version
differences, not discrepancies against a newly verified external solution.
This orbit is far below L=5.5; it does not test the two-region junction.

The October 30 transport optimization was excluded in a separate diagnostic
process: restoring the old linear-scan `_interpolate_strict` gave bitwise
identical current spectra, and that interpolation function received zero
calls in this unshielded calculation. Replacing only the tracer with the
historical implementation made all four original regression checks pass.
Using the current field synthesis with that old tracer gave the same result,
isolating the change to field-line tracing/minimum refinement.

The historical backend fixture now temporarily supplies the four original
L, B/B0, local-B and minimum-B samples while calculating that one regression
case. Its monkeypatch is explicitly scoped to geometry and restored before
returning the result. No map, solar parameter, spectral ordinate, production
default, or field algorithm is changed. A separate unmocked v2 integration
test checks solver/version metadata and retains the provisional-data blocker;
two further unmocked tests check current proton/electron fluence against
their own current mean flux multiplied by mission duration. Analytical and
step-refinement tests of the actual field tracer remain separate.

Verification: the file previously contained four tests; it now contains seven
(four unchanged historical checks plus three current-v2 contract cases), all
passing in 0.91 s. An AST comparison confirmed all six gold arrays and all four
original test bodies, including tolerances, unchanged. The independent
`test_erb_field_trace_accuracy.py` set also passed 35 tests in 1.05 s. Ruff
passed for the corrected test file. The broad run that had already imported
the earlier module still has its three recorded historical-fixture failures;
these targeted reruns do not retrospectively make that full run green or
establish normative readiness for AE8/AP8 or OST.
