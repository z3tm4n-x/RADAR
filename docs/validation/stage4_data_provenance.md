# Stage 4 physical-data provenance

This note records what is reproducible today and what remains ineligible for a
normative OST 134-1044-2007 release.  A successful numerical calculation is not
by itself evidence that an input table is release eligible.

Absolute user-profile and attachment-directory identifiers are intentionally
replaced by local placeholders in this public note; source filenames and
hashes are preserved. Substitute the local interpreter/data locations before
running the example commands.

Current update (2026-11-01): unchanged native NIST ESTAR 1.2 now reproduces all
81 electron/Al rows exactly. Only that resource's provenance gate is closed;
see [the retained build and release-boundary evidence below](#native-nist-estar-12-electronal-reproduction-2026-11-01).
Earlier dated six-finding snapshots remain historical, not the current audit.
The later same-day proton interpolation passport correction changes no
calculated value or eligibility; the current manifest release is
`radar-normative-2026-11-01.2`, documented in the final section.

## TENDL-2025 proton-on-aluminium data

The retained source is the official TENDL-2025 ENDF-6 evaluation for
`p + Al-27`, EVAL-MAY25 REV1 by A. J. Koning:

- source URL:
  `https://tendl.imperial.ac.uk/tendl_2025/proton_files/Al/Al027/lib/endf/p-Al027.tendl`;
- licence declared in the source header: CC-BY-4.0;
- byte count: `2,712,771`;
- SHA-256:
  `f2218940baef62cf9fb72ed3ab4a0326e69f92faabb0b7ee69fed7b0bb4c41a7`.

An independent download from that URL was byte-for-byte identical to the
retained file.  The source is bundled as
`src/radar/data/normative/tendl/p-Al027.tendl.txt`.  The deterministic command

```text
python -m radar.nuclear.tendl tendl/p-Al027.tendl.txt <output_dir>
```

reproduces the bundled MF3 cross-section, MF6 proton-yield and MF6 outgoing
proton-distribution CSV files after canonical LF normalisation.  MF3 and MF6
incident-energy interpolation is linear-linear as declared by the ENDF data;
the outgoing-energy MF6 density is represented as an ENDF histogram-left
distribution. In the retained MF6 TAB2 records the incident-energy code is
`INT=2`, so the ENDF-6 rule is simple Cartesian linear-linear interpolation at
fixed emitted energy; it is not corresponding-point (`INT=12`) or unit-base
(`INT=22`) interpolation. This interpretation follows the primary
[ENDF-6 Formats Manual, ENDF-102](https://www.oecd-nea.org/dbdata/data/manual-endf/endf102.pdf),
sections 0.6.2 and 6.2.2. Duplicate MF3 incident-energy coordinates are retained as typed
one-sided discontinuities rather than collapsed by a maximum: the first value
is the left limit and the second is the right limit. For MF3/MT5 this preserves
zero cross section below 30 MeV and the 0.7151679 barn right limit at 30 MeV,
without inventing a linear branch from 0.25 to 30 MeV. The automated test also checks all three generated files
against their registered hashes.  These three derived resources are therefore
`verified_derived` and release eligible.

Primary-proton nonelastic survival uses MF3/MT3, which the retained TENDL header
explicitly identifies as the non-elastic cross section. MF3/MT5 remains paired
only with its own MF6/MT5 `(p,anything)` yield/distribution, alongside the other
retained MF6 channel MT values. The current diagnostic operator couples one
production generation to MT3 survival of the primary to the production depth
and MT3 survival of the emitted proton from birth to the aluminium exit. It
does not yet iterate higher proton generations after a secondary nonelastic
collision. A nonzero contribution whose production or either survival path
leaves the 0--200 MeV TENDL incident-energy support remains diagnostic and is
explicitly `INCOMPLETE`; it is never silently extrapolated.

The retained MF6 LAW=1 section declares `LCT=3`: outgoing proton energy and
angle are given in the centre-of-mass system. It also contains Kalbach angular
parameters. The compact MF6 CSV currently preserves only the angle-integrated
`f0(E_cm)` energy-density term. It neither transforms centre-of-mass energy and
angle to the laboratory system nor integrates the resulting path length. Every
positive-thickness secondary-proton spectrum is therefore diagnostic and
carries separate stable blockers for the missing CM-to-LAB transformation,
angular/path-length transport and higher collision generations. The eventual
operator is a one-dimensional depth-and-angle transport calculation, not 3D
spacecraft geometry; its precise plane-parallel or centered-sphere assumption
must be declared before release. At zero shielding no secondary interaction or
path exists, so the exact-zero identity does not acquire these blockers or any
TENDL/stopping-table provenance.

For `POINT_SAMPLE` incident spectra the secondary operator also retains an
explicit incident-energy quadrature blocker: a bin-centre response multiplied
by an incident bin weight is not an integral of the response over that bin.
Resolving birth-depth and outgoing-energy quadrature does not resolve this
independent approximation. On the 2026 five-year GCR reference, the isolated
secondary component at 0.01 g/cm2 changes by about 7.31% on 81-to-161-point
refinement even while the primary-dominated total changes by only 0.256%.
The small secondary fraction must not be used to declare that operator closed.

### Proposed incident-energy quadrature — not implemented

Retain all current diagnostic blockers until implementation and validation.
Extract the single-incident-proton response `K_j(E0)` as exit-bin counts and
compute `N_j = integral phi(E0) K_j(E0) dE0`, dividing by the declared exit-bin
width only after integration. For `POINT_SAMPLE`, reconstruct `phi` linearly
on the true intervals `[x_i, x_(i+1)]`; do not invent logarithmic half-bin
tails outside `[x_0, x_n]` or discard positive-to-zero source segments.

Split at the union of source and range-table nodes, MF3/MT3 thresholds and
one-sided discontinuities, both independent MF6 yield/PDF incident grids,
and support boundaries. Include each valid full-depth mapped knot
`R^-1(R(k)+t)` and the stopping boundary `R^-1(R_min+t)`.
Preserve inner depth splits `d=R(E0)-R(k)` and all outgoing histogram edges.
Use open Gaussian quadrature with refinement checks; preserve supported
contributions and explicit reasons for unavailable domains without extrapolation.

Independent analytic regressions: use `R(E)=E`, `S=1`, `t=0.1`, source nodes
`(1,2,3,4,5,6,7,8)` and values `(0,0,0,5,6,7,0,0)`, whose integral is `18`.
With constant total cross section `sigma_tot`, proton-production cross section
`sigma_r`, unit yield and a normalized uniform birth PDF on `[1.5,2.5]`, an
exit grid covering all residual energies gives
`N = n*sigma_r*1e-24*exp(-n*sigma_tot*1e-24*t)*1.8`, where `n=N_A/A_Al`.
Adding a production threshold at `E=3.5` replaces `1.8` by exactly `2069/1200`;
this checks both threshold knots `3.5` and `3.6`. Inserting collinear source
nodes must preserve these answers. These are synthetic operator tests.

The current union contains 172 transport knots, 103 within reaction support;
naive nested refinement can multiply runtime. Compute all exit bins together,
cache responses independently of source amplitude/mode, and skip only proven
zero contributions. Start with an explicit 0.1% internal integration target,
test secondary counts separately, and report `INCOMPLETE` on unmet tolerance
or evaluation-budget exhaustion. This proposed correction does not close
reference-frame, angular/path-length, higher-generation or data-support issues.

## Aluminium range and silicon LET tables

The immediate source of the five current CSV resources has been identified as
the local workbook `ЛПЭ.xlsx`:

- byte count: `522,688`;
- SHA-256:
  `d4ed7928331a0b7abcdfe1b585504b136809a3f6823f220edfed8967c28ec684`;
- relevant sheets: `e_Al`, `p_Al`, `p_Si`, `i_R_Al`, `i_Si_e`, `i_Si_n`.

Running `python -m radar.shielding.stopping_tables` reproduces the proton and
ion CSV files exactly after LF normalisation.  The electron aluminium table is
numerically identical; only non-semantic decimal rendering differs.

This establishes the immediate import path, but not by itself the authoritative
upstream origin of the workbook data.

An independent live query of NIST SRD 124 ESTAR on 2026-08-11 used
`POST https://physics.nist.gov/cgi-bin/Star/e_table-t.pl` with
`matno=013&ShowDefault=on&Energies=`.  All 81 aluminium rows match the bundled
electron table exactly in energy, total stopping power and CSDA range.  The
canonical 81-row response projection has 2,482 bytes and SHA-256
`10985be5a504bb0b7dbf290bb7b47e619fe46f3e6e81615b887fefcb41d104f7`.
That live response has not been retained because NIST identifies Standard
Reference Data as a possible exception to its general public-domain policy.
The separately downloadable STAR 1.2 archive at
`https://physics.nist.gov/PhysRefData/Star/STAR.tar.gz` carries explicit
redistribution permission on the official download page.  The archive has
444,183 bytes and SHA-256
`b8241beb0e721b6b89dd316e32b23ac6c80e9a8908bd43da1570216de51de2c6`.
At the initial review the archive had not yet been built because no Fortran
toolchain was available in that validation environment, so the electron table
remained provisional. This specific reproduction gap was closed on 2026-11-01
using unchanged sources and a repository-local compiler, as recorded below.
The live-web comparison alone was not used to promote the table.

The proton workbook sheets are not exact NIST PSTAR copies.  On common energy
points, the aluminium range differs from PSTAR by as much as 13.2861%, and the
silicon proton stopping values differ by up to 1.8999%, depending on the PSTAR
column used.  They must not be relabelled as PSTAR data.

After the 2026-11-01 native ESTAR reproduction:

- electron aluminium range is `verified_derived`, with unchanged numerical
  data and a retained official archive/build lineage;
- proton aluminium range and HZE aluminium range remain `provisional`;
- proton silicon LET remains `provisional` because the source sheet contains
  collisional and radiative stopping only, while the nuclear contribution in
  the generated table is zero;
- none of those three provisional resources is eligible for normative release until the
  upstream source, version, units and coverage are independently verified.

The HZE silicon table is `quarantined`.  All 15 represented ions have a
non-physical 3.5--4.2 decade collapse of electronic LET immediately above
10 GeV total ion energy.  Exact regeneration proves that the defect is already
present in `ЛПЭ.xlsx`; it is not a CSV-converter or line-ending defect.  The 270
affected rows must not be smoothed, extrapolated through, or accepted for a
normative result.  An older local workbook ends at 10 GeV, but has no evidence
that would make it an authoritative replacement.

### Candidate sources for independently regenerated HZE tables

Primary-source review on 2026-10-02 identified two candidates for a new,
versioned dataset. Neither establishes the upstream provenance of `ЛПЭ.xlsx`,
and neither changes the current table statuses.

- [GSI ATIMA](https://web-docs.gsi.de/~weick/atima/) documents an energy domain
  of 1 keV/u to 450 GeV/u and projectile charges through Z=120, encompassing
  H--U and the required 1 MeV/n to 100 GeV/n domain after an explicit energy
  and mass convention is established. GSI supplies a C interpolator and
  approximately 1.5 GB of precalculated spline data; the
  [installation instructions](https://web-docs.gsi.de/~weick/atima/atima_readme.txt)
  describe GCC/make. The published Python libraries target Linux; GSI also
  identifies LISE++ as a Windows implementation. Native Windows compilation
  of the standalone interpolator has not been verified here. GSI requests
  attribution, but the reviewed pages do not establish an explicit licence
  for redistribution of the archive or derived tables.
- [CATIMA, the author's repository](https://github.com/hrosiak/catima), provides
  C++/CMake source under
  [AGPL-3.0](https://github.com/hrosiak/catima/blob/master/LICENSE). The author
  explicitly states that its physics is not a complete copy of ATIMA.
  [Author-published Python distributions](https://pypi.org/project/pycatima/)
  included Windows x86-64 binaries at the review date; no local installation
  or build was performed. Any candidate must pin the source commit, build
  options, effective-charge model, low-energy model and correction switches.
  Library integration and redistribution of generated data require their own
  licence assessment.

Unit conversion must be verified against the selected interface. The
[official ATIMA example](https://web-docs.gsi.de/~weick/atima/atima.txt) uses
MeV/u and mg/cm2; its printed whole-ion stopping power multiplies the raw
`dEdXin` by projectile mass. The
[CATIMA manual](https://github.com/hrosiak/catima/blob/master/docs/catima_manual.md)
uses MeV/u, g/cm2 and MeV cm2/g. Converting the latter stopping power to
MeV cm2/mg requires division by 1000. The
[CATIMA physics manual](https://www.isotopea.com/webatima/v15/catima.pdf)
distinguishes electronic and nuclear contributions to total stopping.
The range calculation requires total stopping; the LET table must explicitly
match the required normative definition, including whether electronic or
restricted stopping is intended, rather than silently substituting the total.

Before replacement, retain source hashes and generation parameters; specify
every `(Z, A)` and material; verify the entire energy domain, including the
10--30 MeV/u model transition; check positive stopping and monotone ranges;
and test `dR/dE_total = 1/S_total` (or `dR/de = A/S_total` when
`e = E_total/A`). Compare independent reference values in their supported
domains and rerun transport/LET conservation and grid-refinement checks.
Agreement with selected workbook rows is a cross-check, not proof of their
origin or of normative eligibility. The regenerated tables need a new source
passport and an explicit mapping to the OST transport and LET requirements.

## Audit state

The profile-specific OST audit inspects only the 15 resources actually used by
the OST calculation path.  GOST draft data and AE8/AP8 maps remain visible in
the global audit but cannot block the OST profile merely by existing.

After TENDL restoration the global audit had 11 unresolved findings rather
than the original 14. After the electron/Al provenance upgrade it has 10.
The OST audit now has three non-release provisional transport/stopping
findings and two errors for the quarantined HZE silicon LET resource: five
errors in strict mode, or three warnings and two errors in non-strict mode.
These are data-audit findings, not a count of all open Stage 4 tasks.
Independent per-product release
checks still prohibit normative export for every non-release input, including
warnings.

## Open evidence

- Closed on 2026-11-01: rebuild the redistributable official STAR 1.2 archive
  and verify all 81 electron/Al rows. No other table inherits that status.
- Obtain and retain an authoritative source for the proton range/stopping
  tables; the present values must not be represented as exact PSTAR data.
- Obtain a verified ion-transport/LET source covering the required ions and
  energy domain, then regenerate the HZE tables without repairing the current
  values in place.
- Capture isolated COSRAD NEK helium outputs, with COSRAD version, inputs,
  solar-activity values and export precision, before resolving the
  `b_c(He)=6.6e-2` versus `6.6e-3` question.
## Diagnostic incident-energy quadrature implementation (2026-10-02)

The proposed secondary-proton outer integral is now available explicitly as
`incident_quadrature=IncidentEnergyQuadrature()` in
`calculate_secondary_proton_spectrum_through_al`. It is not enabled by default
in any source pipeline. The original production diagnostic retains its
unresolved incident-bin quadrature warning.

For POINT_SAMPLE input, the new option integrates
`integral phi(E0) K_j(E0) dE0`, where K is the vector of exit-bin counts for
one incident proton. Linear source interpolation is restricted to the actual
source intervals; no artificial half-bin tails are added. It splits at source,
range, MF3 threshold/jump, both MF6 incident grids, and full-thickness mapped
knots `R^-1(R(k)+t)`. Existing depth, outgoing histogram and attenuation
operators are retained. A per-call response cache excludes source amplitude.

Open GL4 is compared with GL4 on the two half intervals. The default relative
L1 count-difference target is 0.001, with bounded evaluations and refinement.
This is an error estimator for the outer integral only, not a rigorous bound
on transport physics or other quadratures. Metadata retains tolerance,
estimated absolute/relative errors, evaluations and unresolved interval counts.
Budget exhaustion is explicitly incomplete. Missing range intervals retain
their blocker and use `converged_on_supported_domain` rather than an unqualified
converged status. BIN_AVERAGE input cannot be silently reinterpreted as point
samples. Peak/fluence semantics and source incompleteness are preserved.

Independent synthetic integral checks, including a one-sided reaction jump,
match `18 * 0.1` and `2069/1200` after removal of the independently known
reaction/attenuation factors, to relative precision 1e-10. Collinear source
node insertion does not change those integrals. Real TENDL and Al range data
were also tested with an explicitly synthetic triangular source over 30-40 MeV,
0.01 g/cm2 Al and two depth steps. Nine- and eleven-node grids both gave
`9.363749374166274e-4` exit counts per the declared source normalization, using
96 vector responses per case; estimated relative outer error was approximately
1.21e-7. Each case took approximately 18-19 seconds in the isolated measurement.

This narrow test is not the full GCR 81/161-grid comparison and does not close
depth convergence. Its low-energy exit bin still has a range-domain blocker.
CM-to-laboratory conversion, angular/path-length transport and higher-generation
cascades remain explicitly absent. Cost and full-source convergence must be
evaluated before considering any default-pipeline change.

### Adversarial review: missed narrow response is not convergence

An additional independent counterexample found that agreement of two zero
GL estimates can miss a narrow positive secondary response. For a synthetic
`R(E)=E`, constant stopping, reaction threshold at 5 MeV, birth PDF over
0.39-0.3901 MeV, thickness 3 g/cm2 and a linear source from (7,0) to (8,1),
the initial implementation returned zero with a zero error estimate.
Collinear insertion of the point (7.99,0.99) exposed a nonzero contribution.
This is a failure of sample-based error detection, not evidence of a physical
zero. Zero estimates on a nonzero source must therefore either have an
analytically proven zero reaction over the reachable energy range or remain
unresolved after bounded refinement.

The same counterexample exposed an independent depth-quadrature limitation:
birth-PDF edges crossing exit-bin edges introduce depths
`d* = t - [R(E_birth) - R(E_exit_edge)]` not covered by the existing reaction
and range knots alone. Even after detecting the outer response, 128 depth
steps gave about 2.48% error relative to the independent analytic integral.
This limitation belongs to the shared inner operator, not just the new option;
both paths must retain an explicit unresolved depth/output-crossing blocker.
Fully resolving it requires the depth crossings and their coupled incident
knots `R^-1(R(E_reaction_knot)+d*)`, with controlled computational cost.

The false-zero guard is now implemented. An exact zero is accepted only when
MF3 times yield is zero on every reachable, knot-free energy interval; PDF
absence or sampled zeros are not proofs. Otherwise refinement is bounded by
both the global budget and `max_unproven_zero_refinement_depth` (default 2),
then reports unknown error and incomplete quadrature if still unresolved.
The independent counterexample is retained for both time modes in
`tests/test_proton_incident_zero_response.py`. Both positive-thickness paths
record missing depth/output crossings and unestablished depth accuracy.
Zero-shield identity acquires no new artificial blocker.

After this review correction, 101 secondary/proton-shielding tests (including
SEP, GCR and ERB wrappers, actual TENDL and adversarial zero-response checks)
passed in 43.03 s. This is not certification of the still-incomplete physical
operator or of its full-source numerical accuracy.

## Subsequent crossing-partition correction

The missing depth events and their coupled incident-energy knots have now
been implemented with explicit work budgets. The earlier narrow-response
counterexample agrees with its independent analytic integral at 1, 2, 8 and
128 depth steps. Large real-data partitions still use a clearly marked,
incomplete bounded fallback; their general accuracy is not closed.
See `secondary_quadrature_crossings.md` for the derivation, version changes,
tests and cost measurements.

The opt-in `per_output_bin` variant now computes each original exit bin on its
own complete crossing partition, with one shared response-work budget and
per-partition diagnostics. The global-union reference and production defaults
are retained. Exact reaction/PDF-support reasoning can certify geometrically
zero bins, but unavailable PDFs or out-of-domain paths cannot be used as zero
evidence. Independent tests cover analytic counts, nonlinear depth refinement,
both time modes and explicit partial results on budget exhaustion.

A full-TENDL, thin-shield synthetic-source comparison gives identical bin
counts and a locally measured reduction from 68.0142 to 8.30912 seconds. The
case uses 0.0001 g/cm2, not the required production-thickness matrix; it closes
neither full-source convergence nor physical transport omissions. The report
`secondary_bin_quadrature_benchmark_2026-10-02.json` contains the input/code
hashes, actual vectors and qualification.

Exact active-PDF pruning of the crossing partitions is now implemented as a
third explicit strategy, with the previous two retained as references. Its
derivation, independent boundary/cancellation/zero-proof checks, actual
0.01-g/cm2 narrow-source depth-refinement pilot and ten-case real-GCR
work forecast are recorded in `secondary_active_cells.md`. The forecast
constructs complete partitions but performs no secondary integration. It
shows that full-source work still exceeds the response budget by a wide
margin. The subsequent 2026-10-29 opt-in pre-quadrature classification now
separates supported analytic zeros, unsupported omissions and evaluable
intervals before constructing expensive crossing partitions. Missing physical
contributions remain unknown, with unavailable full-result error estimates;
per-bin input weights are not secondary yields. Its derivation and scoped
verification are recorded in the final section of `secondary_active_cells.md`.
Full-source/depth refinement remains open; unavailable high-energy
contributions are not assigned zero, and production defaults are unchanged.

## Local COSRAD reference installation, 2026-10-30

The user identified an existing installation at
`<LOCAL_SOFTWARE>/cosrad (2025)`. The folder name is not itself
evidence of a particular physical-model revision. Windows file metadata for
`Cosrad.exe` reports FileVersion `4.3.1.1`; the bundled six-page COSRAD 4 user
manual was created in 2017 and the one-page description in 2016. Both PDFs
were read visually, including the result-file table and its time-unit note.
No newly calculated COSRAD reference has yet been obtained.

An isolated working copy was made at
`outputs/cosrad_reference_20261030/run_base`. All 70 files (15287592 bytes)
were verified byte-identical to the original after the attempted launch.
Original files were not edited. Selected SHA-256 identifiers:

- `Cosrad.exe`:
  `67788fbd16e5eb2260490bcc4d922ad1d502df0f7bc0fd5d50a4f20f19a4e17a`;
- `GCR2.EXE`:
  `f39b7eb7df6c9d5d96af0bd9228714d42694767ccb12bd5e1dde336e6e7ae952`;
- `tab_gcr.dat`:
  `5e8847ade875bebc6e8650916ac18c50db162e68a64b47690a8e400d2973d40c`;
- `Tab#gcr.dat`:
  `a37d6e54e84d4438c7c7b379b79d931481cab2a6e07c17927e81a37d33e5fa82`;
- six-page user manual:
  `e00088e86550b09b36dd5db7764fe830246ac745d1d142ffa7cde48a1b10fc97`.

The retained text headers distinguish:

- `gfx.txt`: maximal orbit flux, particles/(MeV/nucleon cm2 day);
- `gfxm.txt`: maximal peak flux, particles/(MeV/nucleon cm2 s);
- `gfn.txt`: mission fluence, particles/(MeV/nucleon cm2).

The first two are different temporal statistics, not simply the same array
in different units. Each file has 50 energy rows and 93 columns: energy,
then Z=1..92. He is the third column, but it is total GCR He, not a separate
NEK or AKL component. Printed values have limited precision, not an external
1e-12 reference. The grid contains 10 and 100 MeV/nucleon but uses 31.6 and
316 instead of 30 and 300; any interpolation policy must be declared.

The existing GCR headers describe start year 1 **from the beginning of the
solar cycle**, ten flight years, a mean even cycle, circular 36000-km orbit
and zero inclination. Other retained output files describe a different
scenario (e.g. `dsum_x.txt`: 426 km, 51.7 degrees, start year 4). These files
must not be presented as a single fresh consistent run.

There are also two different 92-row, four-column GCR parameter tables. In
the apparent `(A,C,gamma,alpha)` structure, the He rows are respectively
`(4,3690,2.77,3.12)` and `(4,1720,2.64,3.12)`. The second He row agrees with
RADAR's OST parameters, but that entire table does not: its H gamma is 2.73,
versus 2.93 in RADAR. Which table the executable actually reads has not yet
been established. Neither file exposes `b_c(He)`.

Launching the isolated GUI copy through the approved Computer Use mechanism
timed out awaiting app-access approval. No COSRAD process remained and no
fresh computation is claimed. Do not bypass that approval or infer a new
reference from the old text files. Next steps require approved execution,
verification of the active parameter table and solar/time configuration,
fresh separately identified outputs, and a justified isolation of NEK He.
The current `6.6e-2` coefficient and `pending_cosrad_comparison` status remain
unchanged; the available total-He columns are not sufficient to select
between `6.6e-2` and `6.6e-3`.

## COSRAD article and comparison policy, 2026-10-30

The user supplied the seven-page article by N. V. Kuznetsov, Yu. M. Malyshkin,
N. I. Nikolaeva, R. A. Nymmik, M. I. Panasyuk, V. M. Uzhegov and M. V. Yakovlev,
«Программный комплекс COSRAD для прогнозирования радиационных условий на
борту космических аппаратов». Printed pages are 72-78; PDF pages are 1-7.
All seven pages were read as text and visually. The supplied scan does not
establish the publication year/issue or the model revision of the installed
executable; those are not inferred from the folder name or bibliography.
SHA-256: `6c1fe8b8d272265b4f4383af7f159acee23db592dbdf32db9011747a560ef81d`.
Source file:
`<LOCAL_ATTACHMENTS>/1-Программный-комплекс-COSRAD-для-прогнозирования-радиационных-условий-на-....pdf`.

The user explicitly requires treating COSRAD as a fallible comparison
implementation, not as the calculation truth standard. The following is a
source-to-action map, not an assertion that every historical article claim
describes the installed 4.3.1.1 binary.

| Article evidence | Consequence for RADAR Stage 4 |
| --- | --- |
| PDF p. 2 / printed p. 73 distinguishes mission fluence, a per-day statistic (worst-year mean for GCR/ERB; orbit-averaged peak for SEP), and maximum local flux per second. | Do not replace one statistic with another by multiplying/dividing by 86400 or mission time. Preserve the actual output header and source-specific statistic. |
| PDF pp. 2-3 describes a mean solar cycle, Wolf-number interpolation for ERB, and different adopted Wolf maxima in the compared programs. | Record the actual solar history, averaging and cycle parity. A cycle-relative year is not calendar year 2026; matching orbit/lifetime alone is insufficient. |
| PDF p. 3 / printed p. 74 explicitly discusses uncertain low-energy GCR-associated proton modelling and cites Zeldovich et al. (2003), bibliography item 8. | This is a lead to primary evidence, not a value for helium `b_c`. Neither this article nor total-He spectra select between the two candidate coefficients. The cited paper has not been reviewed in this continuation. |
| PDF pp. 3-4 contrasts the probabilistic COSRAD SEP peak model with event-based alternatives; the article describes a different OST peak prescription. | Check the actual installed SEP model against the OST text before demanding agreement. Do not replace the OST implementation simply to match COSRAD, or treat the article's summary as a correction to the standard. |
| PDF p. 4 / printed p. 75 attributes shielding data to ESTAR, SRIM and SHIELD and includes secondary protons and neutrons in spherical aluminium. | Names of upstream programs do not establish exact table provenance, revisions, grids or interpolation. They do not qualify RADAR's quarantined LET table or TENDL secondary operator. Match primary and secondary contributions separately before interpreting protected-spectrum differences. |
| PDF p. 5 / printed p. 76 reports roughly a factor-two dose difference from flight data in a particular example and discusses a hemispherical shielding interpretation. | Preserve geometry/angular normalization as explicit assumptions. Do not apply an unexplained factor of two to spectra; this example is neither a universal COSRAD error nor a general accuracy bound. No 3D geometry is added to RADAR. |
| PDF pp. 5-6 describes an exponential SEE cross section, thin-layer/cube choices and conversion of heavy-ion sensitivity to proton sensitivity. | These historical implementation choices do not supersede the user's approved SEE plan or the supplied 2025 STO. Dose/SEE work remains outside the present Stage-4 change. |

The exponential SEE expression on PDF p. 5 was checked visually because text
extraction lost its exponent. The printed expression has the form
`sigma(L) = sigma_sat exp(-10 L_c/L)`; no threshold step or angular integral
is printed in that displayed equation. This is not evidence to omit either
from the subsequently applicable STO calculation. The article provides no
numerical `b_c(He)` value, uncertainty envelope for every output, source-code
identity, complete transport tables or release acceptance procedure.

Comparison protocol:

1. Pin executable, active input/data files, raw outputs and user-entered
   configuration; distinguish a fresh run from retained output files.
2. Establish the same quantity, source/component, solar/time model, orbit,
   shield geometry/material and energy/LET conventions. Record unknowns.
3. Compare unshielded sources, orbit filtering and shielding separately;
   do not diagnose a coefficient using a mixed total spectrum.
4. Preserve native grids and printed precision. Interpolation, conversion
   from directional to omnidirectional flux and component subtraction require
   an independently justified operator, not an assumed scale factor.
5. Classify differences as input/model mismatch, numerical implementation,
   supported correction or unresolved. Agreement is only cross-code evidence;
   disagreement is not automatically a RADAR defect. Do not tune constants
   for agreement. Independent numerical and normative gates remain in force.

After the user was ready to approve access, a second automated launch failed
to expose a targetable window. The user subsequently opened the program and
supplied a screenshot showing `Система COSRAD 4.3.1.1`; it establishes that
the GUI is open, despite the automation inventory returning no COSRAD window.
The screenshot shows 426/426 km, 51.6 degrees, argument of perigee 0,
cycle-relative start year 1, mean activity and flight time 1 year. These are
not the canonical Stage-4 scenario and no fresh calculation is claimed.
The current obstacle is obtaining targetable UI access or a controlled manual
export, not evidence that the executable cannot run. No guessed window
handles or screenshot coordinates are used to bypass that obstacle.

## Strict diagnostic import of retained GCR files

`src/radar/validation/cosrad_gcr_text.py` reads only the observed ASCII
50-energy, 92-ion orbital GCR format. It determines the statistic from the
content header, not `gfn/gfx/gfxm` filenames, retains cycle-relative start year
and native units, and selects ions by ordered atomic-number identifiers.
The native labels (including `Se34` and apparent symbol typos) are preserved,
not silently repaired. Unsupported variants, missing rows/columns, inconsistent
statistic/unit pairs, nonfinite/negative values and non-increasing energy
grids are errors; there is no padding, sorting, extrapolation or conversion.

Every parsed table remains `independent_model_comparison_only`, with
`normative_release_eligible=false` and freshness, executable revision and
active parameter table `not_established`. Parsing does not identify the
program that actually produced retained bytes. He is explicitly labelled
`total_gcr_he_not_isolated_nek`; this reader does not export a NEK-He reference
or infer a directional flux from an orbital flux. The generic NEK JSON loader
still relies on a separately reviewed component-isolation/transcription
procedure; checking a file hash is not that procedure.

The three retained files have the following byte-level identifiers; these
are old output files, not a new run or a physical correctness baseline:

- `gfn.txt`: `5f65e04d65e6094b8245f5f50e1384ad169699f453e0ee397385dc9c455bad50`;
- `gfx.txt`: `133b67fb526668fc9799aa64f514f79c576069bf4ff6fc742d2d756796d3b9a1`;
- `gfxm.txt`: `8b4c9334a0d65b4582748294551a0de74bfcd3f7422834f69a5216d9425fb5e6`.

After the launch/access investigation, all 70 original files and working-copy
files were re-compared by SHA-256 and remained pairwise identical. No fresh
output has replaced these files. Neither physical formulas nor production
coefficients were changed by this comparison-policy/import continuation.

Independent review found and corrected decimal underflow in the new reader:
nonzero `+/-1E-9999` must not silently become zero (or bypass the negative-value
check as negative zero). Such values now fail explicitly. `raw_data_lines`
preserves the exact numeric spelling/spacing and hence printed precision,
including the distinction between `1.0` and `1.00`; parsed floats alone do
not preserve that distinction.

Final focused verification for this continuation: 212 tests passed in 1.42 s
across the COSRAD reader, NEK comparison, GCR parameters/components/model/source
spectra. JUnit: `outputs/pytest_cosrad_review_20261030.xml`. Ruff passed on both
changed/new modules and their tests; strict mypy passed on both modules;
`git diff --check` passed. All three retained outputs were re-read using the
new parser and their hashes above confirmed. This is scoped regression and
format-validation evidence, not a new complete-suite run, a fresh COSRAD
calculation, coefficient validation or Stage-4 completion.

## First manual COSRAD run: actual inclination differs, 2026-10-30

After the user reported completion, the working directory
`outputs/cosrad_reference_20261030/run_001_gcr_geo_cycle1` contained 24 files
whose bytes differed from the 70-file prepared inventory. GFN, GFX and GFXM
were among those changed files. Every file was copied, with before/copy/after
SHA-256 agreement, into the separate capture
`outputs/cosrad_reference_20261030/capture_001_actual_inclination_51p6`.
The original prepared manifest is retained unchanged. The post-run inventory,
hashes, actual scenario and bounded audit are in
`outputs/cosrad_reference_20261030/run_001_postrun_audit.json`.

All five relevant headers (GFN/GFX/GFXM, orbit and cutoff) declare **51.6 deg**
inclination, rather than the requested **0 deg**. The 51 orbit samples contain
latitudes from -51.5 to +51.5 deg, independently corroborating the inclined
trajectory. The other GCR declarations match: 35786/35786 km, argument 0,
cycle-relative year 1, mean/even cycle, one flight year. This is a fresh
inclined-circular diagnostic, not the requested equatorial GEO comparison.
It does not establish why the GUI input differed; no formula defect is inferred.

All three GCR tables pass the strict reader and have matching grids/scenarios:
50 energies from 1 to 79400 MeV/nucleon and 92 atomic-number columns. Each
contains 4250 positive cells and 350 printed zeros; zero columns are
Z=84,85,86,87,88,89,91. These printed zeros do not establish physical absence
or adequate species coverage. Fresh proton fluence is positive throughout;
the inherited old proton-zero pattern is not used to diagnose this run.

Two independent calculations checked native temporal relationships, without
rewriting any spectrum or modifying RADAR's time model:

- Across the 4250 positive cells, the median GFN/GFX ratio is
  364.5833333333 days. Assuming ordinary rounding to the last printed decimal
  place, all cells admit a common ratio only in the intersection
  [364.5621181263, 364.5990922844] days. This contains 31500000/86400, but
  neither 365 nor 365.25. It is evidence consistent with an approximately
  3.15e7-second normalization in this one-year case, **not verified source-code
  provenance for a COSRAD year constant** and not authority to change RADAR.
- There are 669 cells where GFXM exceeds GFX/86400 even after accounting for
  half-last-digit rounding of both values. No inverse inequality survives
  those rounding bounds. These files therefore cannot be treated as the same
  statistic in different time units. At E=1 MeV/nucleon, proton GFN/GFX/GFXM
  are respectively 677000, 1860/day and 0.0463/s in their native differential
  units; total-He values are 976000, 2680/day and 0.0384/s.

The fresh cutoff table reaches a printed plateau of 0.993 from 0.251 GV;
penetration must not be silently set to unity at this altitude. The GCR
protected-spectrum header and `tab_shld.dat` confirm peak-per-second and
one 0.01-g/cm2 shield. Those downstream spectra and dose/SEE products are
excluded from this first comparison.

Source-selection flags remain unverified: `ef.txt` changed and declares the
new orbit/lifetime, whereas `pf.txt` and `sfn.txt` retain the old scenario.
File changes alone do not explain checkbox behaviour. Do not combine these
files into a supposedly coherent new full-source run. Before repeating the
equatorial case, verify the GUI inclination and intended GCR-only selection.
Solar history/lag, active GCR parameter table and NEK-He component isolation
remain unresolved; no model coefficient, formula or release gate was changed.

## Second manual COSRAD run: equatorial pair, 2026-10-30

The user supplied post-run screenshots showing 35786/35786 km, inclination
0 deg, argument 0 deg, cycle-relative start year 1, mean activity and one
flight year. Only GCR is checked on the environment tab. The completion log
is visible. These are observations after the run, not an execution trace of
the checkbox state throughout it. The screenshots do not establish the
internal solar history or a calendar start year of 2026.

All 70 working files were retained in
`outputs/cosrad_reference_20261030/capture_002_equatorial`, with matching
before/copy/after hashes. Thirteen files differ from the first capture. The
inventory and paired native-statistic audit are in
`outputs/cosrad_reference_20261030/run_002_postrun_audit.json`. All 70 captured
hashes were independently rechecked after writing the audit. The first
capture and the original preparation manifest remain unchanged.

GFN/GFX/GFXM all declare the intended equatorial scenario and pass the strict
50-energy by 92-ion reader. `orbit.txt` independently contains 51 points with
constant altitude 35786 km and latitude 0 deg. Native GCR file identifiers:

- GFN: `209a0ada58ce5a444868283bd86b2e42e71474412fec99fde34d83277c38f372`;
- GFX: `63e124f8009929dca5a5e4ada1f2fb8a0e61f7691c29adab3d81f2c5a5adbbd5`;
- GFXM: `caf2f458aa90a8aea72552c8dbd15b0599189d9d49a145ba594ff637ed537341`.

Two independent calculations using the retained decimal spellings found:

- The 4248 positive GFN/GFX pairs admit a common ratio in
  [364.5598194131, 364.5906228132] days under half-last-digit rounding.
  This again includes 31500000/86400 but not 365 or 365.25. GFX remains
  **Maximal orbit FLUX/day**, as labelled by COSRAD; it is not renamed mean
  flux on the basis of this numerical relationship.
- All 4250 positive GFXM cells, equatorial divided by inclined, admit a
  common factor in [1.001978239367, 1.002047082907]. The ratio of printed
  cutoff plateaus, 0.995/0.993, lies inside this interval. This is consistency
  with a common multiplier, not identification of COSRAD's internal algorithm.
- GFN/GFX now print zero for H at 1 and 1.26 MeV/nucleon; GFXM remains
  positive there (0.0464 and 0.0207 in its native per-second differential
  units). The distinction requires review of the statistics and penetration
  operators; it does not by itself prove an error in either program.

Other displayed curves must not be combined into a new full-source result.
`pf.txt` and `sfn.txt` retain the old 36000/36000-km, ten-year scenario.
Conversely, electron `ef.txt`, `ef_x.txt` and `ed_x.txt` changed numerically
despite the GCR-only post-run selection. For example, the 0.05-MeV electron
fluence changed from 3.61e15 to 8.67e15. Checkbox execution semantics are
not established. Neither a timestamp nor a visible old curve establishes
freshness or a coherent scenario for all sources.

### Reproducible cutoff-only comparison

`scripts/compare_cosrad_cutoff.py` validates the observed cutoff format,
cross-checks orbit headers, preserves the native rigidity nodes, and pins
input/model hashes before and after calculation. It rejects malformed,
partial, nonfinite and nonzero-underflowing decimal inputs. Both observed
Fortran header paddings (`=51.6` and `= 0.0`) are supported without changing
the raw header. Output creation is exclusive; existing reports cannot be
overwritten. The final runner hash is
`ae6936d58cfd256742e156b8349b06175ea68770ac091b192eec5593a52ce144`.

Final reports from this runner are
`outputs/cosrad_reference_20261030/cutoff_comparison_equatorial_v2.json`
and `cutoff_comparison_inclined_v2.json`. The earlier equatorial JSON is
retained as a prior runner revision, not silently replaced. Reproduce with:

```powershell
.\.venv\Scripts\python.exe scripts\compare_cosrad_cutoff.py --capture-dir outputs\cosrad_reference_20261030\capture_002_equatorial --inclination-deg 0 --output outputs\cosrad_reference_20261030\cutoff_equatorial_replay_NEW.json
```

RADAR conditions are fixed, not fitted: Kp=3, disturbance enabled,
24 MLT samples, initial RAAN=0 deg, minimum 16-day trajectory and
60/120/240 samples per orbit. The geographic trace is Kepler motion plus
Earth rotation and secular precession, not the separate ERB RAAN ensemble.
COSRAD's Kp, longitude history, averaging convention and internal shadow
operator are unknown. Equal scalar orbit inputs do not resolve these gaps.

| Rigidity, GV | COSRAD equatorial | Current RADAR, 60 samples/orbit |
|---:|---:|---:|
| 0.0501 | 0 | 0 |
| 0.0631 | 0.173 | 0 |
| 0.0794 | 0.753 | 0.994256980109 |
| 0.1 | 0.995 | 0.994256980109 |

The maximum absolute difference on the 55 observed nodes is
0.241256980109 at 0.0794 GV. This is a local penetration-function difference,
not a percentage error in an integrated spectrum or proof of which model
is correct. Increasing RADAR sampling to 120 and 240 changes this equatorial
nodal result by less than 8e-14. That narrow density result cannot establish
model equivalence, general spectral convergence, or L-B / L=5.5 correctness.

### Primary-source check triggered by the pair: Appendix Zh remains open

The supplied OST PDF (SHA-256
`232c84bf7fb6c2d3aec11c44827bcf01a6b6964366a8d0059ff36cd5b143a9e1`)
was checked visually on PDF pages 125, 126, 129, 130, printed 119, 120,
123, 124. This is a focused formula check, not a new complete audit of
table Zh.1. The numerical transcription and model hashes are retained in
`outputs/cosrad_reference_20261030/appendix_zh_review_20261030.json`.

Three issues prevent describing the current operator as a fully verified
literal implementation of Appendix Zh:

1. **Earth shadow:** printed Zh.8-Zh.10 use
   `mu=(1+cos(atan(eta/(1-eta)^2)))/2`, `eta=R_E/(R_E+H)` for H>0,
   with mu=0.5 at the surface. `ost_earth_shadow_factor` instead uses
   `(1+sqrt(1-eta^2))/2`. The latter has an independent geometric meaning
   for an opaque sphere, but the source calls its own expression empirical.
   They are not algebraically identical, and no authorizing erratum was
   found in the inspected repository records. Plausibility alone cannot
   establish a typographical error or justify a silent substitution.
2. **Equality at cutoff:** Zh.7 and the explanation on printed page 124
   require strict `R>Rc*`. The current builder includes equality, `Rc*<=R`.
   The difference is confined to exact equality but matters for a finite
   grid; the existing builder test explicitly expects that equality to pass.
3. **Order of MLT averaging:** current production calculates
   `Rc*=Rc/mean_MLT(delta)` before applying the threshold. The printed
   pointwise correction followed by time weighting does not establish
   equivalence to this approximation. The explicit model version
   `appendix_zh_mlt_average_v1` records an implementation choice, not a proof
   of normative equivalence or a specification of the actual MLT history.

At the code's Earth radius 6371.2 km, the shadow factors are:

| Height, km | Printed empirical Zh.8-Zh.10 | Current geometric RADAR |
|---:|---:|---:|
| 0 | 0.5 | 0.5 |
| 400 | 0.501854385916 | 0.669305746467 |
| 35786 | 0.989353015577 | 0.994256980109 |

These are differences in the shadow factor alone, not demonstrated errors
in complete spectra. COSRAD's printed GEO plateau 0.995 does not match the
literal empirical value either; agreement with COSRAD cannot choose the
normative policy.

A synthetic noncommutation check fixes Rc=0.3 GV, Kp=3 and 24 equal-weight
MLT values 0..23 hours. Independent transcription of Zh.5-Zh.6 reproduces
the pointwise implemented delta values; their mean is 4.2797. At R=0.05 GV,
averaging the individual strict pass indicators gives 7/24, while applying
the threshold after averaging delta gives 0. At R=0.1 GV the corresponding
values are 15/24 and 1. This proves that the operations are not equivalent;
it is not a claimed MLT distribution for the captured spacecraft orbit.

No production formulas, b_c(He), 1.68 correction, source tables or time
normalization were changed in this continuation. These are **open model
qualification issues**, separate from a data-only profile audit. They must
be resolved before normative spectral qualification, even if resource
checks pass. The existing overall release remains blocked; this audit does
not claim a new machine-enforced per-product gate for these three findings.
Next work is a source-backed shadow policy, an explicit pointwise/averaged
MLT contract and strict-boundary regression before any physical change;
additional manual COSRAD runs are not needed to establish these issues.

Final focused verification: **195 tests passed** (6.10 s), including 62
new cutoff-reader/runner tests, existing GCR text/NEK checks and geomagnetic
regressions. JUnit: `outputs/pytest_cosrad_equatorial_final_v2_20261030.xml`.
Ruff and focused strict mypy pass for the two new files; `git diff --check`
passes. Passing legacy physics tests is not independent validation of the
printed formulas: some tests intentionally encode the current geometric
shadow and equality conventions. No full-suite run or Stage-4 completion
is claimed here.

## Pointwise Appendix Zh implementation and release quarantine (2026-10-30)

This continuation supersedes the **implementation state**, not the preserved
measurements, in the preceding Appendix Zh audit. Its `current RADAR` tables
refer to the former `appendix_zh_mlt_average_v1` operator. They must not be
presented as measurements of the new `appendix_zh_pointwise_access_v2` version.
Historical JSON captures and comparisons were not overwritten.

### Source statistic and common orbit-access operator

The supplied OST PDF with SHA-256
`232c84bf7fb6c2d3aec11c44827bcf01a6b6964366a8d0059ff36cd5b143a9e1`
was checked at these specific locations:

- clauses 6.2 and 6.3, PDF pages 14-15 / printed 8-9, and clause 7.3.1,
  PDF page 18 / printed 12: distinct upset/failure assessment statistics;
- Appendix B.2, PDF pages 100-101 / printed 94-95: distinct probabilistic
  SEP fluence and peak-density products;
- Appendix V.2, PDF page 108 / printed 102, and V.3, PDF page 113 / printed
  107: GCR source spectra and referral to Appendix Zh;
- Zh.1-Zh.2, PDF pages 125-126 / printed 119-120, and Zh.7-Zh.10,
  PDF pages 129-130 / printed 123-124: orbit access, time weighting,
  strict cutoff and shadow expressions.

Zh.1 gives one orbit-averaged `psi(R)` multiplying the B/V source spectra.
The checked text does **not** prescribe a separate maximum-of-access
operator for a peak source product. Therefore the same dimensionless
access operator is retained for PEAK and CUMULATIVE; its shared use is
not itself a defect or an additional release blocker. The source peak /
source maximum is not relabelled as a reconstructed instantaneous orbital
maximum. Cumulative fluence is not integrated over mission duration again.
The transformation provenance explicitly records the source-statistic
application and `time_multiplier=none`.

### Implemented change and boundaries of qualification

The new kernel evaluates
`psi(R) = sum_i w_i * mu_i * I(R > Rc_i / delta_i)`, with normalized
non-negative relative weights. With no supplied MLT history, disturbance
correction evaluates an additional equal-weight ensemble of MLT nodes
**inside the indicator average**. It no longer substitutes `mean(delta)`
before thresholding. The exact cutoff equality is excluded as in Zh.7.

Optional `sample_weights` and `magnetic_local_times_hours` allow explicit
per-sample quadrature weights and MLT values. They do not introduce a
mission-duration multiplier. Invalid, nonfinite, mismatched and underflowing
weights are rejected. Explicit MLT is rejected when disturbance correction
is disabled, rather than silently ignored. Compensated bin accumulation
avoids materializing an orbit-by-MLT list for every rigidity node.

The geometric opaque-sphere shadow expression is **unchanged**. Its
disagreement with the printed empirical Zh.9 is unresolved, not declared
an authorized erratum. All OST penetration builders now carry that reason
in their qualification metadata. The default independent uniform MLT
ensemble carries a second reason: it is not a reconstructed mission MLT
history. Supplying actual MLT samples removes only the uniform-ensemble
reason, not the shadow qualification issue.

Both reasons propagate into affected SEP/GCR proton and heavy-ion spectra,
derived GCR integral spectra, component status, model information and
warning details. Main SEP/GCR production-release methods reject those
selected products. The older injected-operator SEP proton path was also
aligned: it retains the operator version and reports unresolved source or
operator qualification without changing numerical values. Generic complete
user-supplied operators retain their existing complete behavior.

The orbit/config builders additionally record the actual orbit sample count,
step and window, requested density and sample limit. If the limit prevents
the requested temporal density, they add a separate `INCOMPLETE` reason.
This makes the cap visible in production spectra as well as external
diagnostics. It does not repair the sampling or change its numerical values.

### Reproducible bounded diagnostics

`scripts/validate_geomagnetic_operator.py` produced
`outputs/geomagnetic_operator_v2_release_check_20261030.json`, SHA-256
`c3e8b0a93422e190b9a11987aa6a519e1bc30d5b3c10f851323ba04515f59ba5`.
Input bytes and all directly used model modules were hashed before and
after the run; their hashes agree. The script writes new files exclusively.
No model coefficient or source spectrum was fitted to COSRAD.
The earlier `geomagnetic_operator_v2_20261030.json` and
`geomagnetic_operator_v2_final_20261030.json` are retained. After adding cap
metadata and correcting the unsupported sampler-docstring claim, the last
run reproduces every case's numerical arrays exactly and retains identical
input hashes, while recording the updated implementation-file hashes. The
filename does not imply release approval: eligibility remains explicitly false.

All cases use Kp=3, initial RAAN=0, unchanged geometric shadow, equal orbit
weights and the production orbit-window policy. The density ladder is
60/120/240 requested samples per orbit at 24 MLT nodes, followed by 48/96
MLT nodes at the highest orbit request. LEO/GEO/HEO use 401 log-rigidity nodes;
the COSRAD cases use their original 55 printed nodes.

| Case | Max absolute new-minus-old psi, baseline | Max absolute psi change, MLT 48 to 96 | Orbit refinement informative? |
|---|---:|---:|---|
| LEO 400/400 km, 51.6 deg | 0.00127517 | 0.00000955 | Yes |
| GEO 35786/35786 km, 0 deg | 0.49712849 | 0.01035684 | Yes |
| HEO 500/39700 km, 63.4 deg | 0.09691786 | 0.00007970 | **No: diagnostic 30000-sample cap** |
| Captured COSRAD GEO, 0 deg | 0.45570112 | 0.00371040 | Yes |
| Captured COSRAD GEO, 51.6 deg | 0.22335506 | 0.00029576 | Yes |

The maximum new-minus-COSRAD absolute discrepancy at the common baseline
is 0.37358939 for the equatorial capture and 0.08692297 for the inclined
capture. Improvement against COSRAD is not required for a source-backed
correction; these differences do not establish which physical model is
more accurate. COSRAD Kp, MLT/longitude history and shadow remain unknown.

All table entries are **absolute dimensionless psi differences**, not
relative particle-spectrum errors. The normalized log-rigidity area in the
JSON is likewise only a numerical diagnostic, not a physical flux integral.
None establishes the Stage-4 requirement of less than 1% change in actual
integrated spectra after grid refinement. For HEO, the requested densities
all produced 30000 samples. This is the diagnostic runner's bounded-cost
override, not the production default of 400000. The existing secular-perigee
window is about 23275 years (734516187341.67 s), with an actual diagnostic
step of 283.38 days: about one sample per 571 orbits. This is a service
averaging window, not a change to mission duration. An unchanged capped
grid is explicitly marked noninformative. Simply increasing the cap is
not an established remedy: window construction, precession units and
representative quadrature require a separate source-backed review.

Focused follow-up establishes the mechanism, not an authorized replacement:
the D.9-D.10 algebra in the sampler matches PDF page 118 / printed 112,
including `E_z=4.1378e5` and `(1-e)`. For this orbit `5*cos(i)^2-1` is
0.002441003711 and the implemented perigee drift is approximately
2.138545554e-11 rad/s, giving a 9310-year perigee-drift period. The **code's
separate choice** to average over 2.5 such periods creates the 23275-year
window. Zh.2 on PDF page 125 / printed 119 instead states that approximately
ten orbits are usually enough and requires segments much shorter than an
orbit. The inspected text does not require 2.5 precession periods.
Even the production 400000-sample limit would yield a step of about 21.25
days (one sample per approximately 43 orbits), so this is not solely an
artifact of the diagnostic cap. The accessible Desktop `ORBIT.py` uses
`num_periods*T`, with `NUM_PERIODS=10` in `constant.py`; it does not establish
the 2.5-period policy. The old sampler docstring's unsubstantiated reference
claim was removed. Numerical precession and window formulas remain unchanged
pending a justified sampling-policy comparison near 63.43495 degrees.

The existing `scripts/compare_cosrad_cutoff.py` was updated to schema v2,
actual pointwise cutoff extrema, operator version, parameters and reasons.
Re-running it no longer labels new numerical output as the old mean-delta
operator. Earlier reports still identify their historical runner hashes.

Remaining work is source-backed shadow qualification, a justified MLT/time
sampling policy, uncapped or demonstrably converged HEO quadrature, and
full physical-spectrum convergence. NEK He `b_c=6.6e-2` remains pending its
independent component comparison; the accepted SEP coefficient 1.68 is
unchanged. This continuation does not close Stage 4 and adds no dose or SEE
model.

### Verification outcome

- Initial full-suite control run: **2132 passed, 2 failed** in 1264.97 s.
  Both failures were old `COMPLETED` expectations for the SEP geomagnetic
  stage, in `test_calculation.py` and `test_pipeline_execution.py`. They were
  updated to the deliberate `INCOMPLETE` qualification, not numerically fitted.
- Final targeted run: **305 passed** in 42.90 s, including pointwise/boundary
  and weighted-access tests, source-statistic/time-unit preservation, selected
  PEAK/CUMULATIVE release rejection, all affected SEP/GCR pipeline routes,
  reader/runner checks, orbit-cap diagnostics and existing L-B / L=5.5
  regressions. JUnit: `outputs/pytest_zh_final_focused_v2_20261030.xml`.
- Final broad repeat: **2132 passed** in 394.96 s. JUnit:
  `outputs/pytest_zh_final_regression_20261030.xml`. This repeat explicitly
  excludes the 11 `test_gcr_upper_domain.py` and 6 `test_sep_upper_domain.py`
  source/transport-domain tests: all 17 passed in the initial full run and
  their implementation paths were unchanged in this continuation. Thus the
  final evidence spans all 2149 currently collected tests across runs; this
  is not claimed to be one fresh 2149-test full-suite pass.
- Ruff passes for the changed Python files; focused strict mypy passes for
  14 source/test/runner files with `--follow-imports=silent`.
  `git diff --check` passes. The four captured input files and eleven
  directly used diagnostic implementation files still match the report hashes.

These checks demonstrate regression behavior and bounded numerical
reproducibility, not physical completeness, normative release approval or
the unperformed full-spectrum less-than-one-percent acceptance matrix.

## Bounded HEO orbit sampling (2026-10-30)

This continuation removes the historical inverse-precession window rule.
It supersedes the preceding sampler state, not its retained diagnostic
reports. The source PDF SHA-256 remains
`232c84bf7fb6c2d3aec11c44827bcf01a6b6964366a8d0059ff36cd5b143a9e1`.
PDF pages 118, 125 and 126 were visually rechecked in full: D.9-D.10 are
unchanged; Zh.2 normally suggests about ten orbits and requires small
time segments, not 2.5 complete perigee-precession cycles.

### Implemented sampling contract

The sampler version is `bounded_orbit_cells_v1`. Its explicit controls are
`min_days` (default 16), `min_orbits` (default 10), `steps_per_orbit`
(default 60) and `max_samples` (default 400000). These defaults are an
engineering starting policy, not a claim that OST mandates sixteen days.

With period `T`, the implementation uses:

```text
dt = T / steps_per_orbit
N = max(min_orbits * steps_per_orbit, ceil(min_days * 86400 / dt))
requested_window = max(min_days * 86400, min_orbits * T)
actual_window = N * dt
sample_times = 0, dt, ..., (N - 1) * dt
```

The actual window consists of complete equal-duration cells, sampled at
their left endpoints; its excess over the requested minimum is less than
one cell apart from floating-point roundoff. Keeping `dt` unchanged preserves
the established circular-orbit sample coordinates. This is a left-rectangle
quadrature, not a claim of exact integration of the cutoff discontinuities.
For explicitly supplied nonuniform weights, the provenance labels that
choice separately rather than calling the quadrature equal-weight.

If `N` exceeds the budget, the sampler raises a descriptive error **before
calculating positions**. It neither increases the step nor shortens the
window. This is an explicit failed calculation, not a partial spectrum
returned with an `INCOMPLETE` tag. Successfully calculated finite-window
spectra retain `INCOMPLETE` until the window's mission representativeness
is justified; this restriction reaches both SEP/GCR release gates.

The trace and spectral transformations record requested and actual windows,
cell count, step, minimum orbit count, initial RAAN and sampling policy/version.
Both source statistics still use the same dimensionless access operator;
there is no new multiplier involving window duration or mission duration.

| Scenario | Samples at 60 / 120 / 240 points per orbit | Default actual window |
|---|---|---|
| LEO 400/400 km, 51.6 deg | 14959 / 29917 / 59833 | 16.00099584 days |
| GEO 35786/35786 km, 0 deg | 963 / 1926 / 3852 | 16.00220803 days |
| HEO 500/39700 km, 63.4 deg | 1936 / 3871 / 7741 | 16.00706780 days |

The HEO orbital period is about 42861.90 s; at 60 points per orbit the step
is 714.365 s, rather than the historical capped step of about 21.25 days.
The same bounded behavior is tested at the critical inclination and its
retrograde counterpart. No division by perigee-precession rate chooses
the new window.

### Independent precession check and limits

The [official JuliaSpace J2 algorithm](https://juliaspace.github.io/SatelliteToolboxPropagators.jl/stable/man/propagators/j2/)
expresses the secular node and perigee rates using `J2*(R/p)^2`, where
`p=a*(1-e^2)`, without the additional `(1-e)` printed in D.9-D.10.
Its corrected mean motion and gravity constants must also be distinguished
from RADAR's retained constants. This establishes a model discrepancy,
not an official erratum or permission to change the OST profile silently.
The printed drift formulas and coefficients were therefore retained.
Removing the unrelated window-expansion rule does not resolve that discrepancy.

### Compatibility replay

`outputs/geomagnetic_bounded_window_20261030.json`, SHA-256
`ee27f02c30545f98c6c278e1d59cad250bc1c97a43cd02ca7c9bf51eb233b7e9`,
repeats the five earlier rigidity diagnostics. All numerical penetration
arrays for LEO, GEO and both saved COSRAD circular cases are exactly equal
to their preceding results. For HEO, the grid genuinely refines instead of
remaining pinned at a sample cap. Earlier HEO curves represent the old
long-window/coarsened sampling and are not numerical reference truth.

The diagnostic schemas were advanced to record the new sampling policy.
The legacy threshold calculation in `validate_geomagnetic_operator.py`
is explicitly a counterfactual old threshold **on the current trace**;
it does not replay the historical orbit sampler.

### Source-weighted window diagnostics

The new runner `scripts/validate_heo_orbit_sampling.py` evaluates the
configured 2026 five-year OST sources once, then keeps them fixed while
varying only orbit/access quadrature. The SEP exceedance probability is 0.1.
The selected products are proton, He and Fe from both SEP and GCR, for
PEAK and CUMULATIVE: twelve differential source products in total. The
diagnostic subset is not a complete normative ion roster.

Rigidities use the production conventions separately: GCR Fe uses `55.8/26`,
whereas SEP Fe uses the B.9 `A/Q=4.12`, not rounded mass divided by atomic
number. The rigidity grid contains every exact source-energy rigidity node.
The compared integral is a finite linear trapezoid between the existing
energy centers, with no tail extension, hidden solid-angle factor or extra
time multiplier. It is not a dose or LET integral, nor a test of the
energy grid itself. Sources retain their actual units, time semantics and
incomplete reasons. The runner pins seven source-resource files and the
RADAR Python implementation before/after the run.

The initial 18-case report is `outputs/heo_orbit_sampling_20261030.json`,
SHA-256 `8772d4fea1f5b05370de6e7a7423f1e93a875368ad4cd96d3263d97ec0912b04`.
Its largest relative integral changes among the twelve selected metrics are:

| Comparison | Largest relative change | Interpretation |
|---|---:|---|
| 60 to 120 points/orbit, requested 16 days | 0.23507% | Density plus a 357.18 s window-rounding change |
| 120 to 240 points/orbit, requested 16 days | 0.10030% | Density plus a 178.59 s window-rounding change |
| 240 to 480 points/orbit | 0.00823% | Actual windows equal in this pair |
| 48 to 96 uniform MLT nodes | 0.00180% | MLT quadrature only, not an observed MLT history |
| 10 to 20 orbits at 480 points/orbit | 1.24430% | Change of averaging window, not grid refinement |
| 20 to 40 orbits at 480 points/orbit | 1.20962% | Change of averaging window |
| 16 to 32 days at 480 points/orbit | 0.36415% | Change of averaging window |
| Initial RAAN 0 to 90 deg, 16 days | 11.76176% | Initial-orientation sensitivity; SEP proton fluence |

For the RAAN 0-to-90 comparison, the GCR metrics change by at most 0.08919%,
but the SEP peak-proton metric changes by 11.37536%. Different source
weights explain why a small GCR integral change cannot establish SEP
stability. This comparison by itself does not identify a formula error.
The retained rate/fluence units are unchanged in either case.

Using the retained D.9 drift and Earth rotation, the Earth-fixed node shifts
by about -179.09949 degrees per orbit. The two-orbit closure residual is
1.80103 degrees, giving a diagnostic ground-track phase scale of roughly
198.32 days. This inference explains why 16-to-32-day stability is compatible
with substantial initial-RAAN sensitivity. It is not a normative requirement
to use a 198-day window, a new precession model, or proof of five-year
representativeness. Longer-window comparisons must be kept separate from
numerical-density convergence and may not silently change production defaults.

### Extended-window replay and remaining qualification

The final 18-case replay with the extended runner is
`outputs/heo_orbit_sampling_final_20261030.json`, SHA-256
`965621864f7bf674f91db6027955e83b3bddb4dbcac20774a519aa2acf519538`.
Its source payload, all variant objects and all comparisons are exactly
equal to the initial 18-case report; only runner/provenance context changed.

The additional report is `outputs/heo_window_extension_20261030.json`, SHA-256
`53fdc756782d360555d49edc929067ed8ca9bd0949b58d5f1006c79656e4f4b6`.
It uses 240 points per orbit and 24 uniform MLT nodes, with requested windows
of 16, 32, 64, 128 and 256 days, separately for initial RAAN 0 and 90 degrees.
The source products are unchanged. This is a phase-coverage experiment,
not a new production-window choice or a long-window grid convergence proof.

| Window comparison | Largest integral change, RAAN 0 | Largest integral change, RAAN 90 |
|---|---:|---:|
| 16 to 32 days | 0.37308% | 0.94124% |
| 32 to 64 days | 4.21495% | 3.41204% |
| 64 to 128 days | 0.93593% | 0.81161% |
| 128 to 256 days | 0.66817% | 0.55537% |

The maximum is among the same twelve finite-energy source-weighted
integrals, relative to the left variant, not a bound over spectral bins.
The RAAN-0 16-to-32-day maximum is SEP He mission fluence; the other
window-comparison maxima are SEP proton mission fluence.

| Requested window | Initial RAAN 0-to-90 difference in SEP proton fluence |
|---|---:|
| 16 days | 11.76329% |
| 32 days | 12.74861% |
| 64 days | 4.49709% |
| 128 days | 2.68789% |
| 256 days | 1.43980% |

Thus a single stable window-doubling pair is insufficient even for these
selected metrics. At 256 days the orientation difference remains above
one percent, and the longer run does not reconstruct the five-year mission.
Neither this difference nor its decrease proves a defect in the source
formula. No default window, RAAN averaging or physics coefficient was
selected by fitting these results. The window qualification reason remains
in both PEAK and CUMULATIVE products and prevents normative release.

An independent read-only review recomputed every stored comparison from
the variant integrals without discrepancy. It also noted that these are
nested-window averages: the shared earlier interval dilutes the change
from the added interval. A small change of the cumulative average therefore
does not bound the difference of consecutive nonoverlapping intervals.
The 240-to-480-point density result above applies to sixteen days, not
to the 256-day runs.

The next obligation is an explicit, physically justified treatment of
mission orientation and temporal coverage, with separate tests of phase
coverage and numerical density. Unspecified initial orientation must not
silently become a proven mission mean. Longer windows alone do not settle
the retained precession, shadow or MLT-model assumptions.

Both final reports pin 135 implementation files and seven input resources,
before and after calculation. All 284 recorded hash checks per report
still match the workspace at handoff. Both reports retain false values for
normative release eligibility, full Stage-4 spectral convergence, complete
source roster and parameter fitting.

### Regression scope for this continuation

- Broad regression: **2185 passed**, zero failures/errors/skips,
  `outputs/pytest_heo_broad_20261030_a.xml` (356.938 s).
  The new diagnostic runner was checked separately; the 17 expensive tests
  in `test_gcr_upper_domain.py` and `test_sep_upper_domain.py` were not rerun
  in this continuation. Their production source/transport paths were not
  modified here. This is not presented as a fresh full-suite pass.
- Final focused sampler/operator/pipeline/diagnostic regression:
  **230 passed**, `outputs/pytest_heo_final_focused_20261030.xml` (35.664 s).
  These overlap the broad run and must not be added as distinct coverage.
- Final extended-runner tests: **19 passed**,
  `outputs/pytest_heo_runner_extension_final_20261030.xml` (1.765 s).
- Final Ruff and strict mypy checks pass for all fourteen Python files
  changed in this continuation (`--follow-imports=silent` for mypy).
  `git diff --check` passes; the existing repository line-ending warnings
  are not errors and no unrelated files were reformatted.

The tests and pinned diagnostics verify implementation behavior and
reproducibility within their stated scopes. They do not close the remaining
physical-model, full-spectrum or normative acceptance obligations.

## Global-time blocks and finite-mission access audit (2026-10-30)

This continuation separates finite-window representativeness from exact
coverage of a specified interval in the retained kinematic model. It does
not promote that model to a reconstructed physical mission history.

### Source and coordinate conventions

The supplied OST PDF was visually rechecked at PDF pages 116-117, 119,
125-126 and 129-130 (printed pages 110-111, 113, 119-120 and 123-124).
D.1 starts time at a reference perigee; D.5-D.7 use the initial node and
perigee orientation. D.18-D.20 describe the Earth-fixed transformation.
Zh.2 and Zh.7 prescribe time-weighted access along a trajectory over the
selected interval. Zh.4-Zh.6 use local time at each point. These passages
do not establish an independent uniform distribution of initial node
angles or MLT values at every trajectory point.

The [ESA Navipedia coordinate transformation](https://gssc.esa.int/navipedia/index.php/Transformation_between_Celestial_and_Terrestrial_Frames)
gives the passive Earth-rotation matrix with first two rows
`(cos(theta), sin(theta), 0)` and `(-sin(theta), cos(theta), 0)`.
In the simplified fixed-pole model this yields east longitude equal to
inertial longitude minus the Greenwich angle. This supports the current
rotation convention, not the literal plus sign printed in OST D.20 or an
official correction to that document. No rotation or drift formula is
changed in this continuation.

In the existing sampler the initial Greenwich angle is implicitly zero,
not derived from the launch year. Consequently `raan_deg` acts as the
initial relative phase `Omega_0 - theta_G0`. Changing only this parameter
rotates the ground track relative to the geographic cutoff table; it is
not a harmless coordinate-origin change. A common change of `Omega_0`
and `theta_G0` must leave the ground track unchanged. The provenance now
states this convention and distinguishes elapsed time from UTC.

[NASA's magnetic-coordinate description](https://omniweb.gsfc.nasa.gov/vitmo/cgmm_des.html)
defines dipole MLT relative to the subsolar magnetic meridian and explains
that other magnetic coordinate systems require care. Therefore geographical
longitude alone does not reconstruct MLT. The existing independent uniform
MLT ensemble remains a diagnostic approximation to the joint trajectory
and local-time distribution, even when every time cell is covered.

### Block arithmetic and interpretation

For a fixed orbital step `dt`, block-local sample `i` uses global time
`t = (start_sample_index + i) * dt`. Kepler motion, secular drift and Earth
rotation all use this same time. A block boundary must not restart the
satellite at perigee, reset the node, or reset the Earth angle. Integer
global indexing also gives identical sample times when a trace is split
into blocks. Initial source spectra and PEAK/CUMULATIVE units are unchanged.

For an exact target interval `[0, D)`, the diagnostic uses `ceil(D/dt)`
left-endpoint cells and shortens only the last cell's quadrature weight to
its represented duration. Generated extra nodes needed to complete the last
whole-orbit allocation are discarded, never assigned mission exposure.
If block `b` represents duration `D_b` and has mean access `psi_b`, the
combined mean is `sum(D_b * psi_b) / D`. This is dimensionless temporal
quadrature, not multiplication of a peak source by mission duration or a
second accumulation of source fluence. All source products remain fixed
while this operator is evaluated.

Nonoverlapping-block comparisons expose temporal variability that nested
averages can hide. Their differences are not quadrature-error estimates:
different blocks contain different orbit phases. Conversely, a density
comparison on the same exact interval tests numerical sampling, not the
physical adequacy of the precession, shadow, MLT or source-time assumptions.

The start index is grid-specific. A doubled density must double a nonzero
start index to retain the same physical start time. The full-duration runs
below all begin at zero, and their common end time is explicitly checked.

### Implemented validation and five-year results

`sample_ost_orbit` and both penetration builders now accept
`start_sample_index=0`. `OstOrbitTrace` carries the index and exposes global
start/end times. Sample indices remain block-local. Default numerical
sampling is unchanged; the additional time-origin version is
`global_sample_index_v1`. Invalid, overflowing or unresolvable time offsets
are rejected before orbital positions are evaluated. The builders retain
the existing mission-representativeness restriction in both source modes.

The new `scripts/validate_heo_mission_window.py` calculates contiguous
32-orbit blocks. The scenario is HEO 500/39700 km, 63.4 degrees, initial
perigee argument zero, 2026 launch year and five model years, with the same
fixed twelve SEP/GCR source products as the preceding diagnostics.
The exact model duration is 157788000 s (1826.25 days), not a reconstructed
UTC calendar interval. The two initial relative node phases are 0 and
90 degrees; Kp=3 and the independent 24-node MLT ensemble are retained.

| Points/orbit | Used temporal cells per orientation | Blocks per orientation | Total orbit-MLT states, both orientations |
|---|---:|---:|---:|
| 60 | 220879 | 116 | 10602192 |
| 120 | 441758 | 116 | 21204384 |
| 240 | 883515 | 116 | 42408720 |

Each run stays within its 50-million-state preflight limit. At 240 points
the largest allocated block contains only 7680 nodes, rather than keeping
the full trajectory in memory. Both orientations cover `[0,157788000)`
without gaps or overlaps; the recorded duration closure error is zero and
the block-duration fractions sum to one. The final used cell has fractional
duration 0.7181865971 of its represented full grid cell. Extra allocated
nodes are not given exposure weight.

The independent `scripts/compare_heo_mission_windows.py` requires identical
source products, input/model hashes, energy/rigidity grids, units, conventions,
physical scenario and exact duration before comparing doubled orbital
density. It rejects nonfinite temporal metadata and reports zero-reference
relative changes as unknown, not zero. It never grants normative acceptance.

| Comparison | Maximum selected integral change, phase 0 | Maximum selected integral change, phase 90 |
|---|---:|---:|
| 60 to 120 points/orbit over the exact five-year interval | 0.145825% | 0.142624% |
| 120 to 240 points/orbit over the exact five-year interval | 0.077438% | 0.078446% |

The first row is set by GCR Fe mission fluence; the second by SEP Fe mission
fluence. These are observed differences of finite-energy integrals of the
twelve selected products, not guaranteed error bounds or full-spectrum
convergence. At 240 points the largest phase-0/phase-90 difference is
0.234860%, for SEP proton mission fluence; the corresponding peak metric
differs by 0.226322%.

For comparison, the earlier 16-day n240 SEP proton integral differs from
the new full-interval integral by -6.194836% at phase 0 and +4.594088% at
phase 90, using the full-interval value as denominator. That comparison
changes the averaging window, not numerical density. It is a counterexample
to treating the default sixteen-day window as automatically representative
of this five-year model scenario. Production defaults were not silently
changed and the release restriction remains.

Consecutive nonoverlapping block metrics still vary, up to 7.44299% and
9.29946% among the selected integrals for the two orientations. These are
variations of the fixed-source access diagnostic, **not fluences produced
during individual blocks**. An independent recomputation of duration-weighted
block metrics reproduces aggregate integrals within 2.08e-16 relative.
Small aggregate phase sensitivity does not justify independently averaging
unknown initial conditions, replacing actual MLT history, or ignoring
correlations between source activity and access.

### Reproducibility and regression for global-time blocks

Final report SHA-256 values:

| Report under `outputs/` | SHA-256 |
|---|---|
| `heo_mission_n60_20261030.json` | `60914ed745a00692f891ec7f2417db6875c79b33774e3c440d7bcc5a6fde36fb` |
| `heo_mission_n120_20261030.json` | `ca2ba23e34f6f94202f5906bb53ff2d8a5623c5b447337cad434c14b4329464e` |
| `heo_mission_n240_20261030.json` | `81cf51f51dde6fc3d102badc715571a2decbc945c0506eda3ad552e7122db3d0` |
| `heo_mission_density_60_120_final_20261030.json` | `a6e6c1e6178f1216d48117bb7c365d046a7a88fc29a6641bc4009558794e2f7d` |
| `heo_mission_density_120_240_final_20261030.json` | `906280aefaee5ca7e1cb283a402d070e1956a6bb92c80aba81dff7be51b23870` |

Each mission report records 136 implementation files and seven input resources
before and after the run. All 286 hash checks per report match the workspace
at handoff. The source payload SHA remains
`ad357b47748db66489e000805e5f4fa9645fbdff5bdf4d295c05472110feb062`.
No source coefficient, transport table or drift formula changed here.
Only `ost_orbit.py` and `ost_penetration.py` changed among the production
Python files pinned by the preceding run. Earlier initial smoke/comparison
reports remain historical artifacts; the table above identifies final evidence.

- Broad regression: **2251 passed**, zero failures/errors/skips,
  `outputs/pytest_global_time_broad_20261030_a.xml` (343.222 s).
  The 17 expensive GCR/SEP upper-domain tests were not rerun; the new mission
  runner, comparator and independent-frame tests were checked separately.
- Independent Kepler-bisection and Cartesian/matrix frame checks:
  **87 passed**, `outputs/pytest_mission_frames_final_20261030.xml`.
  The matrix convention was checked against the independent source above;
  retained drift coefficients are test inputs, not independently validated
  physical rates. Tests span two eccentricities, prograde/polar/retrograde
  orientations and elapsed times through five model years.
- Final mission-runner and cross-report comparator tests: **65 passed**,
  `outputs/pytest_mission_runner_final_20261030.xml`. They cover partial cells,
  aligned and adjacent floating-point boundaries, budget refusal before source
  work, weighted-monolithic equivalence, source units/status, hash changes,
  output overwrite refusal and nonfinite comparison metadata.
- Earlier focused 69 integration and 191 compatibility tests also passed;
  they overlap the broad regression and are not additional distinct coverage.
- Ruff and focused strict mypy pass for the nine changed Python files.

This continuation establishes a bounded way to evaluate and compare the
entire declared model interval. It does not close Stage 4. The next policy
decision is how the public calculation represents initial orientation and
local magnetic time, and when full-duration versus shorter-window averaging
can be justified. Until then, the mission diagnostic is not automatically
substituted for the existing production operator or called normative.

## Local-time convention, trace binding and correlation diagnostics (2026-10-31)

This section records the first local-time investigation. The follow-up
"Exact NASA coefficient match and clock-origin conflict" below establishes
the previously missing coefficient identity, without granting normative release.

### Source boundary: a valid geometry is not a calibrated OST convention

The supplied OST scan was inspected again at PDF pages 129--130 (printed
123--124), equations Zh.4--Zh.9. It calls the input `t_MLT` local time in
hours but does not define its coordinate convention there. The fixed dipole
axis at 79 degrees north, 291 degrees east in D.24 (PDF120, printed114)
belongs to the L--B construction. No explicit link making it the calibrated
axis for the Zh disturbance correction was established from these pages.

Two independent primary sources prevent treating the word "local" or the
variable name as sufficient evidence for a production choice:

- Laundal and Richmond, *Magnetic Coordinate Systems*, section 6, equation 93,
  define MLT using a specified magnetic longitude and centered-dipole
  subsolar longitude. The definition supports a diagnostic centered-dipole
  construction, not automatic identification with OST Zh.6.
  https://link.springer.com/article/10.1007/s11214-016-0275-y
- NASA HEASoft `cornymmik`, referring to Nymmik et al. (2009), explicitly accepts
  local **solar** time. Its equality to the numerical coefficient set of
  OST Zh.6 has not been demonstrated. This is evidence of a real alternative
  convention in a related cutoff method, not permission to substitute it.
  https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/help/cornymmik.html

Thus the convention and calibration remain **UNKNOWN**. No disturbance
coefficient, precession formula, production averaging default, source spectrum
or transport table is changed by this continuation. The coefficient 1.68 and
the pending NEK-He coefficient 6.6e-2 are unchanged.

### Implemented diagnostic geometry and provenance guard

`src/radar/geomagnetic/local_time.py` implements two explicit candidates.
For unit north axis `n`, Sun direction `s`, and position direction `r`,
project the latter two onto the plane normal to `n`. The signed angle is
`atan2(n dot (s_perp cross r_perp), s_perp dot r_perp)`, and the local hour
is `(12 + angle*12/pi) modulo 24`. The centered-dipole candidate requires an
explicit north axis; the geographic solar-direction candidate uses the
geographic north axis. Sunward/east/antisunward/west meridians are 12/18/0/6 h.
Solar-direction hour angle is not asserted to be civil time or mean solar
time, and its physical meaning depends on the supplied Sun vectors.

There is no inferred UTC epoch or ephemeris from a launch year alone, and no
automatic magnetic pole. Zero/nonfinite vectors and ill-conditioned pole
longitudes raise explicit errors. Normalization is scale-safe. A projection
sine threshold of 1e-12 is a numerical conditioning guard, not a physical
cutoff or an approximation that fills polar values with zero.

`LocalTimeHistory` binds its ordered values to a SHA-256 fingerprint of all
ordered orbit sample fields, including global elapsed time and local index.
Factories record the convention, explicit axis, Sun model, time reference,
normalized Sun-vector hash and caller-supplied limitations. A matching hash
is a consistency check, **not proof of authenticity or physical correctness**.
Every history retains a mandatory qualification limitation, including direct
construction with an empty caller reason list.

The existing penetration builders accept an optional `local_time_history`.
It is mutually exclusive with a raw per-sample array; mismatched length,
reordering, changed time or changed coordinates are rejected. Each correlated
history supplies one local hour per orbit point; it is not additionally
averaged over an independent local-time ensemble. Numerical access, strict
cutoff, shadow and dimensionless weighting are unchanged.

An actual diagnostic omission is corrected: a bare array previously removed
the uniform-MLT limitation without recording whether the replacement had a
qualified derivation. Raw arrays now explicitly retain that limitation, and
all disturbed variants retain the unresolved OST convention. Histories carry
their own reasons through builders and source products. Duplicate inherited
reasons are deduplicated before constructing the immutable result, preventing
a false rejection when the history already reports the orbit-window limit.
Both PEAK and CUMULATIVE pipeline tests verify that the convention limitation
reaches production-release checks. Other shadow, mission-window and source
limitations remain in force; this fix does not imply an earlier complete
normative-release bypass.

### Bounded comparison design

`scripts/validate_mlt_correlation.py` compares seven access operators on each
of six traces: quiet, independent uniform24/uniform48, two centered-dipole
histories and two geographic solar-direction histories. HEO 500/39700 km,
63.4 degrees uses sixteen-day requested windows, 240/480 points per orbit and
initial relative node orientations 0/90 degrees. LEO 400 km, 51.6 degrees and
GEO 35786 km, 0 degrees each provide a ten-orbit, 120-point smoke check. Kp=3.
Preflight accounts for 42 operators and 3,761,142 orbit/local-time states.

The two Sun histories are deliberately artificial frozen inertial directions
with zero declination and initial phases 0/90 degrees. Their Earth-fixed phase
is initial phase minus the **same** Earth rotation rate used by the orbit
sampler times global elapsed time. They are not calendar ephemerides or a
five-year solar history. The diagnostic centered-dipole axis 79N/291E is an
explicit borrowed parameter, not an accepted OST time convention.

Comparisons keep distinct: independent local-time quadrature density,
orbital sampling density, initial relative node orientation, artificial Sun
phase, and candidate time definition. Actual-duration differences caused by
whole-cell rounding are exposed. Relative changes use the left/reference
integral as denominator; a zero reference yields unknown, not zero error.
Co-rotation checks rotate the point, Sun and relevant axis together and test
only coordinate geometry, not invariance of the untransformed cutoff tables.

The twelve previously used fixed SEP/GCR p/He/Fe source products remain the
same 2026/five-model-year inputs. Integrals are finite-energy diagnostic
metrics, not full-spectrum convergence or fluences accumulated during these
short diagnostic windows. No extra duration factor is applied to PEAK or
CUMULATIVE. Every report keeps source reasons and records all production
Python files, both imported integration helpers, its own runner and the
seven input resources before/after calculation. Changed inputs invalidate
the run; existing outputs are never overwritten. All normative, history and
Stage4 convergence flags remain false.

### Observed comparison results, not physical error bounds

The final evidence is `outputs/mlt_correlation_final_20261031.json`, SHA-256
`a78280f1194096bcb36e9e4956f340025cb21049a09d8d3115df0e711e57e0a0`.
Its 138 implementation/helper hashes and seven input-resource hashes agree
before/after the calculation. The fixed source payload hash is unchanged:
`ad357b47748db66489e000805e5f4fa9645fbdff5bdf4d295c05472110feb062`.
The earlier unsuffixed report is historical; the final run includes corrected
documentation in `ost_disturbance.py`. Those docstrings now explicitly
distinguish `Rc/<delta>` from `<Rc/delta>` and from pointwise access, and do
not claim that the legacy argument name determines the time convention.
No arithmetic in that module changed.

For HEO, maxima below are over the twelve selected source integrals and
matching pairs on the named axis. Each percentage is
`100*abs(right-left)/left`; it is not an absolute change in psi or a
guaranteed relative error against the unknown physical truth.

| Controlled comparison | Largest observed integral change | Setting/product at the maximum |
|---|---:|---|
| Orbital density 240 to 480, all seven operators | 0.138213% | Quiet, node0, SEP proton mission fluence |
| Orbital density 240 to 480, disturbed operators only | 0.031504% | CD phase0, node0, SEP He mission fluence |
| Independent local-time quadrature 24 to 48 | 0.012911% | n480, node0, SEP proton mission fluence |
| Solar-direction to CD time, identical artificial Sun phase | 1.251137% | n240, node0, phase90, SEP He mission fluence |
| Independent uniform48 to a correlated candidate | 69.423027% | n240, node0, solar phase0, SEP proton mission fluence |
| Artificial Sun phase0 to phase90 | 57.522697% | n240, node90, CD, SEP proton mission fluence |
| Initial relative node0 to node90 | 85.811119% | n480, solar phase90, SEP proton mission fluence |

All fourteen HEO density pairs happen to have exactly the same actual window,
1382474.8845236942 s, because 15482 cells at n480 cover exactly the same
interval as 7741 cells at n240. Thus their actual-duration difference is zero;
the general report guard still records it rather than assuming equality.
The largest disturbed density change compares 10688503844.393488 with
10691871122.418377 cm^-2. These values use a fixed full-mission source
fluence times short-window mean access, not a fluence produced in sixteen days.

The LEO and GEO ten-orbit runs are **smoke checks, not density studies**.
For LEO, the largest uniform24-to48 change is 0.046717% (SEP proton peak
integral); the two time definitions differ by up to 4.344768%, and a correlated
candidate differs from uniform48 by up to 106.335060%. For GEO, corresponding
maxima are 0.302638%, 0.140773% and 0.562444%. The latter two time-definition
maxima use SEP He mission fluence for GEO and SEP proton mission fluence for
LEO. All these comparisons retain the artificial Sun assumptions; they do not
rank the candidates by physical accuracy or qualify either mission history.

Independent read-only replay recomputed all 88 pair comparisons / 1056
integral differences, retaining unknown values for zero references. The
coordinate co-rotation check differs by at most 7.11e-15 h. Crucially,
uniform24 HEO node0 results at n240 and n480 match the preceding
`heo_orbit_sampling_final_20261030.json` exactly for all 1047 psi values and
all twelve source integrals, with identical source payload, grid, step,
window and sample count. New provenance warnings have not retuned the
existing numerical branch.

The experiment establishes that small discretization differences do not
settle a different joint position/local-time assumption. The remaining
source task is to identify a publication or authoritative implementation
with the **same Zh.6 coefficient set** and an explicit definition of its
time variable. Only then can a mission-time/orientation policy be selected
and qualified. COSRAD comparisons can help diagnose it but cannot replace
that derivation. Stage4 remains open.

### Regression and handoff for this continuation

- Broad regression: **2517 passed**, zero failures/errors/skips,
  `outputs/pytest_local_time_broad_20261031.xml`, 335.662 s in JUnit metadata.
  Only the 17 expensive GCR/SEP upper-domain tests were excluded; source and
  transport algorithms were not changed. This is not claimed as a complete
  rerun of every test in the repository.
- The three new local-time/trace-binding/diagnostic test modules contain
  **114 passing tests**, also included in the broad total. Their standalone
  record is `outputs/pytest_local_time_final_20261031.xml`.
- After the final documentation-only disturbance-module clarification,
  **31 focused tests passed**, recorded in
  `outputs/pytest_local_time_post_doc_20261031.xml`.
- Ruff and focused strict mypy pass for the ten changed Python files.
  Tracked-file whitespace checks report no errors. No unrelated worktree
  changes were staged, reverted or committed.
- At handoff, all 290 before/after implementation/resource hash entries in
  the final report match the actual workspace, with zero mismatches.

Current-turn production changes are confined to the new `local_time.py`,
history/provenance handling in `ost_penetration.py`, and clarification of
three docstrings in `ost_disturbance.py`. The numerical formulas in the
existing source, L--B, orbit-drift, cutoff and transport models were preserved.

## Exact NASA coefficient match and clock-origin conflict (2026-10-31 follow-up)

### New primary-source evidence

The official NASA HEASoft 6.37.1 source tree is available as individual files,
not only as a complete distribution. The `cornymmik` task is version 1.0,
dated 2020-12-29. Its source and help, the relevant `prefilter` sources and
library functions are retained under `research/heasoft_nymmik_20261031/`.
The acquisition manifest records exact URLs, byte counts and SHA-256 for
16 files, including the official Attitude subset archive retained as
`outputs/heasoft_attitude_source_20261031.tar.gz`. The separately downloaded
`cornymmik` and `prefilter/derive.c` are byte-identical to their members in
that archive. No downloaded Perl/C program was executed or installed.

Primary source:
https://heasarc.gsfc.nasa.gov/FTP/software/lheasoft/lheasoft6.37.1/src/attitude/tasks/cornymmik/cornymmik

All **18 numerical sinusoid parameters**, including signs and phase shifts,
match the supplied OST scan, printed page 123, Zh.6.1--Zh.6.9. The expressions
for C, A, B, the capped exponential delta and Rc/delta also match Zh.4--Zh.5.
NASA expresses local time in degrees. With `lt_degrees = 15*t_hours`, its
`pi/180*(lt_degrees +/- phase_hours*15)` equals the OST `pi/12*(t +/- phase)`.
This closes the earlier gap between a merely related method and the exact
coefficient set; it does not independently validate the physical calibration.

`cornymmik.html` calls this input local solar time and cites Nymmik et al.
(2009), DOI 10.1134/S0010952509030010. `prefilter.html` defines the producer
column LOCAL_TIME as apparent solar time, in degrees, with noon at 180 degrees
and the Sun/spacecraft sharing a geographic longitude. These are explicit
documentation statements, not deductions from the letters MLT.

- https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/help/cornymmik.html
- https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/help/prefilter.html

### The producer code and its documentation disagree by twelve hours

The inspected source chain is unambiguous about its vector operations:

1. `prefilter_derive.c`, lines 259--275, calls the Earth-to-Sun routine and
   identifies negated spacecraft position as the spacecraft-to-Earth vector.
2. In `derive_local_time`, lines 724--728, it negates the outward geocentric
   position, converts that reversed vector to longitude, then calculates
   `180 + longitude(-r) - longitude(sun)` modulo 360 degrees.
3. `atInvVect.c` really is componentwise negation, and `atVectToPol.c`
   obtains longitude with `atan2(y,x)`, wrapped into [0,2*pi).

For nonzero equatorial projections the resulting expression simplifies to
`[longitude(r)-longitude(sun)] modulo 360`. The documented solar clock is
instead `[180+longitude(r)-longitude(sun)] modulo 360`.

| Relative meridian, spacecraft minus Sun | Documented solar time / RADAR solar helper | Translation of inspected prefilter arithmetic |
|---|---:|---:|
| 0 degrees, sunward | 12 h | 0 h |
| 90 degrees east | 18 h | 6 h |
| 180 degrees, antisunward | 0 h | 12 h |
| 270 degrees east | 6 h | 18 h |

The independent US Naval Observatory definition is apparent solar time =
12 h + the Sun's local hour angle in hours, consistent with the documented
clock and the existing RADAR solar-direction geometry:
https://aa.usno.navy.mil/faq/eqtime ("Apparent and Mean Solar Time").
This supports noon=12 for ordinary apparent solar time; it is not evidence
of the authors' original time convention for the fitted Zh coefficients.

NASA release notes for 6.29 mention a previous LOCAL_TIME bug fix, but do not
specify its formula or invalidate the current 6.37.1 source observation:
https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/release_notes/RelNotes_6.29.0.html
The `ut-cornymmik` test replays a FITS/template comparison; the inspected test
contains no independent noon/midnight assertion.

The twelve-hour conflict is established by source inspection and an explicit
translation of its arithmetic, **not** by running a complete compiled
prefilter/HEASoft pipeline. Solar ephemeris accuracy, J2000-to-date frames and
real mission timing are not validated by these tests. It would be incorrect
to silently introduce this phase shift into RADAR to reproduce another tool.

### Reproducible comparison

`scripts/validate_nymmik_source.py` pins six relevant NASA source/help files.
It uses a restricted literal parser for the six sinusoid strings and checks
their organizing expressions and degree/hour conversion. There is no
`eval`, external execution or arbitrary formula interpreter. Changed source
hashes, changed implementation during a run, unexpected expression structure,
nonfinite output and output overwriting cause explicit failures.

The numerical comparison covers integer Kp=0..9, 96 local-hour values from
0 through 23.75 in quarter-hour increments, and Rc = 1e-4, 1e-3, 0.1, 1, 5,
15 and 100 GV. The reference is evaluated in degrees directly from the
parsed NASA parameters; RADAR is evaluated in hours.

| Check | Cases | Maximum observed difference |
|---|---:|---:|
| C/A/B coefficient tuples | 960 | 1.652e-15 relative |
| Disturbance factor delta | 6720 | 1.652e-15 relative |
| Disturbed cutoff Rc/delta | 6720 | 1.680e-15 relative |
| Solar geometry against documented hour angle | 364 | 7.106e-15 h |
| Source-code clock shift against exactly 12 h | 364 | 7.106e-15 h |

The clock cases comprise four cardinal positions and 360 additional synthetic
longitude pairs. Numerical agreement of coefficient expressions does not
qualify their clock input semantics. Twelve separate phase-shift examples
record effects on Rc/delta with Kp=3, Rc=1/5/15 GV and hours=0/6/12/18;
they are not radiation-spectrum errors or mission predictions.

The source formula has no explicit lower numerical cutoff, whereas RADAR
retains its existing 1e-4 GV evaluation floor. Values below that floor and
Rc=0 are excluded from this numerical identity claim, as are fractional Kp
and noncanonical hours. No existing cutoff arithmetic was changed.

NASA literally labels COR_SAX/COR_NYM as `GeV/c`; RADAR uses `GV` for
rigidity. This diagnostic compares the same numerical cutoff scalar in the
matching formulas. It does not verify a general particle-momentum/charge
conversion or the interpretation of actual FITS columns. The final report
records both unit labels and sets `ion_momentum_unit_conversion_verified=false`.

Final report: `outputs/nymmik_source_comparison_final_20261031.json`, SHA-256
`96dc06b0522f05908fa80b2d326dc2be5595a941ca1568ff924018d3d93d1f8e`.
The unsuffixed report is historical; the final report adds this explicit
unit-label scope without changing any numerical result.
The report pins all RADAR Python files plus its runner, and all six external
inputs before/after the run. Normative release, mission-history qualification,
full-HEASoft replay and production-policy-change flags are all false.

### Scope of implementation changes and remaining decision

Two independently reproduced input-boundary defects were corrected without
changing valid numerical inputs:

- `orbit_samples_fingerprint` now refuses integer sample fields whose values
  cannot be preserved exactly as floats. Previously `2**53` and `2**53+1`
  seconds could collide through numeric conversion. Current bounded mission
  traces were not affected; exact small or large integer/float equivalence
  and signed-zero canonicalization remain supported.
- All five public disturbance helpers now reject boolean numeric arguments.
  The higher-level penetration builder already rejected them, so this aligns
  helper validation rather than changing the physical model.

Forty-one additional tests cover these boundaries. The 42-case local-time
matrix was rerun as `outputs/mlt_correlation_boundary_review_20261031.json`,
SHA-256 `d1819e11f70ce66ed87923381db6e6b919edec86e363cf1601f50973782fad45`.
Its numerical operators, histories, source payload and comparisons match the
previous final report exactly. Earlier hash-pinned reports remain historical;
the new report records the stricter input guards.

The solar-hour-angle interpretation now has an exact matching independent
implementation and explicit documentary support. The remaining question is
the original calibrated convention and trustworthy clock origin, not whether
the NASA coefficients merely resemble Zh.6. Full author texts for 2009/2010
were not obtained in this search; lack of an accessible copy is not proof
that no copy exists. Until this is resolved, neither NASA's producer output
nor an inferred calendar/initial orientation replaces the existing production
policy. No twelve-hour offset is introduced into RADAR. The diagnostic
solar helper keeps its standard noon=12 convention, and all existing
INCOMPLETE/release restrictions remain. Stage4 is not closed.

### Regression and evidence checks for the exact-source follow-up

- Broad regression: **2558 passed**, zero errors/failures/skips,
  `outputs/pytest_nymmik_broad_20261031.xml` (330.83 s console duration).
  It excludes the 17 expensive GCR/SEP upper-domain tests and the new
  source-comparison test module, which was run separately after final review.
- New source-comparison module: **26 passed**. The final focused run covering
  it plus local-time geometry, disturbance inputs and history integration
  totals **167 passed**, `outputs/pytest_nymmik_focused_final_20261031.xml`.
  The other 141 are included in the broad run, not additional distinct tests.
  Thus 2584 distinct selected tests passed across these runs; this is not a
  claim that the 17 excluded tests were rerun.
- Ruff and focused strict mypy pass for all six Python files changed in this
  follow-up. Existing formulas, source/transport data, 1.68, pending6.6e-2,
  PEAK/CUMULATIVE time semantics and production defaults were preserved.
- Independent review verified the pinned source parser and independently
  recomputed the numerical identity and twelve-hour clock discrepancy. The
  scope remains source arithmetic, not a full external executable replay.

## Independent HZE stopping/range candidate (2026-10-31)

### Scope and normative boundary

The supplied OST PDF was visually re-opened at PDF pages 147--149 (printed
141--143), section I.7 and Table I.4. Its SHA-256 remains
`232c84bf7fb6c2d3aec11c44827bcf01a6b6964366a8d0059ff36cd5b143a9e1`.
These pages discuss continuous-slowing-down engineering calculations and
illustrate stopping powers using ITRAN and SRIM. They do not independently
qualify CATIMA or settle the electronic-versus-total LET definition.
The new candidate remains outside production loaders and the normative registry.

This pilot explicitly preserves the legacy transport isotope choices:
He4, C12, N14, O16, Ne20, Mg24, Si28, S32, Ar40, Ca40, Cr52, Mn55, Fe56,
Co59 and Ni58. Each is calculated in Al and Si. This is not complete GCR
Z=2..92 or SEP Z=2..28 coverage. Source-model effective mass, rounded
ParticleSpecies metadata and transport isotope mass are distinct concepts.
Ni58 is not silently changed to Ni59 because the source mass parameter is
58.7. SEP B.9 effective mass-to-charge parameters are not physical projectile
charge states. Existing source models and metadata are unchanged.

### Retained source and explicit configuration

Files are retained in `research/ion_stopping_20261031/`:

- Author package: https://pypi.org/project/pycatima/1.982/ .
- CPython 3.12 Windows wheel SHA-256:
  `24dab5c194e383bdd68df1d6ffbe8e768703169e0178c946232cee239bf1af9c`.
- Source distribution SHA-256:
  `79f239f8c7425b20fcf19dd756b9cf39b24d970423c2c03da7b43b1ea3395971`.
- Publishing commit `17c915cd38da58300e52ab5facb2727629fc55db` declares
  CATIMA v1.91 in CMake. Retained archive SHA-256:
  `c2eabcc1b83d121e82d0cb9a71ed2731198c4e2477492be804fd32bc8e85db4e`.
  Source: https://github.com/hrosiak/catima/tree/v1.91 .

The wheel was unpacked into an isolated research runtime after local pip
target installation failed on temporary-directory permissions. No RADAR
dependency, system installation or application setting changed. The native
library was executed; it was not rebuilt locally. Source-to-binary equivalence
is not independently proved by package hashes. The wrapper's literal
`CATIMA version = 1.7` is hard-coded at `src/_ext.cpp:19`; reports retain it
alongside the package/source identifiers rather than silently relabelling it.

The binary build has GLOBAL disabled. The generator explicitly uses supported
Pierce-Blann effective charge (1), SRIM-85 low-energy mode (0), corrections=0,
calculation=1, scattering=255 and scattering_factor=1. Unsupported GLOBAL or
ATIMA14 charge modes are not selected. This is not the same physics as GSI
ATIMA 1.41. The retained source licence is AGPL-3.0; integration and generated
data redistribution have not been qualified and remain separate decisions.

### Quantities, generation and validation

`scripts/generate_catima_candidate.py` generates JSON program data without
overwriting the quarantined CSV. It pins sources, verifies every unpacked wheel
member, rejects additional files/cached bytecode/preloaded pycatima submodules,
and checks the imported native extension path. Bytecode writing is disabled.
The empty cache directory from the initial probe was preserved outside the
runtime as `historical_python_cache`. No source or historical result was removed.
All 160 declared RADAR Python/resource entries are hashed before and after;
outputs must be new files and finite JSON.

The binding does not expose nuclear stopping. A mathematical replay of pinned
`calculations.cpp:41--61` supplies it, including total energy `1000*A*e` in
keV and the library's scaled Avogadro constant 6.022140857. Four author
He-on-carbon checkpoints test the replay. In the dense run, all 36,030 checks
against native `dedx - sezi_dedx_e` at e<=10 MeV/u agree within 5.552e-16
relative. This verifies source arithmetic, not independent nuclear calibration.

Above 50,000 MeV/u, the routine named `bethek_dedx_e` also adds pair production
and bremsstrahlung. Two otherwise identical native configurations separate:

- full stopping: corrections=0;
- nonradiative stopping: corrections=8, disabling only `no_highenergy`;
- radiative stopping: full minus nonradiative;
- collisional electronic stopping: nonradiative minus nuclear.

All three components are retained in whole-ion MeV cm2/g. Range uses full
stopping. Candidate electronic LET divides electronic stopping by 1000 and
excludes nuclear/radiative terms; this does not yet qualify a production LET
definition. No negative values are clipped or missing contributions zero-filled.

`src/radar/validation/ion_stopping_candidate.py` checks finite values, exact
isotope/material/node coverage, increasing ranges and
`Delta R = integral A/S_total(e) de`, with an independent analytic integral of
piecewise log-log stopping. Stopping/LET need not be monotone. CATIMA sets
R(0.001 MeV/u)=0; absolute range from zero energy is not verified. The increment
check cannot detect a common constant range offset, which is explicitly stated.

The sampled domain is 0.01..100000 MeV/u, including exact 10, 30 and 50000
MeV/u transition nodes. Complete shielding transport still needs a justified
lower-energy exit-domain policy, not extrapolation below this pilot's minimum.

| Points/decade | Nodes | Rows, both materials | Max local range/stopping discrepancy |
|---|---:|---:|---:|
| 100 | 703 | 21090 | 0.120876% |
| 200 | 1403 | 42090 | 0.324724% |
| 400 | 2803 | 84090 | 0.417850% |

All sampled local checks are below the declared 1% threshold. The increasing
sampled maximum is not uniform convergence. Whole-interval quadrature changes
over 30 isotope/material groups are at most 0.000058273% (100->200) and
0.000014874% (200->400); all shared raw nodes are exactly equal. These are
table-quadrature checks, not protected-spectrum convergence. The native range
spline derivative differs from direct stopping by up to 0.473399% on the
densest grid. Exporting denser tables does not remove source interpolation or
physical uncertainty.

### Independent GSI checkpoints

Three raw HTML responses were retained. The official form says V1.41 while
the responses literally say ATIMA 1.4; reports record both labels:
https://web-docs.gsi.de/~weick/atima/atima14.html .
The exact query pattern is
`https://web-docs.gsi.de/~weick/atima/atima14_script.cgi?zp=26&massp=56&energy=100&Zt=13&at=0&thickness=1`.
The other cases change `(zp,massp,energy,Zt)` to `(26,56,1000,14)` and
`(2,4,100,14)`. At=0 selects the native target mass; thickness is 1 mg/cm2.

`scripts/compare_ion_stopping_candidate.py` verifies request/projectile/target
readback, literal units, six-decimal precision, matching candidate nodes and
unchanged input hashes. It compares entrance stopping and range, not exit loss.

| Case | Signed stopping difference, candidate/GSI-1 | Signed range difference |
|---|---:|---:|
| Fe56 / Al, 100 MeV/u | -0.207069% | +0.472143% |
| Fe56 / Si, 1000 MeV/u | -0.002905% | +0.021157% |
| He4 / Si, 100 MeV/u | -0.002692% | +0.000289% |

Different charge models/material constants and limited printed precision remain
explicit. Three-point agreement does not validate the full domain or authorize
normative release.

### Artifacts and reproduction

Final raw data use `r3`; earlier files are historical snapshots preceding
import/cache hardening and were not overwritten. SHA-256 under `outputs/`:

| Artifact | SHA-256 |
|---|---|
| ion_stopping_candidate_r3_100_20261031.json | 6bd5081c97d015888c11bab31d0719f8b6b09330909a9c544a8447dd423e9b56 |
| ion_stopping_candidate_r3_200_20261031.json | 1570e2a9ba025cff9f825a2de8e7ada1ea4998c2c243248074857856a5b9760e |
| ion_stopping_candidate_r3_400_20261031.json | a2012f87e3d41aa1e926eb9b31d4335fa36df853b076d51b65890048ce34482c |
| ion_stopping_validation_r3_400_20261031.json | ba3129b3c704fe060c755466b2f5e180576c5f2b453edd1db60cb0454dcb8dd0 |
| ion_stopping_grid_r3_20261031.json | 40cadc236e908d51679a4f9168a5034b7de399691b79db5beb86ea282dd52c70 |
| ion_stopping_grid_r3_200_400_20261031.json | 59aa525dd399999d8df324d5bfd70f6c722e2da3ffb8db651c33efa780e8fed1 |
| ion_stopping_gsi_comparison_r3_20261031.json | 54296a3df2d6c2da89f25b9156ee95546e91ef387d0ecbab159913d7ffc859f3 |

Use CPython 3.12 for the isolated generator and RADAR's environment for
validation; choose unused output paths:

```powershell
& '<CPYTHON_3_12>\python.exe' -B scripts/generate_catima_candidate.py --points-per-decade 200 --output outputs/ion_candidate_replay.json
.\.venv\Scripts\python.exe scripts/validate_ion_stopping_candidate.py outputs/ion_candidate_replay.json --output outputs/ion_candidate_validation_replay.json
.\.venv\Scripts\python.exe scripts/compare_ion_stopping_candidate.py --candidate outputs/ion_candidate_replay.json --output outputs/ion_candidate_gsi_replay.json
.\.venv\Scripts\python.exe scripts/compare_ion_stopping_grids.py outputs/ion_stopping_candidate_r3_100_20261031.json outputs/ion_candidate_replay.json --output outputs/ion_candidate_grid_replay.json
```

### Remaining conditions

Before production replacement: complete ion/mass-policy coverage, establish the
normative LET definition and low-energy range/exit policy, broaden independent
checks around transitions and the legacy high-energy defect, and validate the
actual transport/LET operators. Integration/data distribution also needs a
separate decision. No damaged CSV was smoothed or replaced. Existing audit
restrictions, local-time questions, pending6.6e-2, 1.68, PEAK/CUMULATIVE semantics,
dose/SEE scope and INCOMPLETE status are preserved.

The five new test modules have **205 passing tests**, recorded in
`outputs/pytest_ion_candidate_final_20261031.xml`. Ruff and focused strict mypy
pass all ten new Python files. Tests cover units/mass factors, malformed and
missing rows, nonmonotone stopping, radiation separation, source/runtime changes,
cached/preloaded code, output overwrite refusal and nonfinite checks.

Broad regression completed with **2778 passed** in 330.51 s, zero failures,
errors or skips, recorded in `outputs/pytest_ion_candidate_broad_20261031.xml`.
It excludes the 17 expensive GCR/SEP upper-domain tests and the 11 new
grid-comparison tests, which passed separately. Thus **2789 distinct selected
tests passed**; the 205 focused tests mostly overlap the broad run and must
not be added again. This is not a claim that the 17 expensive tests were rerun.
At handoff, all 160 declared implementation/resource hashes, six pinned
source/package hashes and the generator hash still match the dense r3 report.
No existing production numerical module or resource was edited in this block;
the new validation module is not imported by production calculations.

## HZE mass, LET convention and lower output window (2026-10-31)

This follow-up does not qualify or install the CATIMA candidate. It separates
three previously conflated questions: the mass parameter in the source model,
the isotope used for transport, and the stopping component used as LET.
Stage 4 remains INCOMPLETE. No source formula, spectrum ordinate, shielding
grid, bundled stopping table, 1.68 correction or pending 6.6e-2 policy changed.

### Literal OST masses and explicit candidate inventory

Visually checked the original OST 134-1044-2007 PDF: table B.9 on PDF page 106
(printed 100) and table V.1 on PDF pages 110-112 (printed 104-106). The PDF
SHA256 is `232c84bf7fb6c2d3aec11c44827bcf01a6b6964366a8d0059ff36cd5b143a9e1`.
Fractional A values are printed source parameters; they are not a declared
isotopic composition. The unusual heavy-ion entries, including Sm=145.0 and
Pb=200.4, also occur in the original. They were not replaced by periodic-table
values. SEP A/Q remains independent of A/Z (e.g. Fe A=55.8, A/Q=4.12).

The new `radar.validation.ion_mass_contract` audit compares the actual source
rosters with an explicitly supplied isotope inventory. It never chooses an
isotope by rounding. It distinguishes literal numeric agreement (not isotope
proof), unresolved different masses, missing positive sources and analytically
declared zero sources. Zero classification uses source component coefficients,
not sampled spectrum ordinates. It checks inventory only, not material or
energy coverage, and cannot grant normative export.

| OST source | HZE source rows | Positive HZE | Declared zero | Positive missing from pilot 15 |
|---|---:|---:|---:|---:|
| GCR | 91 | 84 | 7 | 69 |
| SEP | 27 | 24 | 3 | 9 |

The nine missing positive SEP species are F, Na, Al, P, Cl, K, Sc, Ti and V.
GCR additionally retains seven declared-zero rows Po, At, Rn, Fr, Ra, Ac and
Pa; these are not missing positive contributions. No production roster gate
was relaxed. Among the 15 present pilot ions, each source has five exact numeric
mass matches and ten unresolved differences. Nine differences were hidden by
the existing rounded integer metadata; Ni58 versus source58.7 was already
distinguished. Numeric agreement alone still does not prove isotope identity.

Production GCR trace now preserves `source_tabulated_mass_parameter` explicitly.
SEP retains its existing literal A and A/Q and adds the same explicit field.
Both label the legacy rounded species value as not isotope identification and
record `transport_mass_policy=not_established`. Spectrum values are unchanged;
the existing rounded-mass guards are not presented as sufficient qualification.

Pinned audit: `outputs/ion_mass_contract_r2_20261031.json`, SHA256
`78c60756d68585d039c74d4fb89085898e74c46b93cf305e58ec2fbeee952ebd`.
Input: the unchanged r3 200-points-per-decade candidate with SHA256
`1570e2a9ba025cff9f825a2de8e7ada1ea4998c2c243248074857856a5b9760e`.
The runner pins candidate/document bytes and checks source-definition hashes
before/after. It explicitly does not claim to read the PDF independently or
validate stopping rows. The earlier unsuffixed audit predates a type-only loop
variable rename and is retained as historical output, not the current hash set.

```powershell
.\.venv\Scripts\python.exe scripts/audit_ion_mass_contract.py outputs/ion_stopping_candidate_r3_200_20261031.json --candidate-sha256 1570e2a9ba025cff9f825a2de8e7ada1ea4998c2c243248074857856a5b9760e --output outputs/ion_mass_contract_replay.json
```

### LET means ionization loss, not all energy loss

Visually checked STO GK Roscosmos 04.01.0010-2025 section 3.1, rendered page 9:
the definition concerns energy lost in ionization per path length and density.
Independently, [ESCC 25100 Issue 2, October 2014](https://escies.org/escc-specs/published/25100.pdf),
section 2, page 6, explicitly defines SEE LET as mass electronic stopping power.
Its retained source `research/ion_stopping_20261031/escc_25100_issue2.pdf` has
SHA256 `97839cdbda517fdef8951b458deb1c853737efe8048bd26c247b4cd0b0f16d74`;
the complete relevant page was rendered and visually checked.

This supports electronic collision stopping in Si for direct-ionization LET,
with MeV cm2/g divided by 1000 to obtain MeV cm2/mg. Nuclear and radiative loss
are not added to that LET. Range/energy degradation remains a distinct operator:
[NIST PSTAR/ASTAR section 3.5.4](https://physics.nist.gov/PhysRefData/Star/Text/programs.html)
uses total stopping in its CSDA integral. NIST's cited domain is protons/helium,
not independent numerical validation of every CATIMA heavy ion. A finite
delta-electron cutoff and a source isotopic distribution are not established
by these definitions.

The isolated candidate already separates electronic, nuclear and radiative
columns. The legacy production silicon conversion still consumes its old
total column and remains quarantined. Its trace now says
`stopping_component_used=legacy_table_total` and
`direct_ionization_let_convention=not_qualified`. We did not replace a corrupted
table by choosing another column from it or silently relabel total as electronic.

### Explicit lower-output-window omission

`shield_hze_spectrum_primary_through_al` inherits its initial output window from
the source logarithmic edges. A degraded particle may lie below that window
while both input and output energies remain supported by the range table.
The existing domain diagnostics did not separately account for this omission.

For positive thickness, transformation v4 now records a disjoint three-way
partition of the integral over the declared point-sample source interval only:

1. supported exits in the inherited output window;
2. supported exits below the window;
3. unresolved transport (incident energies outside the table or residual ranges
   below the table minimum).

The third category is **not** a physically stopped count. A positive second
category adds an explicit INCOMPLETE reason. The output grid and ordinates are
unchanged; this is transparent accounting, not recovery of the omitted spectrum.
Flux integrals remain cm-2 s-1 and fluence integrals remain cm-2; no mission-time
factor is inserted. Zero shielding remains the original exact identity.

Analytic witness: j=2 on E=10..100, R(E)=E on table E=1..200, t=20. The inherited
lower edge is 10/sqrt(3), while supported outputs span E=1..80. The partition is
148.4529946162075 in the output window + 9.54700538379251 below it + 22 unresolved
= 180 in the declared source interval. Tests repeat the calculation with
R(E)=E-1 (zero table minimum) and R(E)=E+137. The additive range origin cancels
for supported input/output differences; it does not establish stopping at E=0.

An independent read-only review also checked 1512 analytic cases with affine
range and source functions, partial/disjoint supports, range offsets, four
accepted quantity types and six thicknesses. No blocker was found; maximum
normalized partition/output discrepancy was about 1.01e-12. This is an
algorithmic check, not calibration of real stopping data.

### Next acceptance work

The next numerical experiment should compare explicitly declared source-mass
and isotope transport conventions, extend positive-ion coverage, then test the
electronic-LET mapping and full supported exit domain with the replacement
tables. Do not infer isotope assignments from rounded labels or issue a full
normative spectrum from the pilot set. Low-energy physical stopping, source
mass-policy qualification, independent numerical table validation and release
eligibility remain separate open conditions. Dose/SEE implementation stays out
of scope until Stage 4 is closed.

### Verification of this follow-up

Broad regression: **2841 passed** in 328.76 s, zero failures/errors/skips,
`outputs/pytest_mass_domain_broad_20261031.xml`. The 17 expensive tests in
`test_gcr_upper_domain.py` and `test_sep_upper_domain.py` were explicitly excluded
and were not rerun; their collection count was checked separately. The 52 new
tests also passed in a focused final run,
`outputs/pytest_mass_domain_final_20261031.xml`; they overlap the broad run and
must not be added to its count. Ruff and strict mypy pass the nine reviewed
implementation/new-test files. Targeted `git diff --check` passes.

At handoff all seven source-definition hashes in the r2 mass audit still match
the current files. The previous stopping candidate's implementation hash set
is historical after this follow-up's deliberate metadata/diagnostic changes;
it is not represented as a hash of the current production implementation.

## Full positive HZE mass-parameter experiment (2026-10-31)

This block generates a complete **positive source-inventory candidate**, not a
qualified isotope table or a replacement production dataset. No existing
production calculation, old isotope-v1 contract, source formula or normative
resource was changed. A new isolated contract accepts an explicitly named
`mass_parameter_u`; it does not redefine the integer isotope field of v1.

### Mass conventions and source coverage

`scripts/build_stopping_mass_manifest.py` reconstructs all current OST source
rows. It preserves literal fractional A, separate SEP A/Q, positive component
membership and declared-zero entries. All 24 positive SEP HZE entries have the
same literal mass and element identity as their positive GCR counterparts.
The resulting candidate has 84 positive GCR elements and contains all 24
positive SEP elements. The seven GCR and three SEP declared-zero entries remain
in the source manifest but are not presented as missing nonzero contributions.

The paired control is the explicitly supplied 15-isotope inventory from the
SHA256-pinned earlier candidate. No rounding selects any new isotope. Two
datasets are generated on exactly the same energy grid and backend settings:

- `source_tabulated_parameter_diagnostic`: literal source A supplied numerically
  as the backend projectile-mass argument;
- `explicit_pilot_isotope_diagnostic`: the unchanged explicit integer A values
  from the historical pilot, not a newly determined isotope distribution.

The first convention deliberately assumes numerical compatibility between
source MeV/nucleon and backend MeV/u. Even integer A is not a precision atomic
mass in u. Neither assumption is promoted to a qualified physical mass policy.
Source SEP A/Q is retained but is not supplied as transport Q. The pinned
Pierce-Blann configuration uses equilibrium charge from Z and velocity.

### Backend limits are visible

The retained CATIMA implementation accepts continuous projectile A and compares
it exactly in its cache. The present Z range stays at or below uranium. The
LS mass correction uses a linear weight `(A/Aref - 1)/0.05` between its elemental
reference mass and 1.05 times that reference; outside this interval it
extrapolates. The reference mass is obtained from the backend's elemental
material table and recorded for every projectile. **64/84** source-parameter
projectiles and **9/15** pilot projectiles lie outside this mass interval.
This is recorded as a model limitation, not clipped, hidden or automatically
interpreted as a measured error. It includes the unusual literal heavy masses
already audited in OST. No source-table mass was silently corrected.

The native range spline retains its own fixed internal energy grid. Increasing
the exported grid density does not refine that native calculation. Range
differences are usable only within supported energies; the native origin at
0.001 MeV/u remains unqualified as a zero-energy physical range.

### Generated matrix and comparisons

The energy domain is 0.01..100000 MeV/u, with explicit 10, 30, 300 and 50000
nodes. The extra 300 node supplies a common diagnostic control, not a model
transition. Both Al and Si are calculated with separated electronic collision,
nuclear and high-energy radiative stopping. The new validator independently
checks the declared source inventory and the rectangular row coverage, then
checks range increments against `integral mass_parameter/S_total de` using
piecewise log-log stopping interpolation.

| Output density | Energy nodes | Source-parameter rows | Pilot rows |
|---|---:|---:|---:|
| 100 per decade | 704 | 118272 | 21120 |
| 200 per decade | 1404 | 235872 | 42120 |

For the 15 paired elements, changes below use **source-parameter / explicit
pilot - 1**, at the same energy and material. They are sensitivity to the chosen
mass convention, not error relative to an experimental truth or changes in a
complete transported spectrum.

| Quantity | Largest absolute relative change | Location and masses |
|---|---:|---|
| Al range | +1.2540272119% | Mg24.3 vs Mg24, 100000 MeV/u |
| Electronic LET in Si | -0.0174393353% | Ni58.7 vs Ni58, 100000 MeV/u |
| Total stopping, Al/Si | +0.0957592385% | Ni58.7 vs Ni58, Si, 0.01 MeV/u |

Electronic stopping in Al has a slightly larger maximum than Si
(-0.0175020032% at Ni/100000 MeV/u); it must not be reported as the Si LET result.
The comparison was independently recomputed from the dense raw rows with
before/after file hashes. All 42090 historical pilot rows at common nodes are
exactly reproduced; the 30 additional dense pilot rows are the new 300-MeV/u
node for 15 ions in two materials. Historical production hashes in the old
candidate are not falsely required to describe the current implementation.

At 200 nodes per decade, the largest range-increment discrepancy is
**0.3247236046%**, Cr52/Al, 0.3311311215..0.3349654392 MeV/u, for both datasets.
An independent Simpson quadrature check of all 235704 source intervals and
42090 pilot intervals reproduced this maximum within about 3.52e-12 in relative
discrepancy. This confirms arithmetic consistency, not model calibration.
The coarse source maximum is 0.1368091324%, so the local maximum increases on
refinement; a small whole-domain integral change must not be called uniform
convergence of the range derivative or convergence of the full RADAR spectrum.

The 100-to-200 comparison has exactly equal common rows in both datasets. The
largest absolute relative change of the whole-domain quadrature is
6.2678025037e-7 (0.0000626780%) for source-parameter Sr87.6/Si and
5.8301536043e-7 (0.0000583015%) for pilot Ni58/Si. The denominator is the
coarse-grid integral. These are table-integral checks, not spectral convergence
or proof that every local derivative discrepancy decreases.

### Reproducibility and remaining gate

Native generation runs under the isolated pinned CPython-3.12 wheel; validation
does not load that native calculator. Manifest, source files, runtime package,
generators and declared production files are checked for mutation. Candidate
outputs are exclusively created; old outputs are not overwritten.

| Artifact in outputs/ | SHA256 |
|---|---|
| stopping_mass_manifest_20261031.json | f367a1b5f047451e40ea37d4cb03139ff6cbdb7c7eb9333888b44c11d46250fc |
| parameterized_stopping_100_20261031.json | 23299c0eb992959f376bbe02bb497b31502d0bf6c215c2de72a2931bfee39fef |
| parameterized_stopping_200_20261031.json | 77810ff713f6421c1900441207b18b54da701d19259f0c6bcc3ed5094a48f309 |
| parameterized_stopping_validation_r2_100_20261031.json | 578d76494206e1b70331b7030b7324ab2fff248c5105e0e2dd1c953325e2e160 |
| parameterized_stopping_validation_r2_200_20261031.json | 9135623b75a9a6af9078d8696d6bec8df15931b4f2bdecd4ef4159f175ab2516 |
| parameterized_stopping_grid_comparison_r2_20261031.json | e0d4c9a7d21b61b89c57a4c9770e7624e8e42abfbe8efb37a0446fa38a663814 |

The r2 validations additionally pin their own implementation and replay the full
manifest from the pinned historical candidate and current source definitions.
Semantic strings for energy, charge and LET are exact checked conventions, not
arbitrary labels. The final independent review confirmed both safeguards. The
repeated r2 grid comparison is byte-identical to its first retained report.

Reproduction uses unused output names; no external calculator is loaded during
validation or grid comparison:

```powershell
& '<CPYTHON_3_12>\python.exe' -B scripts/generate_parameterized_stopping.py --manifest outputs/stopping_mass_manifest_20261031.json --manifest-sha256 f367a1b5f047451e40ea37d4cb03139ff6cbdb7c7eb9333888b44c11d46250fc --points-per-decade 200 --output outputs/parameterized_stopping_replay.json
.\.venv\Scripts\python.exe scripts/validate_parameterized_stopping.py outputs/parameterized_stopping_replay.json --output outputs/parameterized_validation_replay.json
.\.venv\Scripts\python.exe scripts/compare_parameterized_stopping_grids.py outputs/parameterized_stopping_100_20261031.json outputs/parameterized_stopping_replay.json --legacy-pilot outputs/ion_stopping_candidate_r3_200_20261031.json --legacy-pilot-sha256 1570e2a9ba025cff9f825a2de8e7ada1ea4998c2c243248074857856a5b9760e --output outputs/parameterized_grid_replay.json
```

All **244 new tests** pass in `outputs/pytest_parameterized_final_20261031.xml`.
Ruff and strict mypy pass all ten implementation/test files. The original
integer-isotope validation API and the numerical production modules are
unchanged. These software checks do not close the physical qualification gate.

Broad regression: **3019 passed** in 335.42 s, recorded in
`outputs/pytest_parameterized_broad_20261031.xml`. It excludes the 17 expensive
GCR/SEP upper-domain tests and the 66 new runner/grid tests, which passed in
the focused run. XML testcase identities confirm **3085 distinct selected
tests passed** across the broad and focused runs; overlapping tests are not
double-counted. This is not a claim that the 17 expensive tests were rerun.
At handoff all 162 declared production/resource hashes and all three final
validator implementation hashes still match; the documentation-only follow-up
does not change those inputs.

Production HZE tables remain unchanged and quarantined/provisional as before.
The next work is a conservative transport/electronic-LET adapter with complete
supported output coverage, followed by independent physical checks and an
explicit mass-policy decision. Internal source coverage now exists for the
candidate, but isotope identification, absolute low-energy stopping, physical
table accuracy, fragmentation, integration/licensing and normative release
remain open. Stage 4 stays INCOMPLETE; no dose or SEE implementation is advanced.

## Conservative candidate primary transport and electronic LET (2026-10-31)

This continuation implements the next diagnostic operator on the full positive
mass-parameter candidate. It does **not** replace any production table or
production shielding/LET function. `candidate_transport.py` and
`candidate_let.py` live in `radar.validation`, require explicit candidate
references and always return `INCOMPLETE`; their release flags cannot be enabled
through constructor arguments. Existing source warnings survive both operators.

### Physical definitions and bounded numerical contract

The supplied STO 04.01.0010-2025 section 3.1 (rendered page 9) defines LET through
ionization losses divided by path length and density. ESCC 25100 issue 2,
section 2/page 6, explicitly identifies the mass electronic stopping power as
LET for its heavy-ion SEE context. Both complete pages were visually rechecked.
The retained documents and hashes are listed in the earlier stopping-source
audit. The independent [NIST PSTAR/ASTAR method description, section 3.5.4](https://physics.nist.gov/PhysRefData/Star/Text/programs.html)
defines CSDA range through the reciprocal **total** stopping power. NIST's
proton/helium database is not thereby a validation of all candidate heavy ions.
These definitions motivate separate Al range and Si electronic-LET quantities;
they do not make the following interpolation policy a normative formula.

For the declared diagnostic mass policy and straight-path primary CSDA model:

`R(E_in) - R(E_out) = t`.

`R` is piecewise linear in tabulated energy per nucleon. For an exit interval,
the operator integrates the **original** incident source over the inverse-mapped
incident interval. Point-sample source densities are piecewise linear; a
bin-average source explicitly uses a piecewise-constant density. Integration
never extends outside the declared source interval. No additional time,
solid-angle, mass or material-density multiplier is applied. This is the
spherical-shell-at-centre path convention, not a new 3D geometry model.

For every Si energy segment, `L_e(E_out)` is linearly interpolated from the
electronic collisional stopping divided by 1000, giving MeV cm2/mg. Each LET bin
receives the sum of source integrals over **all** its preimage intervals.
Increasing and decreasing branches are handled separately; plateaus belong to
one left-closed/right-open bin, with the final upper edge included. There is no
division by `dL/dE`, no centre-bin assignment and no reconstruction of the
transported measure from an intermediate energy histogram. Cropping the energy
display therefore cannot silently crop a subsequent LET result.

The adapter requires exact agreement of the explicit source mass parameter,
the table mass and any source mass trace. Rounded `ParticleSpecies.mass_number`
remains a display label, not a new isotope assignment. A pilot-isotope table
with a different mass cannot be passed by rewriting the source trace: this run
uses only `source_tabulated_parameter_diagnostic`, not a paired isotope
transport experiment. Candidate SHA256, manifest SHA256 and mass policy appear
in the dataset reference. Legacy CSV references are not substituted for them.

### Support accounting and independent arithmetic checks

For positive thickness the default output covers the full Al table energy
domain. Explicit energy windows have separate below/above-window counts. The
primary ledger partitions the finite source integral into supported exits,
incident energies outside Al support, and residual ranges below supported
range. That last category is **unresolved**, not proven stopped. A separate LET
ledger accounts for both LET-window tails and exits outside Si support. The
source's unrepresented tails are unknown and do not acquire a zero value from
ledger closure. Zero thickness preserves original energy x/y/ordinate basis
exactly even outside Al support; Si support is still checked separately.

The analytic witness `j=2` on [10,100], `R(E)=E` on [1,200], `t=20` yields
supported exits [1,80], integral **158**, and unresolved integral **22**. Thus
supported exits below the original source's energy minimum are actually
represented, rather than merely reported as omitted. A triangular electronic
LET curve checks both branches, and plateau tests check exact boundary ownership.

Independent review found a floating-point origin defect during development:
with a common range offset of 1e15, subtraction of a 0.01-g/cm2 shield could
disappear while the particle-count ledger still closed. The final adapter
subtracts the common tabulated range origin **before** interpolation and range
arithmetic. Regressions cover 0.01 and 0.1-g/cm2 shields with that offset,
ordinary offsets, and representable slivers around support boundaries.

`scripts/check_candidate_pushforward_witness.py` supplies an independent
incident-energy midpoint integration, using no adapter integrals for its
reference calculation. A nonlinear-range, nonconstant-source, nonmonotone-LET
case with a plateau and cropped support checks 14 ledger/bin quantities. With
2 million and 4 million incident points, the maximum absolute differences are
9.4071e-6 and 4.9572e-6 respectively; the latter is 1.6977e-7 of the analytic
source integral 29.2. This verifies bounded arithmetic, not physical accuracy.

### Full positive-inventory two-grid experiment

`scripts/validate_candidate_transport.py` independently validates each pinned
raw candidate and its reconstructed mass manifest before using any rows. It
compares 100 and 200 table nodes per decade on the **same fixed external source
spectra** and the same display edges. Source-grid density and the backend's
internal range grid are not refined by this experiment.

- GCR: all 84 positive HZE entries; existing main+NEK+AKL source implementation;
  fixed W=60, lagged W=50, Wmin=10, Wmax=100, cycle 24 after reversal, AKL year 2.
  Existing mission-product code builds a maximum and a single-bin 5-year
  fluence. This fixed state is not a calibrated 2026 mission solar history.
- SEP: all 24 positive HZE entries; existing peak/fluence models, launch 2026,
  five years, exceedance probability 0.1, explicitly constant monthly W=60.
- Both modes; shields 0, 0.01, 0.1, 1, 3 and 10 g/cm2: **1296 cases per grid**.
  GCR source support is 1..100000 MeV/nucleon (1001 nodes); SEP support is
  1..10^4.1 MeV/nucleon (821 nodes). This is not the LEO/GEO/HEO acceptance matrix.

All selected integral comparisons are below 1%. The largest change is
**0.4521201129%**, GCR Li, PEAK, 3 g/cm2, electronic-LET integral above
1 MeV cm2/mg (within the declared LET window). Coarse values are denominators;
a zero coarse value with a positive dense value is undefined, not assigned a
zero relative error. Selected LET thresholds are 0.01, 0.1, 1, 10, 50 and
100 MeV cm2/mg. The upper LET edge is 1000 MeV cm2/mg; its separate tail ledger
is zero for this matrix, not generally assumed zero.

The largest histogram L1 difference divided by the coarse integral is
**0.1014982018%** for energy (SEP Mg, CUMULATIVE, 0.1 g/cm2) and
**0.0976821586%** for LET (SEP F, CUMULATIVE, 1 g/cm2). These are normalized
distribution differences, not maximum pointwise errors. The dense-grid maximum
relative primary/LET ledger residuals are 1.88e-16 / 2.22e-16. All zero-shield
energy identities pass. No source lies outside Al support and no supported exit
lies outside Si support in this particular bounded matrix.

The newly represented exits below the source minimum of 1 MeV/nucleon reach
11.30145409% of the declared incident integral for SEP He, CUMULATIVE,
0.01 g/cm2. This is a domain-accounting result for the fixed source convention,
not a physical accuracy claim. At thick shields the unresolved low-residual
category may dominate; it has deliberately not been relabelled as stopped.

The final r2 replay reproduces all coarse/dense numerical cases, comparisons
and source fingerprints exactly. It adds CLI selection, matched mass/unit and
final manifest-stability guards. All eight experiment implementation pins and
all 162 declared production/resource pins match at handoff. The raw 100/200
candidates and production tables are unchanged.

| New retained artifact in outputs/ | SHA256 |
|---|---|
| candidate_transport_full_r2_20261031.json | 707115dd6f10572a2b865702a6457aef540e6160f973ea7ea8d3984825c98e44 |
| candidate_pushforward_witness_20261031.json | f154b16db918061228581e15326c39f96b7af86a7487d5369521f8b51aeb36c4 |

Reproduce with unused output names:

```powershell
.\.venv\Scripts\python.exe scripts/validate_candidate_transport.py outputs/parameterized_stopping_100_20261031.json outputs/parameterized_stopping_200_20261031.json --output outputs/candidate_transport_replay.json
.\.venv\Scripts\python.exe scripts/check_candidate_pushforward_witness.py --output outputs/candidate_pushforward_replay.json
```

The final focused run has **178 passed** (92 transport, 44 LET, 42 runner),
recorded in `outputs/pytest_candidate_final_20261031.xml`. Ruff and strict mypy
pass all seven new implementation/test files. Numerical conservation and this
fixed-source grid comparison do not qualify the physical mass convention,
low-energy absolute range, native stopping model or material data, nor do they
resolve licensing/integration or attenuation/fragmentation assumptions.

The broad regression completed with **3221 passed** in 337.41 s, with no failures,
errors or skips (`outputs/pytest_candidate_broad_20261031.xml`). It explicitly
excluded the 17 expensive GCR/SEP upper-domain tests. The 42 runner tests were
added after that run's collection and passed in the final focused run instead.
XML testcase identities give **3263 distinct selected tests passed**, with 136
overlapping cases; 3221 + 178 must not be reported as independent tests. This is
not a claim that the excluded 17 tests were rerun. `git diff --check` is clean;
Ruff/mypy additionally cover all seven new untracked code/test files.

The live strict OST audit still has **six blocking findings in five production
tables**. Existing AKL/He-NEK, orbital/geomagnetic and secondary-proton blockers
also remain. Next is independent physical table qualification and an explicit
mass-policy decision before any production replacement, followed by the full
orbital acceptance matrix. Stage 4 remains INCOMPLETE; dose and SEE stay outside
this continuation.

## Independent ASTAR helium comparison (2026-10-31)

The next physical-data check compares the existing diagnostic CATIMA He4
candidate with the independently evaluated NIST ASTAR data for aluminium and
silicon. No candidate or production value is fitted, rescaled or replaced.
There is no assertion that ASTAR is an error-free oracle, nor that the spectral
grid criterion of 1% is a physical accuracy requirement for stopping tables.

### Source, version and material identity

The official [STAR 1.2 archive](https://physics.nist.gov/PhysRefData/Star/STAR.tar.gz)
was downloaded into `research/nist_star_20261031/STAR.tar.gz`. Its 444183 bytes
have the previously recorded SHA256
`b8241beb0e721b6b89dd316e32b23ac6c80e9a8908bd43da1570216de51de2c6`.
The [download page](https://physics.nist.gov/PhysRefData/Star/Text/download.html)
permits redistribution/modification with derivative/modified notices, and
explicitly warns that the downloadable and current online versions need not
be identical. Retained source descriptions, material pages and current online
responses have individual SHA256 pins in the comparison script/report.
Derivative notices are included in the new archive reader and report.

`src/radar/validation/nist_astar.py` reads native archive nodes without running
Fortran or porting its interpolation. `ACONV.f` documents four arrays per
material: electronic stopping, nuclear stopping, CSDA range and detour factor.
There are 74 materials and 122 nodes each. `ENG.ALF` is independently matched
against `APOL`'s `DATA E` grid. The reader rejects invalid record boundaries,
missing/duplicate nodes, nonfinite data, invalid ranges and unsafe tar entries.

A consequential source detail was checked independently: `IDNO` maps display
labels in `MATS`, whereas `AREAD/IND` maps records in `FALPH`. Graphite 906 is
in different positions in those inventories. Al/Si are respectively FALPH
records **8/9**, not MATS records 9/10. The parser joins by material identifier.
An independent flat-array decoder matched all **9028 rows x 5 fields** exactly.
This is source-data readback, not compiled Fortran execution: original ACONV
uses four-byte default REAL, while the reader preserves the native decimal
table values as Python floats.

The current official online form uses
`POST https://physics.nist.gov/cgi-bin/Star/ap_table-t.pl`, with `prog=ASTAR`,
`matno=013` or `014`, `ShowDefault=on`, and an empty custom energy list. The
initial guessed `a_table-t.pl` address returned 404 and produced no reference
file; it was not interpreted as absent scientific data. The verified form's
responses are retained as `live_astar_013.html` and `live_astar_014.html`.

The separate inert HTML reader `nist_astar_live.py` verifies title, particle,
material/link, units, seven columns, exact printed tokens and range ordering.
Both current responses contain **121 nodes**, missing **22.5 MeV** from the
122-node archive grid. The comparison records this absence; it neither fills
the missing row nor declares it zero. On the 121 shared nodes, all 847 printed
values per material agree with the native archive values within their printed
rounding intervals. Total stopping and projected range are derived separately
from the native electronic/nuclear and CSDA/detour fields.

Decimal arithmetic is required at rounding boundaries: binary subtraction can
place a true decimal tie just outside a half-printed-unit interval. Independent
review found and corrected this false-positive comparison defect. Final checks
find no out-of-cell values (six exact ties for Al, three for Si). This bounded
printed-precision agreement is not a claim of full online-version equivalence
or replay of ASTAR's single-precision logarithm/spline/exponential operations.

### Matching quantities and separating numerical interpolation

The archive source explicitly defines ASTAR energy as the **whole alpha-particle
kinetic energy**, not energy per nucleon. Under the retained diagnostic He mass
parameter 4.0, native CATIMA is evaluated at `e = E_ASTAr / 4`. Electronic and
nuclear mass stopping powers are compared directly in MeV cm2/g, without a
factor of four or an additional density factor. Si electronic LET is obtained
by dividing electronic stopping by 1000. CSDA range is used, not projected
range; the latter is separately `CSDA * detour` in ASTAR.

The matched overlap consists of **101 nodes per material**, 0.04..1000 MeV
whole-alpha energy, equivalently 0.01..250 MeV/u under the stated convention.
This does not establish the physical precision of the mass parameter, the
other 83 positive GCR elements, or candidate energies above this overlap.

`scripts/generate_astar_helium_controls.py` obtains **202 exact native samples**
at those ASTAR energies, using the unchanged pinned CPython-3.12 wheel and the
same configuration as the candidate. It makes no material-I override. Recorded
native Al/Si values are A=26.9815/28.0855, density=2.699/2.33 g/cm3, and elemental
I=166/173 eV. The current [NIST Al](https://physics.nist.gov/cgi-bin/Star/compos.pl?matno=013)
and [Si](https://physics.nist.gov/cgi-bin/Star/compos.pl?matno=014) descriptions
have the same I values and densities 2.69890/2.33000. The small Al density
difference is recorded, not silently retuned. The retained CATIMA source selects
`srim_85` at low energy; this is not the same fitted model as ASTAR's ICRU-based
evaluation. The comparison alone does not identify which model is more accurate.

The exact native samples separate output-table interpolation from inter-model
differences. At 200 candidate nodes per decade, the largest electronic-stopping
interpolation differences relative to exact native evaluation are
**-0.04776474% Al** and **-0.01928366% Si**, both at 0.1 MeV whole-alpha energy.
The corresponding maximum raw-range interpolation differences are
**+0.00227484% Al** and **+0.00243542% Si**, both at 85 MeV. Thus interpolation of
this output grid does not explain the much larger ASTAR differences below.
The native low-energy electronic-component closure has 134 checks and maximum
relative discrepancy 2.22e-16; this remains an internal arithmetic check.

### Inter-model differences, not an acceptance verdict

Values below are maximum absolute differences, with their **signed** value
shown as `100 * (native_CATIMA / native_ASTAR - 1)`. All energy locations are
whole-alpha MeV. The range increment is relative to the common supported anchor
0.04 MeV: `deltaR(E) = R(E) - R(0.04)`; the anchor itself has no relative ratio.

| Quantity | Aluminium | Silicon |
|---|---|---|
| Electronic mass stopping | -6.317593% at 0.04 MeV | +8.632568% at 0.4 MeV |
| Total mass stopping | -6.241052% at 0.04 MeV | +8.594157% at 0.4 MeV |
| Anchored CSDA range increment | +6.458636% at 0.045 MeV | -6.954853% at 0.65 MeV |
| Raw absolute range, origin-confounded | -12.994967% at 0.04 MeV | -25.004684% at 0.04 MeV |

The raw-range line must not be described as an isolated stopping-model error:
CATIMA's native range origin is 0.001 MeV/u, or 0.004 MeV whole He under the
retained mass convention, and is not qualified as physical zero energy.
Anchoring removes a common additive offset but not differences in stopping
physics. No ASTAR offset is added to CATIMA. The report also retains nuclear
stopping separately: its large relative differences at high energy must not
be equated to the same percentage difference in total stopping.

The comparison is energy-dependent. For example, on the native nodes from
100..1000 MeV whole-alpha energy, maximum electronic-stopping differences are
-0.430486% Al and +1.197010% Si, while the anchored-range maxima are +0.463176%
and -2.542642%. The complete report includes five overlapping endpoint-inclusive
energy bands; shared endpoints are intentionally counted in each band's summary.

NIST's [method description](https://physics.nist.gov/PhysRefData/Star/Text/programs.html)
gives energy-dependent uncertainty guidance, including larger low-energy
uncertainty than at high energy. Those published estimates are not a measured
uncertainty for our candidate and are not assigned as an automatic acceptance
band. A numerical `<1%` grid result cannot resolve this physical qualification
question. The next decision needs an explicit model/domain/accuracy basis,
not rescaling the candidate to make two programs agree.

### Retained artifacts and release status

| Artifact in outputs/ | SHA256 |
|---|---|
| astar_helium_native_controls_20261031.json | 75162717801d59ab4048c6a11d9dd9a229f56f90e9d148e6b3939e2f8cd157a6 |
| astar_helium_comparison_20261031.json | 4fbf11ff8637de6c01a7b8c4c8635f21bf2c4167914858562bf95effa7414333 |

Reproduction uses unused output names and never changes production resources:

```powershell
& '<CPYTHON_3_12>\python.exe' -B scripts/generate_astar_helium_controls.py research/nist_star_20261031/STAR.tar.gz --output outputs/astar_helium_native_replay.json
.\.venv\Scripts\python.exe scripts/compare_astar_helium.py outputs/parameterized_stopping_100_20261031.json outputs/parameterized_stopping_200_20261031.json outputs/astar_helium_native_replay.json --native-sha256 75162717801d59ab4048c6a11d9dd9a229f56f90e9d148e6b3939e2f8cd157a6 --output outputs/astar_helium_comparison_replay_new.json
```

All **324 new tests** pass (95 archive reader, 80 online reader, 42 native
control generator, 107 comparison), recorded in
`outputs/pytest_astar_final_20261031.xml`. Ruff and strict mypy pass all eight
new implementation/test files. Native control and comparison replays are both
byte-identical to their retained reports, not just numerically close. All nine
comparison implementation pins and all 166 declared production/resource pins
still match after the checks.

Broad regression: **3587 passed** in 379.87 s, with zero failures, errors or
skips (`outputs/pytest_astar_broad_20261031.xml`). This run includes all 324 new
tests; their focused run is not added again to the total. The 17 expensive
GCR/SEP upper-domain tests were explicitly excluded and were not rerun in this
continuation. XML testcase identities confirm the focused set is contained in
the broad set. `git diff --check` is clean; Ruff/mypy also cover the untracked
new source/test files.

The raw candidate tables, source spectra, existing operators and accepted
coefficients are unchanged. All physical-accuracy, mass-policy and normative
approval flags remain false. The strict OST audit still reports six findings
in five production tables; neither this He comparison nor the new archive
reader closes the electron/proton table provenance gates. Stage 4 remains
INCOMPLETE.

## Bounded helium spectral impact of ASTAR differences (2026-10-31)

The next experiment evaluates how the preceding table differences propagate
through primary shielding and electronic-LET conversion. It does not select a
production stopping model or replace a table. The implementation is
`scripts/compare_astar_helium_impact.py`; it uses the existing diagnostic
conservative source-measure operators without changing them.

### Matched support and independent operator choices

The common table support is **0.01..250 MeV/n**, corresponding to native ASTAR
whole-alpha energies **0.04..1000 MeV** under the retained mass parameter 4.0.
The archive is the previously pinned redistributable STAR 1.2 release. The
conversion remains `E_n = E_alpha/4` and `L_e = S_e/1000`; stopping power and
range are not divided by four, and no density or mission-time factor is added.
Native ASTAR CSDA ranges, not projected ranges, are used.

Four factorial variants use the **same 101 energies**: the first letter chooses
the Al range and the second chooses Si electronic LET; C is exact native CATIMA
at those energies and A is the native ASTAR table. Thus CC, AC, CA and AA separate
range and LET effects, with interaction `AA - AC - CA + CC`. The mixed variants
are numerical decompositions, not proposed physical hybrid models. All use
RADAR's piecewise-linear interpolation of the retained nodes, not ASTAR's APOL
spline, a compiled ASTAR executable or online ASTAR interpolation.

Two additional CATIMA variants restrict the existing 100/200-per-decade tables
to the same support, retaining **442/882 nodes**. The endpoint 250 MeV/n is
obtained by bounded interpolation, never extrapolation. The 101-node versus
dense-CATIMA comparison measures sparse-grid sensitivity for that candidate;
it does **not** bound ASTAR's unknown inter-node interpolation error.

The source state and full source energy grids are unchanged from the preceding
candidate-transport experiment: external GCR and SEP He in both PEAK and
CUMULATIVE modes, behind 0, 0.01, 0.1, 1, 3 and 10 g/cm2. This gives **144 cases**:
four source/mode spectra times six shields times six table variants. There is
no orbit, geomagnetic attenuation, source-grid refinement or additional time
integration in this calculation.

### Coverage cannot be replaced by zero

The fraction of the declared original source integral **above 250 MeV/n** is:

| Source and mode | Fraction above common table support |
|---|---:|
| GCR PEAK and CUMULATIVE | 88.37542955% |
| SEP PEAK | 0.006691883% |
| SEP CUMULATIVE | 0.004419456% |

These fractions are conditional on the fixed diagnostic source state, not
general statements about all orbital environments. Original source tails are
not cropped and renormalized. At positive thickness, their integrals remain
outside-domain unresolved contributions. Below-range residuals are separately
unresolved, not classified as physically stopped. Undeclared source tails
beyond the original energy grid remain unknown as before.

At zero shielding, the entire energy spectrum is still exactly the input:
an Al table is not needed for that identity. The reported common-window energy
histogram is explicitly distinguished from that full identity. Si LET is only
mapped over the supported interval; the high-energy contribution remains in
the outside-Si ledger even when the shield is zero.

All 24 source/mode/shield combinations satisfy the factorial checks: changing
only Si LET leaves the energy-bin integrals unchanged and preserves the total
full-window mapped measure. Across 144 cases, maximum absolute primary and
LET ledger residuals divided by the original source integral are respectively
below **1.625e-16** and **1.693e-16**. These are arithmetic conservation checks, not
physical-accuracy evidence.

### Representative effects and resolution sensitivity

For SEP PEAK at 0.01 g/cm2, the same-node comparisons are:

| Integral over the supported response | CC | AA | `(AA-CC)/CC` |
|---|---:|---:|---:|
| All mapped primary He, cm^-2 s^-1 | 111723.1694 | 111817.5833 | +0.084507% |
| Electronic LET >= 1.4 MeV cm2/mg, cm^-2 s^-1 | 2969.1030 | 1100.3017 | -62.941613% |
| Electronic LET >= 1.5 MeV cm2/mg, cm^-2 s^-1 | 1406.9212 | 0 | -100% |

The last two changes are respectively **-1.048663%** and **-0.789482%** of
the original source integral inside the common energy domain. The last line
is a structural cutoff in this particular interpolant: native ASTAR's maximum
electronic LET is 1.41706, whereas CATIMA's matched-node maximum is 1.537869
MeV cm2/mg. It is not proof of a physically zero complete high-LET flux or of
zero SEE. No SEE response has been calculated here.

For comparison, the largest combined-table energy-integral relative change
among the selected positive-denominator metrics is **+1.009765%**, SEP
CUMULATIVE at 0.1 g/cm2, for exits 0.01..250 MeV/n. The largest change normalized
by the common-domain source integral among the selected LET-tail/mapped-count
metrics is **-1.954063%**, SEP CUMULATIVE at zero shielding and LET >= 0.5; its
relative change against the CC tail itself is **-5.224622%**. Different
denominators must not be interchanged.

The two numerical witnesses also need separate interpretation:

- CATIMA 100 -> 200 nodes/decade: maximum absolute selected energy/mapped integral
  relative change **0.0120985%**; maximum absolute selected LET-tail relative change **0.4829424%**
  (GCR CUMULATIVE, 3 g/cm2, LET >= 1.5).
- Dense CATIMA -> the 101 ASTAR-node CATIMA controls: maximum absolute selected
  energy-integral relative change **0.9703728%**; maximum absolute selected LET-tail relative change
  **4.8303357%** (GCR PEAK, 3 g/cm2, LET >= 1.5).

The sparse-node witness therefore must not be called uniformly converged below
1%. At the representative SEP PEAK / 0.01 g/cm2 case, its LET >= 1.4 and 1.5
relative changes are only -0.0277865% and -0.2274844%, respectively. This helps
separate numerical effects locally but still does not establish an ASTAR
interpolation-error bound or physical uncertainty.

Every metric retains left/right values, signed absolute difference, separate
original-source/common-source normalized differences and an explicit baseline
ratio. A zero baseline gives `None`, including 0/0; it is not turned into a
zero relative error or an infinite numerical result. Units, exact source
fingerprints, table-coordinate hashes, coverage and incomplete reasons are
retained per case. No multiplicative correction is inferred from this study.

### Artifacts and release boundary

The retained report `outputs/astar_helium_impact_20261031.json` has SHA256
`b45ecd544a5c0fbdf7282d5d935e103f5fedbc3b936b3ff0e9931fd26ff1ceb2`.
Its earlier `*_draft_20261031.json` is superseded: numerical results are the
same, but the final report additionally records mandatory candidate-consistency
checks and the unique four-source roster. Both full candidate validators run;
a failed numerical-consistency status is rejected rather than omitted.

```powershell
.\.venv\Scripts\python.exe scripts/compare_astar_helium_impact.py outputs/parameterized_stopping_100_20261031.json outputs/parameterized_stopping_200_20261031.json outputs/astar_helium_native_controls_20261031.json --native-sha256 75162717801d59ab4048c6a11d9dd9a229f56f90e9d148e6b3939e2f8cd157a6 --output outputs/astar_helium_impact_replay_new.json
```

The existing production/resource snapshot still contains 166 matching files.
The strict OST audit remains six blocking findings in five production tables;
all physical-accuracy, mass-policy, production-replacement, ASTAR-interpolation
and Stage-4 completion flags remain false. This bounded He experiment is not
the full LEO/GEO/HEO acceptance matrix and does not resolve the higher-energy
GCR range or the other positive HZE species.

### Independent forward-quadrature witness

`scripts/check_astar_helium_impact_witness.py` independently reconstructs the CC
and AA coordinate arrays from the pinned native controls and ASTAR archive,
checks their coordinate hashes, and reconstructs the unchanged source
fingerprints. It does not call the transport/LET preimage integrators for its
estimates. The global calculation samples original incident log-energy and
integrates `j(E) * E * d(log E)`, with explicit Al, residual-range and Si domain
masks. Original point densities are interpolated linearly, with no source
normalization or time factor.

Six real cases (CC/AA for SEP PEAK at 0.01 and 1 g/cm2 and GCR PEAK at 10 g/cm2)
have 25 retained reference metrics checked at both 2 and 4 million midpoints.
At 4 million points the largest relative discrepancy among positive-reference
metrics is 0.006909%/0.002139% for SEP at 0.01 g/cm2 and
0.595494%/0.251446% for SEP at 1 g/cm2, respectively CC/AA. The same global
sampling leaves 7.850928%/4.456664% relative discrepancies in narrow GCR tails
at 10 g/cm2. Small errors divided by the whole source do not justify ignoring
these component-level discrepancies; they remain explicitly visible in the
final report.

A second, localized forward calculation resolves those selected supported
high-LET tails without solving individual LET-threshold preimages. Within
each piecewise-linear Si table, all LET >= 1.25 MeV cm2/mg is confined to exit
energies 0.01..1 MeV/n: the lower boundary is the exact table support, and the
maximum on the upper complement (including the interpolated boundary and all
remaining knots) is only 0.716491871 for CC and 0.713128001 for AA. These are
upper bounds for the **represented interpolant**, not physical bounds outside
the table.

The monotone Al range maps this enclosing band to incident intervals
99.690774752..99.715708742 MeV/n (CC) and
99.665416906..99.690050962 MeV/n (AA). Direct forward midpoint quadrature on
these intervals, at 100000 and 200000 points, checks LET thresholds 1.25, 1.4,
1.45 and 1.5. At 200000 points the maximum positive-reference relative errors
are **0.00127260% CC** and **0.00397026% AA**. AA's zero tails above 1.45 and 1.5
match exactly, with relative errors `None`. All threshold sample counts and
absolute errors are retained. The localized integral is not a whole-source
integral, and unsupported incident/low-residual contributions stay unknown.

The final witness is
`outputs/astar_helium_impact_witness_r2_20261031.json`, SHA256
`eb82d25df51ffa8e06904a67d30546426d7d83c42c2f7ecbda5b3a0d3426dd41`.
Its global results are numerically identical to the earlier witness without
localization; the earlier file is historical and its own script pin predates
the added localized checks. The r2 report retains 182 current matching input
and implementation pins. Execution used Python 3.13.5 and NumPy 2.4.6.

```powershell
.\.venv\Scripts\python.exe scripts/check_astar_helium_impact_witness.py outputs/astar_helium_impact_20261031.json --report-sha256 b45ecd544a5c0fbdf7282d5d935e103f5fedbc3b936b3ff0e9931fd26ff1ceb2 --output outputs/astar_helium_impact_witness_replay_new.json
```

Both the impact report and the final witness replay are **byte-identical** to
their retained counterparts. All **135 new focused tests** pass on final code
(70 impact tests and 65 independent-witness tests), recorded in
`outputs/pytest_astar_impact_final_r2_20261031.xml`. Analytic controls include
two-branch LET thresholds, exact count partitions, source scaling and temporal
units, zero shielding versus limited Si support, range-origin invariance,
nonfinite/duplicate input rejection, pinned-file changes and unsupported
localization. Ruff and strict mypy pass all four new code/test files. This
verifies the bounded implementation and selected quadratures, not a normative
accuracy tolerance or universal convergence.

Broad regression: **3714 passed** in 378.44 s, with no failures, errors or
skips (`outputs/pytest_astar_impact_broad_20261031.xml`). Its collection preceded
the eight added localized-witness tests; the final focused run passed all 135
new tests on final code. XML testcase identities give **3722 distinct selected
tests passed**, with 127 overlapping cases; the two run totals must not simply
be added. The 17 expensive GCR/SEP upper-domain tests were explicitly excluded
from the broad run and were not rerun here. A fresh collection finds 3722 tests
with those two files excluded. `git diff --check` and the explicit new-file
whitespace check are clean. Final live OST audit still examines 15 resources
and returns the same six blocking findings. All 182 witness pins remain
current after regression; no production table or physics implementation was
changed during this continuation.

## Experimental heavy-ion stopping controls (2026-10-31)

The next bounded check is recorded in
[iaea_heavy_ion_stopping.md](iaea_heavy_ion_stopping.md). It adds a pinned,
lossless IAEA compilation parser and exact native comparisons for 2612 selected
Al/Si heavy-ion observations, plus seven visually verified RADEF Si checkpoints.
The native calculations and comparison each replay byte-for-byte. The 272 new
tests pass; no production formula, table, coefficient or release status changes.

This is not a declaration of universal agreement: low-energy outliers require
original-source review, and selected experimental coverage ends at 985 MeV/u
for Al and 90.63 MeV/u for Si. The IAEA electronic-stopping convention,
duplicates, missing/range uncertainties, provider-conversion discrepancy and
unverified correction notices are preserved explicitly. The global median
residual is descriptive, not a physical acceptance bound. Stage 4 remains
INCOMPLETE and the full orbital matrix remains open.

Final regression for this continuation: **3994 passed** in 388.36 s, zero
failures/errors/skips (`outputs/pytest_iaea_broad_20261031.xml`). The 272 new
tests are included, not additional to this total. The same 17 expensive
GCR/SEP upper-domain tests remain explicitly excluded. The final strict OST
audit still reports 15 audited resources and six blocking findings; all 166
production/resource hashes match the frozen native-control snapshot.

The 2026-11-01 original-source follow-up is appended to
[iaea_heavy_ion_stopping.md](iaea_heavy_ion_stopping.md). The full Tp1962 paper
confirms measured film losses rather than an explicitly electronic-only Ne
table. Comparing native total stopping does not explain the Ne-Al discrepancy;
it increases the worst selected difference from 85.09% to 88.07%. The original
Ri1983 table/correction provenance and Zhi1998 numerical pages remain unavailable
through ordinary accessible sources. Publisher metadata is not substituted
for those pages. No model, source value, selection, prior numerical report or
qualification flag is changed; the 166 production/resource pins still match.

## Uploaded Ri1983 and Zhi1998 full-text update (2026-11-01)

The preceding access-limited account is historical and is superseded by the
user-supplied PDFs. Both four-page articles have now been read in full and
visually checked; exact attachment paths, SHA256 hashes, printed pages and the
unit arithmetic are recorded in
[iaea_heavy_ion_stopping.md](iaea_heavy_ion_stopping.md#original-source-follow-up-2026-11-01).
Ri1983 (pp. 3234-3237) uses model-assisted extraction for 107Ag in eight metals,
including Al but not Si. Its Table I formula is `Se = k * (v/v0)^(2p)` in
MeV cm2/mg. The correction mentioned by IAEA, and whether compiled values
include it, remain UNKNOWN; the uploaded article does not supply it.

Zhi1998 (pp. 128-131) infers Au/Si electronic stopping from projected ranges,
fixed model nuclear stopping and a mean/projected-range relation. Conversion
of its two in-domain Table 2 values reproduces the IAEA scale to less than
0.02%; the Au197 isotope label remains a compilation input, not an explicit
statement of the article. Correct units do not turn this model-assisted
extraction into an unconditional independent accuracy bound.

The articles are targeted supporting evidence, not new mandatory Stage-4
barriers requiring every historical disagreement to be resolved. The 1%
grid-refinement criterion concerns numerical convergence of specified
integrals, not universal agreement with experiments. Raw archives, selections,
residual reports, coefficients and production tables remain unchanged, with no
fitted correction. This documentation update does not claim a new full test
run or change the INCOMPLETE/release qualification status.

## Supported primary LET attribution (2026-11-01)

The next bounded calculation addresses a practical question raised by the
source review: which species and supported residual-energy intervals supply
the computed LET spectrum? This is a diagnostic decomposition, not a fitted
stopping model, a stopping-error estimate or a new acceptance requirement.

`src/radar/validation/candidate_let_attribution.py` integrates the **original
source measure** through the existing supported primary range mapping. For
exit-energy band `[a,b]` and inclusive threshold `T`, its quantity is
`integral j(E_in) * 1[a <= E_exit <= b] * 1[L_e(E_exit) >= T] dE_in`,
restricted to known Al transport and Si electronic-LET support. Shared energy
endpoints carry no point mass. All rising/falling Si segments are included;
a plateau exactly equal to `T` is included in full. The implementation does
not reconstruct sub-bin structure from an energy-display histogram, divide
by a numerical LET derivative, or apply another time/solid-angle factor.

The threshold partition is checked against the full **supported Si** tail;
known tails outside the requested exit-energy window remain separate. Counts
outside the Si domain have unknown LET, so they enter a separate domain
balance and are not arbitrarily assigned above or below a threshold. Unknown
source tails and unresolved low residual ranges remain unknown. At zero shield,
the unchanged finite source grid starts at 1 MeV/nucleon: zero contribution
below that energy is a fact about that source representation, not proof of
the absence of a physical low-energy spectrum.

`scripts/attribute_candidate_let.py` replays the pinned full-roster candidate
transport report, both raw-table validators and the exact source fingerprints.
It checks the original primary ledgers and existing LET tails before accepting
the decomposition. The new tails have no artificial upper LET cut within the
represented Si table; comparison with the older 1000 MeV cm2/mg window is
allowed only after checking that every represented Si value lies below it.

The calculation retains all 84 positive GCR and 24 positive SEP HZE entries,
both modes, and shields 0, 0.01, 0.1, 1, 3 and 10 g/cm2. Exit-energy edges are
0.01, 0.03, 0.1, 0.3, 1, 10, 100, 1000 and 100000 MeV/nucleon; LET thresholds
are 0, 0.01, 0.1, 1, 10, 50 and 100 MeV cm2/mg. These are declared diagnostic
partitions, not additional device sensitivity models or qualified energy
domains. Source masses, including Au = 195.1, are preserved; element-level
Ag/Au/Ne attribution is not isotope-matched verification against Ag107/Au197/Ne20.

Aggregation requires the exact positive species/source/mode/shield inventory.
GCR/SEP and PEAK/CUMULATIVE are never pooled. Each group's denominator is its
supported all-species LET tail, not all possible physical fluence. A zero
denominator is `null`, not a claimed zero relative uncertainty. Per-species
and aggregate table-grid changes are retained separately so a small sum
cannot hide a larger component change. Two zero counts are explicitly marked;
a zero coarse/nonzero dense pair has no invented finite relative error.

The bands describe **exit energy in Si**, not the energy history in Al.
Consequently they cannot establish the influence of the Ag/Al or Ne/Al
stopping discrepancies. Nor does Au/Si band attribution quantify the result
of changing its stopping curve: a physical perturbation could move particles
across LET thresholds. No model was refitted and no production table was
replaced. All physical-accuracy, error-bound, Al-path and normative-release
flags remain false.

### Full-roster attribution results and reproduction

The final report is `outputs/candidate_let_attribution_20261101.json`, SHA256
`dfad58a11b7b2b5d871a11b85622eb756b60dfb12bbbba17e8167cd493bedfd8`.
It contains **1296 cases per table grid**, 24 separate source/mode/shield
aggregates per grid, and their energy-band/LET-tail and species contributions.
All original source fingerprints, primary ledgers and existing supported LET
tails reproduce. All 162 production/resource pins declared by this frozen
transport reference and the 11 attribution
implementation pins are unchanged before/after and match the retained files.

The maximum relative coarse-to-dense change in a positive **species**
energy-band/LET-tail integral is **0.7432868765%**: GCR Na, CUMULATIVE,
3 g/cm2, exit energy 0.1..0.3 MeV/nucleon, LET >= 10 MeV cm2/mg
(130.422027111 -> 131.391436923 cm^-2). The maximum aggregate-band change is
**0.4561289920%**: SEP CUMULATIVE, 3 g/cm2, exit energy 0.01..0.03,
LET >= 1 (177.841403847 -> 178.652590050 cm^-2). The denominator is the
coarse value in each case. No coarse-zero/dense-positive pair occurs. These
are selected diagnostic partitions on fixed external spectra, not proof of
source-grid, orbit, native-backend or physical convergence. The maximum
normalized threshold-partition residual is 1.930e-16; all known
below/above-energy-window tails are zero for these particular full-domain
bands, while unresolved transport counts are still retained.

For example, on the dense table and **PEAK** inputs behind 0.01 g/cm2:

| Source | Exit 0.01..0.1 MeV/n contribution to all supported Si counts | Same exit band contribution to supported LET >= 1 MeV cm2/mg |
|---|---:|---:|
| GCR | 0.00985436% | 1.04874984% |
| SEP | 1.89299745% | 13.12578915% |

These are fractions of the separately declared denominators, not percentage
errors. In the same GCR case, Au supplies 1.95045437% of the supported
LET >= 50 tail despite its very small share of all supported counts. Neither
the low-energy region nor a rare element can be discarded solely because its
total-count fraction is small. Conversely this does not turn a source-paper
stopping residual into that fraction of the spectral error. The fixed
external source state, finite support and unchanged fractional source masses
remain essential qualifiers.

```powershell
.\.venv\Scripts\python.exe scripts/attribute_candidate_let.py outputs/candidate_transport_full_r2_20261031.json --reference-sha256 707115dd6f10572a2b865702a6457aef540e6160f973ea7ea8d3984825c98e44 --output outputs/candidate_let_attribution_replay_new.json
```

The initial development run reached every case but correctly refused to write
an output because helper/runner hardening changed implementation pins during
that run. The final run used the frozen implementation and completed normally;
the rejected development run is not counted as a successful replay.

### Verification and retained limits

There are 168 new tests (76 helper, 92 runner). The final focused selection
passes **346 tests** including the existing candidate transport, LET and
runner checks (`outputs/pytest_let_attribution_final_20261101.xml`). Ruff and
strict mypy pass all four new implementation/test files. The tests cover
both branches, threshold-equal plateaus, partial support, original-source
versus display-bin integration, both temporal units, range-origin invariance,
missing/duplicate species, source/mass/units mutation, numerical flags, zero
denominators, changed pins and exclusive report writing.

The independent forward witness is retained as
`outputs/candidate_let_attribution_review_20261101.py` (SHA256
`a7f84c3846446d7a8bf3a64a50fa3a66f2918e94fb4d2420f7245fa441074f65`)
and `outputs/candidate_let_attribution_review_20261101.json` (SHA256
`2761bea6450880a1f85a7b4d8aca6265c4d381372f6ab4a15810e80f440b7dca`).
Its comparison side directly samples incident energy with NumPy interpolation
and masks, without using the helper's `integral_on_exit`. It checks 51 metrics
in each of two synthetic cases, with shields 0 and 6.3 g/cm2, at 2 and 4 million
midpoints. At 4 million the largest positive-reference relative discrepancies
are 0.00104% and 0.00590%; all zero tails match. The plateau at LET = 2 and
the next representable threshold above 2 are checked separately. The witness
replays byte-for-byte; this does not assert full real-table replay or physical
accuracy. All seven witness source/implementation hashes are unchanged.

A separate direct aggregation of the old frozen per-species LET report
reproduces **336** new group/threshold totals, with maximum relative arithmetic
difference 2.074e-16. It does not call the new aggregation function. The newer
IAEA native-control snapshot independently retains 166 production/resource
pins (rather than the older transport reference's 162); all 166 still match.
The live strict OST audit remains 15 resources, six errors in five tables,
`release_ready=false`. No production formula, coefficient, resource or release
flag was changed by this continuation.

The final broad regression passes **4162 tests in 416.41 s**, with no
failures, errors or skips (`outputs/pytest_let_attribution_broad_20261101.xml`).
This includes the 168 new tests, not an additional count. As in the previous
broad run, the 17 expensive tests in `test_gcr_upper_domain.py` and
`test_sep_upper_domain.py` were explicitly excluded, not rerun or claimed
passed here. The full LEO/GEO/HEO acceptance matrix remains open. Final
whitespace checks and retained report/source hashes are unchanged.

## Native NIST ESTAR 1.2 electron/Al reproduction (2026-11-01)

The electron aluminium table's remaining provenance finding is now closed.
This is a source/build reproduction result, not a new experimental validation
of ESTAR or a qualification of the complete electron shielding operator.
No numerical production CSV, transport formula or coefficient changed.

The retained official archive is
`research/nist_star_20261031/STAR.tar.gz` (444183 bytes, SHA256
`b8241beb0e721b6b89dd316e32b23ac6c80e9a8908bd43da1570216de51de2c6`).
NIST's [official download page](https://physics.nist.gov/PhysRefData/Star/Text/download.html)
identifies STAR 1.2, permits redistribution and derivatives with appropriate
notices, and warns that the archive revision need not equal every web revision.
The new Python reader carries an attribution/derivative notice; the unchanged
native fixture has the adjacent `tests/fixtures/nist_estar_NOTICE.txt` notice.
the Fortran and input files are unchanged. The old MS-DOS executables were not
used. This work makes no claim of identity to an unspecified current web build.

`scripts/reproduce_nist_estar.py` verifies the archive and six source/data
members, then builds `EDCONV`, `CONVERT` and `ESTAR` separately at both `-O0`
and `-O2`. Both converters use the same compiler as ESTAR; the generated
direct-access files are 122400 bytes (`UEDAT`) and 74772 bytes (`UCOMP`).
Input selects default-grid range output, material 13 (Al), and accepts the
unchanged source I-value of 166.0 eV. Density is 2.6989 g/cm3 and the retained
source atomic mass is 26.981539. Neither masses nor material data are adjusted.

The compiler is GNU Fortran 15.2.0 (Ubuntu 15.2.0-16ubuntu1), run in WSL Ubuntu.
Two official Ubuntu packages were downloaded and extracted inside
`research/nist_star_build_20261101/`; no global compiler installation, system
package upgrade or source patch was performed:

| Retained package in `toolchain_packages/` | Bytes | SHA256 |
|---|---:|---|
| `gfortran-15-x86-64-linux-gnu_15.2.0-16ubuntu1_amd64.deb` | 13634176 | `11ef9271f616d35f523c4d53173c2fa5f95a02703332e6a38899b960cca5ebe8` |
| `libgfortran-15-dev_15.2.0-16ubuntu1_amd64.deb` | 972612 | `7b711bfd286d4148c23bfdde476349ad3da2c60f5cfe545106c9c49f1e612c51` |

The flags are `-std=legacy`, `-ffixed-line-length-72`,
`-static-libgfortran`, the selected optimization level and explicit compiler
search prefixes. There is no fast-math, real-type promotion or replacement
of the source's mixed-precision literals. Legacy dummy-array-size warnings
are retained verbatim; successful execution is not a bounds-clean claim.
The report records before/after integrity checks for the compiler, selected
compiler/linker dependencies, source members, input CSV and Python
implementation. Generated outputs are hashed after execution. The 18 command
logs preserve arguments, input, output, warnings and exit codes; those log
files are retained separately and their hashes are not included in the report.
This is **not a hermetic toolchain build**: environment, startup objects and
all shared/system libraries are not comprehensively pinned.

### Preserved quantities and exact comparison

The source uses `TLOSS=CLOSS+RLOSS`: total electron mass stopping power is
collision plus radiative stopping, not the electronic-plus-nuclear ion split.
Energy is MeV per electron, stopping is MeV cm2/g and CSDA range is g/cm2.
No additional density or atomic-mass multiplier belongs in those conversions.
Radiative stopping includes energy carried away as bremsstrahlung: it is not
a local absorbed-dose response or an electronic-only LET. This reproduction
does not validate dose, SHIELDOSE or bremsstrahlung transport.

The internal range calculation uses all **113** energy nodes from 0.001 to
10000 MeV; the exported **81** nodes span 0.01 to 1000 MeV. The original
`MOUT=1` path uses hardcoded `ER` values and does not read `ENG.ELE`; that
file is a pinned supporting archive artifact, not an active grid input here.
The original
log-spline/21-point quadrature and `RG(1)=0.5*T(1)/TLOSS(1)` are retained,
not reconstructed by integrating the rounded exported stopping rows. The
export loops are 17..56 and 57..97. The legacy readme's lower output bound
does not override the actual source grid. This output is not evidence of
table support outside the exported 0.01..1000 MeV interval.

`src/radar/validation/nist_estar.py` requires both full output pages, the
material/units headers, exact grid and monotone ranges. It preserves all
seven printed tokens per row. The four-significant-digit collision,
radiative and total fields are tested for compatible rounding cells, not
incorrectly required to sum exactly after separate rounding. That rounding
check is not a physical uncertainty bound.

Each of the 81 CSV rows matches exactly in Decimal for energy, total stopping,
CSDA range and redundant range in mg/cm2, with no tolerance. The mg column
also equals 1000 times the g column exactly. Any numerical mismatch, including
an mg-only mismatch, makes the command unsuccessful. Structural defects fail
closed. The parser alone cannot authenticate a producing executable, and
neither parsing nor the runner automatically promotes a release status.
Exactness here is to the four-significant-digit printed ESTAR columns, not
to unprinted native floating-point arrays. Nor does node equality validate
RADAR's separate between-node log-log interpolation or the entire shielding
calculation; those numerical/operator checks remain separate.

Final report: `outputs/nist_estar_reproduction_final_20261101/report.json`,
SHA256 `4486fbed685636712d8264f4c9aa17e9022f2b6bd6d33b2e0a80c310c5deeeee`.
Both O0/O2 native outputs and `tests/fixtures/nist_estar_al_1_2.txt` are
byte-identical (7558 bytes), SHA256
`e2806353a9eff3c1e3902dee93ac61414e0857fd22225c5941c9dc2b862063b4`.
The first development build is retained separately and is not the passport's
final implementation-pinned evidence. An independent regex/Decimal check,
without importing the new reader, reproduced all 81 by four numeric fields
for both builds with zero differences; an independent Fraction check found
no incompatible printed stopping-power rounding cells.

To replay on the retained local WSL toolchain, use a new output directory:

```powershell
wsl.exe -d Ubuntu --cd '<RADAR_WSL_ROOT>' --exec python3 scripts/reproduce_nist_estar.py --compiler '<RADAR_WSL_ROOT>/research/nist_star_build_20261101/toolchain/usr/bin/x86_64-linux-gnu-gfortran-15' --compiler-prefix '<RADAR_WSL_ROOT>/research/nist_star_build_20261101/toolchain/usr/lib/gcc/x86_64-linux-gnu/15' --compiler-prefix /usr/lib/gcc/x86_64-linux-gnu/15 --output outputs/nist_estar_reproduction_replay_new
```

This command rebuilds and compares; it does not overwrite the production CSV
or install a compiler. Another host must supply its own explicit compiler
path/prefixes and will produce a separately fingerprinted report. Equality of
native outputs, not identity of all host-dependent report paths, is checked.

### Narrow passport transition and remaining gates

`stopping/electron_al_range.csv` remains byte-identical (2304 bytes, 81 rows),
SHA256 `b53413110cd515bb11bc15fc074acfbb5eb0f5dcc473279af8b25ecc7f2f7d71`.
Its passport is now `verified_derived`, release eligible as an **input table**,
and references the official archive plus the final build report. The historical
`ЛПЭ.xlsx` ancestry and unchanged CSV formatting are explicitly retained.
The other 22 resource records are JSON-identical to the old manifest.

The new manifest release is `radar-normative-2026-11-01.1`, SHA256
`a5106e891023db087f0a8d5a7f57f392d7b53bb4fba80f94e05cb23c6b2f3803`.
Its exact predecessor is saved as
`research/nist_star_build_20261101/integrity_manifest_before.json`, SHA256
`87087bb9df1bac9f141e67cc02143d52483df079abce4671ab5b0d3d6edc7854`.
Older transport/LET/IAEA reports include that old manifest fingerprint. They
are preserved historical snapshots: their current-tree replay intentionally
detects this passport change. No old result is repinned or called freshly
replayed under the new manifest; a future numerical replay needs new evidence.

The strict OST audit now checks 15 resources and has **five** errors: provisional
proton/Al range, proton/Si stopping and HZE/Al range; quarantined HZE/Si LET
and its independently detected multidecade discontinuity. The strict global
audit checks 23 resources and has ten errors. Both retain `release_ready=false`.
Five data findings are not the entirety of the remaining Stage 4 work: full
source/orbit/shield acceptance, model questions and secondary transport gates
are independent. No GOST or AE8/AP8 resource was promoted. No HZE candidate,
COSRAD comparison, dose or SEE functionality acquires eligibility from ESTAR.

The new reader and mocked-build tests pass **135 tests** (89 + 46), with Ruff
and strict mypy clean across their four implementation/test files. The focused
manifest/loader/electron/contract selection separately passes 68 tests; the
counts overlap the broad suite and must not be added to its total. Tests cover
header/grid/units loss, small numeric mismatches, source changes, archive
links/duplicates, subprocess failure, selected dependencies, before/after
fingerprints, exclusive outputs and non-automatic release flags.

The final broad regression passes **4306 tests in 388.88 s**, with no failures,
errors or skips (`outputs/pytest_estar_broad_20261101.xml`). The same 17 expensive
tests in `test_gcr_upper_domain.py` and `test_sep_upper_domain.py` were explicitly
excluded; they were not rerun or claimed passed in this continuation. Ruff and
strict mypy pass the five changed/new Python implementation and test files.
Final source, fixture, production CSV, manifest and native report fingerprints
match their retained values. The independent review's two wording/attribution
corrections are applied: command logs are not falsely described as hashed in
the report, and the unchanged native fixture has an adjacent notice.

## LET implementation boundary and proton table contract (2026-11-01)

The ESTAR provenance closure above is for electrons in aluminium. It is not
a replacement of the heavy-ion silicon LET data or a decision to abandon
SRIM-derived tables. The current production GCR/SEP paths still obtain
`stopping/hze_si_let.csv` and `stopping/proton_si_let.csv` through
`load_normative_shielding_tables`. Their established immediate source is the
identified `ЛПЭ.xlsx`; original SRIM run files/version/settings are not yet
retained as authoritative upstream evidence. An attribution to SRIM is not
the same as having the reproducible original export.

Two distinct changes have been investigated only in `radar.validation`:
CATIMA-generated candidate tables, and a conservative energy-to-LET mapping
which integrates the original distribution over all preimages of each LET
interval. The latter uses electronic stopping; the legacy production HZE
operator uses its table's total-stopping column and a bin-centre histogram
mapping. This is a real difference in both numerical treatment and physical
quantity, not merely a different implementation name. Neither candidate has
been selected or imported into production. Its results remain diagnostic,
`INCOMPLETE`, and ineligible for normative export. Choosing a source does not
by itself settle the required LET definition or validate either operator.

One descriptive error was identified during this boundary audit:
`proton_si_let` declared `log_log_piecewise` interpolation in its passport,
while `ProtonSiLetTable.total_let_at_energy` actually performs piecewise linear
interpolation in both energy and stopping. The correction describes the
existing runtime as `linear_linear_piecewise`; it does not change that runtime,
the CSV, any calculated value, or the provisional release status. This is
contract consistency, not a claim that the current interpolation is physically
qualified or preferable to another law.

The next independent proton-table check must also preserve a still-open
semantic distinction. The legacy `p_Al` importer converts a workbook column
named `R` from mm to g/cm2 using `rho * R / 10`; its source passport does not
establish whether this is CSDA path range or projected range. The retained
native PSTAR source outputs both `RANGE(c)` and `RANGE(p)`, with
`RANGE(p) = DETOUR * RANGE(c)`. A difference from only the CSDA column is not
proof of an error in SRIM or a reason to replace the legacy data. Future
comparison must keep both ranges, and Si electronic/nuclear/total stopping,
separate. An absent source nuclear column is unknown, not evidence of a
physically zero nuclear contribution. Existing numerical differences quoted
earlier are historical comparisons, not newly reproduced error estimates.

Recovering and checking the original SRIM exports remains a valid way to
restore table provenance. An independent PSTAR or CATIMA comparison does not
authorize silently relabelling the workbook, fitting a correction or changing
the production model. No physical table or computational formula is replaced
by this contract correction, and the five strict OST data findings remain open.

The previous `.1` manifest is retained byte-identically at
`research/proton_let_contract_20261101/integrity_manifest_before.json`, SHA256
`a5106e891023db087f0a8d5a7f57f392d7b53bb4fba80f94e05cb23c6b2f3803`.
The `.2` manifest SHA256 is
`b5f1b04cd3da03ac458bfbbb85a3716622debfe234aabbd933f9d3ea2bd87592`.
Only the proton Si entry's interpolation description and verification note
changed, alongside the release identifier; the other 22 entries and every
registered numeric resource hash are unchanged. This is another explicitly
versioned metadata transition, not a repinning of older numerical reports.

Three added tests check the retained provisional status/CSV hash, a synthetic
midpoint that distinguishes linear from log-log interpolation, and an actual
loaded table interval. The agent's focused selection passes 73 tests; the
root's broader LET/loader/contract selection passes **125 tests**
(`outputs/pytest_let_boundary_root_20261101.xml`). These selections overlap and
are not additive. Ruff and strict mypy pass both edited Python test files.
The previous 4306-test broad run belongs to the `.1` snapshot; it was not
repeated or claimed for this metadata-only correction. Strict OST remains
15 resources / five errors / `release_ready=false`; strict global remains
23 resources / ten errors / `release_ready=false`.

The user reports that SRIM is not currently installed and can download it.
A new installation would provide fresh, versioned comparisons; it cannot by
itself establish which executable/settings produced the historical workbook.
No download, installation, production replacement or SRIM redistribution has
been performed in this continuation.

## User-supplied SRIM module access and isolated smoke check (2026-11-01)

The user subsequently supplied the extracted package at
`<LOCAL_SOFTWARE>/SRIM-2013` and explicitly approved the first
isolated module launch. This supersedes the preceding installation-availability
note, not any physical qualification decision. No installer, registration
command, security setting or system-package modification was executed.

The bundled `SR Module/HELP- SR Module.rtf` documents file-driven execution.
An isolated copy of `SRModule.exe`, `SCOEF03.dat`, `SNUC03.dat`, `VERSION` and
the help file reads a new ASCII `SR.IN`; the original inputs are not edited.
The first probe exited with 0xC0000135 and produced no result. The package's
compressed `SRIM-Setup/SRIM-AutoSetup/MSVBVM50.dl_` was then expanded into a
new repository-local directory and the DLL copied beside the isolated module.
No DLL was copied into Windows or registered. This local dependency resolved
the observed startup failure; this does not qualify the full GUI installation.

Retained dependency identities:

- User installer `SRIM-2013-Pro.exe`: SHA256
  `07f7f46d1a97635bd3a34ae26c80df385473833a4e99706b2c4dbbd77005cb40`.
- Module `SRModule.exe`: SHA256
  `a679e55defdb12ebe39c02ca0562016b79ad73105dcaeddc9e8ba40f57c16288`.
- Compressed bundled VB runtime: SHA256
  `c03a4b1528792c3ab5ba5818a9c1aa0d93e21be2551cea67777722a9edad2f9b`.
- Expanded private `MSVBVM50.dll`: 1355776 bytes, version 05.02.8244 (SP2), SHA256
  `8611dc1b60ae5c383bba6cb3ffd8a51aeebff23b95844f0ab3d6e5ecd0fadc84`.

The successful case is hydrogen Z=1, A=1.008 u in solid Al (A=26.981539,
density 2.6989 g/cm3), 1000..10000 keV total ion energy, stopping-unit selector
5 (MeV/(mg/cm2)). The native process exited 0 and generated 27 rows spanning
1..10 MeV. The output retains separate electronic/nuclear stopping and
explicitly **Projected Range**, not CSDA range. A read-only check confirms
positive finite fields, ordered energy and projected range, declared units
and endpoints. This is a file-interface smoke check, not numerical agreement
with the historical workbook or physical accuracy validation.

The output uses decimal commas under the current locale despite the ASCII
input's decimal points; no Windows locale was changed. Its title says
`Calculation using SRIM-2006`, and its version line says `SRIM-2012.01`.
The parent package's `Data/VERSION` says `SRIM-2013.00`. Normal module
`SCOEF03.dat` and `SNUC03.dat` match the parent Data copies, but the underscored
`_SCOEF03.dat` differs. None of these labels was rewritten or silently equated;
underscored alternative files were not substituted. The help's special-mode
nuclear-stopping warning remains relevant before any explicit-grid experiment.

Successful raw output:
`outputs/srim_probe_h_al_privatevb_20261101/probe_h_al.txt`, SHA256
`cea7b04a00688166d7e51443d4be338d6193aab4a990d1da75352f34ca641592`.
Probe report:
`outputs/srim_probe_h_al_privatevb_20261101/probe_report.json`, SHA256
`8a36981c28b0be665c2315be2135ec5e2f39966dda808594f026d323f2972ea6`.
The failed first report is retained separately. These are development smoke
artifacts, not a validated general SRIM runner/parser or redistributable
runtime package. Upstream source/settings, version semantics and distribution
conditions must be addressed before any production use. The production
manifest remains `.2` with its unchanged SHA256 and all release blockers.

## Repeated native SRIM controls and legacy-table comparison (2026-11-01)

The continuation adds an isolated normal-mode control runner
`scripts/run_srim_controls.py`, a strict lossless printed-output reader
`radar.validation.srim_output`, and a direct-node comparison reader
`radar.validation.srim_comparison`. None is imported by a production radiation
pipeline or changes a normative table. Executable, normal coefficient/version
files and private VB runtime are checked against fixed SHA256 values before
launch; input, output, command records and implementation/production identities
are retained. Each native run has a separate working directory. Existing outputs
cannot be overwritten, and a timeout terminates only that runner's own child.

All six controls below were run twice after the implementation was frozen.
Every native process exited zero, the two raw outputs for each case are byte
identical, and all printed values/metadata have identical semantic hashes.
There are 638 compared nodes across the six cases, or 1276 native rows including
the repeats. These are repeated calculations with one implementation, not
independent physical validation.

| Control | Actual total-ion energy, MeV | Native/matched nodes | Legacy nodes above this range |
| --- | --- | ---: | ---: |
| H in Al | 0.1–1000 | 105 | 52 |
| H in Si | 0.1–1000 | 105 | 0 |
| He-4 in Al | 1–12000 | 107 | 16 |
| He-4 in Si | 1–12000 | 107 | 16 |
| Fe-56 in Al | 1–12000 | 107 | 16 |
| Fe-56 in Si | 1–12000 | 107 | 16 |

The input masses are H=1.008 u, He=4 u, Fe=56 u. Targets are solid Al,
A=26.981539 u and density=2.70 g/cm3, and solid Si, A=28.0855 u and
density=2.3290 g/cm3, both with compound correction=1 and stopping selector=5.
These are explicit control parameters, not recovered original workbook settings
or the package's default densities. `Data/ATOMDATA` instead records Al=2.7020
and Si=2.3212 g/cm3, with rounded masses 26.982 and 28.086 u. No fitting to
legacy values or silent substitution of defaults was performed.

The initial H/Al objective of 0.1–10000 MeV failed. A bounded retry's native
window was inspected using Windows Computer Use and displayed
`Run-time error '6': Overflow`; it did not produce a numeric table. The observed
message, exact case, command hashes and ownership/timeout scope are retained at
`research/srim_controls_20261101/native_overflow_observation.json`. A separate
8000 MeV endpoint also timed out, but its message was not inspected and is not
claimed identical. Lower-range probes through 1000 MeV succeeded. The internal
cause and maximum usable energy are unresolved; the final H/Al run is explicitly
restricted to 1000 MeV, retains the original intended upper bound of 10000 MeV,
and reports `planned_control_range_covered=false`. No failed interval is filled
by zeros, interpolation or extrapolation. Package documentation's stated upper
limit of 10 GeV/u does not establish successful module operation or accuracy.

The normal 26-mantissa-per-decade energy sequence is now checked in full for
each fixed control. Matching endpoints and monotonicity alone are insufficient:
missing, extra or changed internal nodes are rejected. The general text reader
checks complete headers/footers, all six numeric columns, supported units,
ordered energies/projected ranges and consistent retained metadata. It supports
the observed comma decimals without changing Windows locale. Exact original
tokens and raw hashes survive; relative comparison percentages are calculated
at 80 decimal digits, not used as an acceptance tolerance. The optional summary
contains rounded Decimal ratios; the full comparison ledger remains authoritative.
Independent rational-arithmetic review of the development outputs checked all
six columns of all 1276 rows, including GeV/MeV and distance-unit transitions.
The guard hardening did not change those rows or their semantic hashes.

Comparison is exact at shared coordinates, allowing only an explicitly recorded
eight-binary64-ULP energy-coordinate roundtrip for legacy E/A conversion. This
is not tolerance on stopping or range values. Every unmatched legacy/native
node is retained. Fe has 11 exact Decimal and 96 roundtrip coordinate matches
per material; H and He coordinate matches are exact. Neither missing nuclear
provenance in proton Si nor untested ions is interpreted as a physical zero.

Bounded observations from the final run:

- H/Al total stopping differs by at most 0.078645% across 105 matched nodes;
  the legacy range versus native projected range differs by at most 0.398407%.
- H/Si electronic stopping differs by at most 0.050353%. Comparing its legacy
  incomplete total with native electronic+nuclear stopping is separately
  labelled; it must not qualify the missing legacy nuclear source.
- He/Al and Fe/Al range differences are at most 1.448652% and 0.980393%
  over these controls. Legacy range meaning is still unverified, so these are
  differences, not established physical errors or CSDA accuracy estimates.
- For He/Si and Fe/Si, electronic stopping through 10000 MeV total energy
  differs by at most 0.495050% and 0.576924%. Above that boundary the retained
  legacy electronic values collapse, whereas the fresh normal-mode outputs do
  not. The discrepancy is in the retained numbers, not a new E/u conversion.

For example, all electronic values below are MeV*cm2/mg; energy is **total ion
energy**, not MeV/u. Legacy values are displayed rounded here, not in the ledger.

| Energy, MeV | Legacy He/Si | Native He/Si | Legacy Fe/Si | Native Fe/Si |
| ---: | ---: | ---: | ---: | ---: |
| 9000 | 0.00666 | 0.006663 | 2.76 | 2.765 |
| 10000 | 0.00666 | 0.006663 | 2.58 | 2.579 |
| 11000 | 4.06671417e-7 | 0.006672 | 0.000720626344 | 2.426 |
| 12000 | 3.74954821e-7 | 0.006689 | 0.000665781688 | 2.296 |

The module still labels itself SRIM-2006 / SRIM-2012.01 inside the user package
labelled SRIM-2013.00. The local help warns generally that SR Module nuclear
stopping differs from full GUI tables; that warning is not confined here to the
underscore-file special mode. No such alternative files were used. Native
footer conversions have small differences from the rounded printed density
(Al 26.999 versus 10*rho=27; Si 23.289 versus 23.29); these are preserved rather
than silently repaired. Projected range is never relabelled CSDA.

Final artifacts:

- `outputs/srim_controls_final_20261101/report.json`, SHA256
  `edf9fe709600d7c1dc4a042e2706418b95dac4bca9fa341cc8f3e2a9ef633565`.
- `research/srim_controls_20261101/final_evidence.json`, SHA256
  `0f9b8390ca15cf85661e6070fc3cae6867a65ac7178309fe13c3a7e0f47adee7`.
- Overflow observation SHA256
  `32a6255fa88744f2b009e2a04438c9ae78b3e24dfc9352837bd649639977e305`.

The frozen focused suite passes **196 tests**; Ruff and strict mypy pass all six
new implementation/test files. The earlier broad run was deliberately stopped
when the independent review required additional guards; it is not reported as
passing. The replacement broad run uses the final frozen implementation and
its completed result is recorded below.

The completed frozen broad selection passes **4505 tests in 468.56 seconds**:
`outputs/pytest_srim_stage4_frozen_20261101.xml`. It includes the 196 focused
tests above, which are not additive. The 17 tests in `test_gcr_upper_domain.py`
and `test_sep_upper_domain.py` were explicitly excluded from this run; collection
confirmed 4522 tests in the complete suite. Those 17 are not claimed rerun or
passed here. Ruff and strict mypy also pass the final six-file implementation
and test snapshot. Independent final review verified every retained native
input/output/command hash, parsed row, exact control grid and semantic identity,
and confirmed that the three parser/runner guard findings were resolved.
The broad JUnit XML SHA256 is
`9fa6210fb6a06a347a526e9fd84e5c3b41145e947f675f0a0702de1f97b919f6`.
After the tests, the three native-control implementation hashes and all six
protected production-file hashes still match the frozen native report.

This establishes a reproducible local comparison route, not a replacement
release. The next scope is to resolve the module/full-table version and
high-energy limitations, generate independently checked coverage for the
remaining 13 legacy HZE ions and missing energy nodes, and explicitly settle
electronic/total stopping and projected/CSDA transport semantics before choosing
replacement resources. Covering all 15 legacy ions would still not by itself
qualify the required GCR/SEP species and mass-mapping contract. The old table's
ancestry is not recovered by a new installation. No physical source, profile,
coefficient or production formula was changed.
Strict OST remains 15 resources / five errors; global remains 23 resources /
ten errors. `INCOMPLETE`, `release_ready=false` and the `.2` manifest SHA256
`b5f1b04cd3da03ac458bfbbb85a3716622debfe234aabbd933f9d3ea2bd87592`
are unchanged. Stage 4 is not closed; dose and SEE development has not begun.

## Native SRIM extension across the legacy roster (2026-11-01 continuation)

The new `scripts/run_srim_coverage.py` preserves the preceding frozen runner,
parser and comparison implementation byte-for-byte. It checks the pilot report
SHA256 and every old implementation/production dependency before execution.
Three fixed catalogs are supported; there is no general energy-list, coefficient
substitution or automatic fitting mode. Every successful control requires two
complete outputs with the exact expected energy grid and identical semantic
hashes. A native timeout/nonzero exit is retained as `NATIVE_FAILED`, an invalid
table as `INVALID_OUTPUT`, and differing successful repeats as `REPEAT_MISMATCH`.
Failed output never receives a numeric comparison. Each case is checkpointed,
so a failure does not silently remove an ion from the report. Mutation checks
also run after failures. Only the owned child is stopped on timeout.

The frozen native results are:

| Catalog | Fixed scope, total ion energy | Repeated complete controls | Failed controls |
| --- | --- | ---: | ---: |
| baseline | 15 legacy HZE ions, Al and Si, 1–12000 MeV | 30 | 0 |
| upper | 14 legacy HZE ions other than He, Al and Si, 1–45000 MeV | 28 | 0 |
| boundary | H/Al through 2000, 4000, 6000 MeV; H/Si through 4000 MeV | 3 | 1 |

This is 123 native attempts: 122 successful runs and one timeout. Each complete
case's repeats have identical raw output bytes. All 107 shared native rows of
each of the 28 upper controls exactly match its baseline output, including every
numeric field and printed token: 2996 shared rows in total. No change to the
stopping values was introduced by extending the upper control boundary.

Selecting the upper series for 14 ions and the baseline for He yields direct
comparisons at **1829 of 1845 legacy nodes per material**. All 123 legacy nodes
for each non-He ion are matched; He has 107 matched and 16 higher unmatched
nodes. This is coverage of the *old table only*, not coverage of the full GCR/SEP
source inventory or energy domain. In particular, 254 of the 270 quarantined
Si tail nodes above 10000 MeV total ion energy now have fresh native counterparts;
the other 16 are the unfilled He tail. No CSV has been replaced with these data.

The old electronic-stopping collapse at 10000→11000 MeV is not reproduced by
any of the **15** freshly calculated ions. Across that step the native ratio
S(11000)/S(10000) ranges from approximately 0.939658 to 1.001351; the old ratio
ranges from about 0.000061062 to 0.000284343. Up to and including 10000 MeV,
the largest electronic-stopping difference across all 15 ions is 0.576924%
(Fe), using the legacy printed value as denominator. These are exact-node
comparisons of printed outputs, not uncertainty bounds, proof of source
ancestry, or independent experimental qualification of SRIM.

The extension also reveals a separate discrepancy in **Al ranges**. For all
14 upper-catalog ions, the largest absolute difference from the legacy range
occurs at 45000 MeV total ion energy. Native projected ranges are 14.027–31.346%
smaller there, whereas all 14 maximum absolute differences at or below
10000 MeV are below 1%. Selected direct-node results are:

| Ion | Total energy, MeV | Legacy range, cm | Native projected range, cm | (native - legacy) / legacy, % |
| --- | ---: | ---: | ---: | ---: |
| C-12 | 45000 | 388.9043773845562 | 267.00 | -31.34559148 |
| Fe-56 | 45000 | 11.3180589710826 | 9.525 | -15.84246005 |
| Ni-58 | 45000 | 9.356428158628226 | 8.044 | -14.02702117 |

Distance units and the common density conversion were independently checked.
For example, the native C value is printed as 2.67 m, hence 267 cm and
720.9 g/cm2 at 2.70 g/cm3; the legacy value corresponds to
1050.041818938302 g/cm2. The discrepancy is not an m-to-cm or density-conversion
artifact. It does **not**, however, establish which range is physically correct:
the native quantity is explicitly projected range, while the legacy range
definition, its upstream construction and the high-energy accuracy remain
unverified. It must not be conflated with the separate Si electronic-stopping
collapse or corrected by an empirical rescaling.

A read-only XML inspection of the source workbook at
`<LOCAL_DATA>/ЛПЭ.xlsx` (SHA256
`d4ed7928331a0b7abcdfe1b585504b136809a3f6823f220edfed8967c28ec684`)
found no stored cell formulas in any of its nine worksheets and no external-link
parts. The Downloads copy has the same hash. Thus this workbook does not retain
the formulas that could settle the high-energy construction of `i_R_Al`;
numerical patterns alone are not evidence of a particular extrapolation law.
The workbook was neither edited nor recalculated.

The proton boundary controls supersede only the earlier *tested* 1000 MeV
limit, not the frozen six-case report. H/Al through 2000 and 4000 MeV and H/Si
through 4000 MeV all repeat successfully. Their common rows match the earlier
controls exactly. H/Al through 6000 MeV again times out with an empty output.
No UI message was inspected for this new timeout, so it is not relabelled as
the earlier observed 10000 MeV `Overflow 6`. The maximum usable energy and
the internal cause remain unresolved. A separate exploratory He/Si run through
30000 MeV total energy also times out with no numeric result. It is retained
at `outputs/srim_boundary_he4_si_30000_20261101/result.json`, outside the 123
formal catalog attempts. These outcomes are not physical stopping-law failures.

### Limits that prevent automatic table replacement

The local package documentation is now explicitly separated by scope:

- `SRIMHelp/SRTrouble.rtf`, line 9, says the table program will not create a
  table beyond 10 GeV/u. `Data/VERSION.rtf`, line 115, records that limit's
  historical introduction. A stated range is not an execution guarantee.
- `SRIMHelp/help2.rtf`, line 9, is worded for **TRIM**. It warns of sparse
  experimental stopping data above 1 GeV/u and potential inaccuracy in
  1–10 GeV/u. This is retained as a package caution; it is not silently converted
  into an independently established accuracy interval for SRModule.
- That file and `SRIMHelp/srhelp1.rtf`, line 9, warn about omitted energy loss
  into nuclear reactions above 1 MeV/u. This is distinct from the elastic
  nuclear-stopping column. The help's general small-percentage statement is
  not adopted as a guaranteed bound for RADAR materials or spectra. Existing
  separate nuclear-interaction operators are neither removed nor validated
  by the present stopping-table checks.

The full old He tail ends at 45000 MeV total / 4 = **11250 MeV/u**, already
beyond the table program's stated limit. Thus it is intentionally excluded from
the upper native catalog, not extrapolated. The successful He baseline ends
at 3000 MeV/u; the remaining 14 baseline ions end at or below 1000 MeV/u.
Some successful upper controls enter the package's high-energy caution region,
which is recorded per case and does not become a physical-accuracy claim.

The verified source inventory also remains materially larger than the old
table: 84 positive GCR HZE species and 24 positive SEP HZE species, compared
with 15 legacy species. Thus **69 GCR and nine SEP positive species** still
have no member in this legacy control roster. The nine SEP gaps are F, Na, Al,
P, Cl, K, Sc, Ti and V. Source-declared zero species remain distinct from these
missing positive contributions.

Ten legacy integer masses differ numerically from the literal source parameters:
Ne 20/20.2, Mg 24/24.3, Si 28/28.1, S 32/32.1, Ar 40/39.9, Ca 40/40.1,
Mn 55/54.9, Fe 56/55.8, Co 59/58.9 and Ni 58/58.7 (legacy/source). This
comparison deliberately retains the old integer masses; it does not select a
new physical isotope policy or round source parameters. SEP A/Q is not A/Z.
The mass inventory's six source-file hashes were rechecked without changing it.

Current computation centers extend to 10^4.9 MeV/u for GCR and 10^4.1 MeV/u
for SEP; the GCR integral boundary is 100000 MeV/u. These grids therefore
extend beyond the package's 10000 MeV/u statement. Their units and computational
status must not be confused with SRIM's total-energy input bounds or with an
OST-imposed cutoff. Production grids were not narrowed to make the native
controls pass. Independent coverage/upper-tail treatment, the transport mass
policy, and projected-range versus CSDA semantics remain separate decisions.

### Retained artifacts and verification

- Baseline report: `outputs/srim_legacy_baseline_final_20261101/report.json`, SHA256
  `8ed401964181371ebb508f16bec41aa5b3600b136f47f763be4c0596f448c9cc`.
- Upper report: `outputs/srim_legacy_upper_final_20261101/report.json`, SHA256
  `3bbb3992b432074860b049820b065d8f47dd19d5cfc9cd074d8541f41754c029`.
- Boundary report: `outputs/srim_legacy_boundary_final_20261101/report.json`, SHA256
  `4bd0adfcf7dbe1320ace41f34113cf50ffa846c5c438375b63a591acbb1ddd24`.
- Combined evidence: `research/srim_coverage_20261101/evidence.json`, SHA256
  `a4d381e77be680caacf4f834117fa4d2a3cc5442ef7ea7d681578e9ca0c4eb99`.

The combined evidence retains each ion's direct-node coverage and mismatch
statistics, boundary values, 80-digit ratios, mass differences, missing positive
species, exact source-help identities and current profile/global audits. It
does not create a transport resource. The focused SRIM/integrity/mass selection
passes **296 tests**, including 43 new synthetic runner tests. Ruff and strict
mypy pass the two new Python files.

The completed new broad selection passes **4548 tests in 460.38 seconds**:
`outputs/pytest_srim_coverage_broad_20261101.xml`, SHA256
`c934d4a45c41ee9616a5e85951a4769cc1af771f6a0d5af4762925612858c86f`.
The focused 296 tests are included, not additive. The 17 tests in
`test_gcr_upper_domain.py` and `test_sep_upper_domain.py` were explicitly
excluded; no rerun or pass is claimed for them. This is a new run of the frozen
implementation, not reuse of the earlier 4505-test result. After completion,
all 18 distinct protected file identities recorded across the three native
catalogs still match, as do the new runner/test and combined-evidence hashes.

Independent read-only review revalidated the three native reports and the
combined evidence. It checked all six numeric columns of **14024 valid native
rows** using rational arithmetic and independently recomputed **42232 comparison
metrics**, including 80-digit percentages. Input/output/command hashes, runtime
and production identities, per-case grids, all 2996 overlapping rows and the
unqualified status flags agreed. No discrepancy was found; no native program
was executed by that reviewer.

All results remain `INCOMPLETE` and ineligible for normative export. The
production manifest and numeric tables are unchanged: strict OST still has
five errors across 15 resources; global has ten across 23. The next work is
bounded qualification of mass/ion coverage and a supported high-energy method,
with explicit treatment of the He/proton native failures and range definition.
Simply replacing the quarantined CSV would not close these conditions.

## Range/stopping operator contract (2026-11-01 continuation)

This continuation checks the physical meaning required by the existing operator,
without changing a production table, source spectrum, isotope policy or formula.
The relevant OST 134-1044-2007 Appendix I.7 pages (printed 141-143; PDF 147-149)
discuss continuous-slowing-down engineering models and SRIM/ITRAN. They do not
identify the undefined historical workbook range column as CSDA. The pages were
visually inspected; the retained PDF `tmp/pdfs/gcr_domain/ost1044.pdf` has SHA256
`232c84bf7fb6c2d3aec11c44827bcf01a6b6964366a8d0059ff36cd5b143a9e1`.

The independent definitions in the [NIST quantity appendix](https://physics.nist.gov/PhysRefData/Star/Text/appendix.html)
and [PSTAR/ASTAR methods, section 3.5.4](https://physics.nist.gov/PhysRefData/Star/Text/programs.html)
(accessed 2026-11-01) distinguish integrated CSDA path range from projected
penetration depth. Their definitions, rather than an assumption that two
programs agree, give the following contract for the selected CSDA operator.
This is a derivation for that approximation, not a claim that OST explicitly
prints these equations or that CSDA is the only allowed transport method.

For total ion kinetic energy T (MeV), areal path x (g/cm2), and positive total
mass stopping S(T) (MeV cm2/g):

```text
dT/dx = -S(T)
R(T_b) - R(T_a) = integral[T_a..T_b] dT / S(T)
R(T_in) = R(T_out) + x
```

For a fixed, explicitly declared mass parameter A and epsilon = T/A,
`dR/d(epsilon) = A/S(A*epsilon)`. Dividing the stopping or the mass range by A
is not part of this coordinate conversion. The differential-spectrum Jacobian
is `R'(epsilon_out)/R'(epsilon_in)`, equal to
`S(T_in)/S(T_out)` only for a consistent CSDA range/stopping pair. This
Jacobian is already implicit in RADAR's conservative bin-preimage integration;
it must not be multiplied into the bin-integrated result again. None of these
relations introduces a mission-duration multiplier or changes PEAK/CUMULATIVE.

`shielding/proton_al.py` and `shielding/hze_al.py` map output-bin boundaries by
adding the shield thickness to the tabulated range and applying its inverse.
The numerical mapping can operate on any accepted monotone table; that alone
does not establish its CSDA meaning. Their current table structures have no
range-kind field and do not enforce the integral relation above. In addition,
the proton survival/secondary operators integrate cross sections divided by
stopping power. A physically inconsistent R/S pair would make those operators
use different effective areal paths. Existing provisional-data release blocks
remain necessary; the following diagnostic is not a replacement release gate.

### Bounded consistency diagnostic

The new `validation/range_stopping_consistency.py` compares **increments** of a
supplied range with anchored integrals of reciprocal total stopping. Two
explicit interpolants are evaluated analytically on each adjacent energy
interval: linear S versus energy, and log-linear S versus log-energy. Neither
is represented as SRIM's internal interpolation. Their difference is numerical
sensitivity to these choices, not a bound on interpolation or physical error.

There is no extension below the first point, no assertion that R(first)=0,
no extrapolation, no fit of the legacy data, and no numerical pass threshold.
Projected, undefined and caller-declared CSDA ranges retain their original
meaning even when differences are small. The native electronic and elastic
nuclear stopping components are summed; nuclear-reaction attenuation and
fragmentation are not added to that stopping sum or qualified by this check.

`scripts/audit_srim_range_stopping.py` replays the three frozen native catalogs,
their protected file identities, original inputs, outputs and complete repeated
grids, without launching SRIM. It selects He/Al from the baseline, the other
14 legacy ions/Al from the upper catalog, and H/Al through 4000 MeV from the
boundary catalog. It retains the failed 6000 MeV H/Al case as unavailable.
All energies remain total ion MeV. Each selected case compares:

- native projected-range increments against native total stopping;
- legacy undefined-range increments against the same native total stopping,
  explicitly labelled a **cross-dataset** comparison, not internal legacy
  consistency or proof of its construction.

Shared-node energy matches retain the prior exact/ULP-roundtrip policy, never a
new tolerance on physical values. Bands cover the observed shared support and
its available portions below/above 10000 MeV. An absent band is `NOT_COVERED`,
not zero. Printed precision is retained in the source artifacts; many digits in
the arithmetic output are not a statement of physical precision.

### Mass-policy boundary

The existing source/transport mass audit is not superseded. It already records
the 69 GCR / nine SEP missing positive species and ten legacy/source mass
differences. Source traces preserve literal mass parameters; the production
HZE table compatibility checks still compare rounded species metadata, whereas
the separate candidate-transport diagnostic rejects unequal literal masses.
The native SRIM header's mass in amu must not be silently converted to an
integer isotope declaration, or its projected range passed to a CSDA candidate
adapter. This continuation deliberately remains on total-energy native nodes
and chooses no new isotope distribution or source-to-transport mass policy.

### Retained results and interpretation

For the **10000..45000 MeV total-energy increment**, not the absolute endpoint
range, selected signed differences from the stopping integral are:

| Ion | Native projected increment / linear-S integral - 1, % | Legacy increment / linear-S integral - 1, % | Native projected increment / log-log-S integral - 1, % | Legacy increment / log-log-S integral - 1, % |
| --- | ---: | ---: | ---: | ---: |
| C-12 | 0.146146 | 54.055891 | 0.143668 | 54.052078 |
| Fe-56 | 0.026790 | 20.758952 | -0.014215 | 20.709448 |
| Ni-58 | 0.036587 | 17.997030 | -0.005689 | 17.947164 |

Across all 14 upper-catalog ions, this high-band native difference has absolute
magnitude below 0.27% for both selected interpolants. The legacy increments
exceed those integrals by about 18..54%. These percentages have a different
denominator and anchor from the earlier 14..31% endpoint comparison and must
not be substituted for that earlier metric. For He, the available 10000..12000
MeV increment gives about +0.46% native and +8.23% legacy; its missing upper
tail is not filled. The H comparison extends only through 4000 MeV.

This bounded result gives no basis to explain the entire high-energy legacy
discrepancy merely by renaming projected range as CSDA. It still does not
establish what stopping model constructed the old range, or the independent
physical accuracy of the new stopping model. Source construction remains open.

Small aggregate differences do **not** imply small local differences. Across
the complete selected support, the largest absolute local native difference
under log-log-S interpolation is **19.155070%** for Cr-52, 1.60..1.70 MeV total
energy; the corresponding cumulative endpoint difference can be much smaller.
Printed range precision, interpolation and projected/path-range differences
remain separate possible contributions. The diagnostic does not attribute the
entire difference to a physical-model error or assign a guaranteed uncertainty.

The resulting implementation direction for RADAR's selected CSDA approximation
is therefore: construct consistent **anchored path-range differences from the
chosen total stopping data**, retain the native projected range separately,
and qualify material/mass/energy coverage before a production adapter. Using
SRIM stopping for that construction is not a switch to CATIMA, a fit to COSRAD,
or a proof that the unresolved low/high-energy regions are negligible. No
absolute stopping threshold or production replacement is established here.

### Reproduction and tests

The final report `outputs/srim_range_stopping_final_20261101.json` and its replay
`outputs/srim_range_stopping_replay_20261101.json` are byte-identical, SHA256
`8a2deb5b889db0bbff18a0c4732bf0f4a1c228bb3f81737dedd82ea318e65cfd`.
They contain 16 selected controls and 7740 interval records; these counts include
overlapping bands and native/legacy variants, not independent physical samples.

```powershell
.venv/Scripts/python.exe scripts/audit_srim_range_stopping.py --output outputs/srim_range_stopping_new_replay.json
```

Original reports, protected files and retained input/output/command files are
hashed before use and rechecked at completion. The output path must be new.
The two implementation hashes are:

- `validation/range_stopping_consistency.py`:
  `3cdce16f4064834097aa47ab8faf9c4a97989993c7f364b67ce0b0872d22b545`;
- `scripts/audit_srim_range_stopping.py`:
  `7ad8bc1e7594d494e0baa88a02f2f9d88991690772e141e4b9e613d3fa1f2c1c`.

The new focused suite passes **107 tests** (70 mathematical diagnostics and
37 replay/provenance tests). Ruff and strict mypy pass all four implementation
and test files. An additional 375-test focused selection includes the 70
mathematical tests and existing SRIM, primary transport and integrity tests;
these counts overlap and are not additive. A constant-cross-section synthetic
test verifies consistency of path thickness and attenuation for a consistent
R/S pair, and demonstrates the separate diagnostic failure for an inconsistent
pair without modifying production acceptance behavior.

Separate direct formulas at 200-digit precision recomputed all 139320 stored
numeric metrics in the 7740 interval records; maximum scaled difference was
4.86e-80. Independent SciPy quadrature for both interpolants (15480 integrals)
agreed within 3.56e-15 relative. Another 120 varied-scale mathematical cases
agreed within 2.93e-12 relative. Manual reading checked 1951 native table nodes
and their 7834 references in the overlapping report bands. These calculations
checked the development report; all numerical selected-control data in the final
report are identical. The only report differences are implementation hashes,
the wording "tabulated nodes", and removal of an unsupported URL fragment from
the NIST reference. This was checked recursively across all report fields.

Independent runner review identified and closed three replay-guard issues:
unsorted matched energies are rejected **before** band filtering, Decimal unit
conversion uses a private fixed context, and retained native files are checked
again after the calculation. Tests cover these cases, incomplete/failed runs,
unit factors, tail reanchoring and exclusive output creation. No native program
was executed in this continuation. The strict OST audit was rerun: five errors
across 15 resources, `release_ready=false`, manifest `.2` unchanged.

The frozen broad regression completed with **4655 passed in 452.02 seconds**
and no failures, errors or skips in the selected set. Its JUnit report is
`outputs/pytest_range_contract_broad_20261101.xml`, SHA256
`8f27d403cabbe41c2570f5dff384f53baf1d171492f97746fcce41b9f7d2d043`.
It includes the 107 new tests and the earlier focused selections, which are not
added again. As in the preceding broad run, the 17 tests in
`test_gcr_upper_domain.py` and `test_sep_upper_domain.py` were explicitly excluded
and are not claimed rerun or passed here. After completion, both diagnostic
implementation hashes, all three parent-report hashes and all 18 distinct
protected identities still match. Global audit remains ten errors across
23 resources. Neither this regression nor the consistency report closes Stage 4.

## Separate anchored SRIM range candidate (2026-11-01 continuation)

The preceding range/stopping contract is now implemented as a separate,
opt-in diagnostic object, not merely a comparison of endpoint tables. No
production resource, source spectrum, isotope policy or release gate changed.
No new native SRIM process was launched: the three frozen control catalogs
were replayed from their retained raw files.

### Candidate construction and interpolation

`validation/srim_anchored_range.py` provides `AnchoredSrimRangeTable.from_native`
with an explicit `linear_stopping` or `loglog_stopping` choice. Each constructs
an anchored path-range coordinate from the native electronic plus elastic
nuclear stopping, multiplied by 1000 to obtain MeV cm2/g. Density is **not**
applied again to this mass-stopping integral. Native projected range in cm and
its density conversion remain separate, unchanged comparison quantities.
Both choices are retained; neither is selected as the production model or
described as SRIM's internal interpolation or a physical uncertainty bound.

For each observed energy interval the range and its inverse use the **same
analytic stopping interpolant**. The inverse is not obtained by linearly
interpolating the integrated range nodes, which would introduce a different
inter-node slowing-down law. The constructors derive the range arrays; callers
cannot inject precomputed range values into the constructor. Raw and semantic
native hashes, mass, units, both stopping components and projected data survive
serialization.

The anchor `R(T_min)=0` is only a coordinate offset. It is not a statement that
the positive first observed energy is physical rest. For an incident energy
within observed support:

- zero shielding returns that exact energy;
- `x = R(T)` returns the positive observed `T_min`;
- `x > R(T)` returns `None`, meaning unresolved below table support, not zero
  exit energy, physical stopping, or a zero contribution to a spectrum;
- out-of-domain inputs and nonrepresentable positive energy/range increments
  are rejected, not extrapolated, clamped or silently rounded away.

Private Decimal contexts cover construction, validation, coordinate conversion
and evaluation. Working anchors retain 110-digit arithmetic; public coordinates
are rounded to 80 digits. Those digits describe numerical bookkeeping, not
accuracy of printed SRIM values. Tests specifically cover interior values that
would round to a tabulated endpoint and positive shields that would otherwise
return unchanged energy. Native mass below Z is rejected. The data provider and
source-mass policy still require independent qualification.

### Native inventory and coverage audit

`validation/srim_native_coverage.py` connects the exact native observations to
the actual OST source roster, separately for Al and Si. It does not construct
an isotope declaration from native amu, or use rounded spectrum labels to hide
mass differences. Source A, SEP A/Q, native amu, and both numerical energy-axis
quotients (`T/native_mass` and `T/source_A`) are retained separately. Their
rounded quotients are not proof of equivalent physical coordinates.

Each source/material row distinguishes a model-declared zero, missing positive
species, missing material, or an observed energy interval with unresolved mass
policy. Even numerical mass equality does not establish an isotope distribution.
Duplicate Z/material tables and conflicting Al/Si masses are rejected. H is
reported separately and never included in the HZE coverage denominator; its
transport uses total MeV per proton, not energy divided by native mass 1.008.

The fixed native selection contains **32 outputs**: 15 HZE species in Al and
Si, plus H in each material. Actual total-energy support is:

| Selected observations | Native nodes per material | Total kinetic energy, MeV |
| --- | ---: | ---: |
| He, baseline | 107 | 1..12000 |
| Each of the other 14 legacy HZE species, upper | 123 | 1..45000 |
| H, boundary | 122 | 0.1..4000 |

There are 1951 Al and 1951 Si native rows. The separate 32 **Al candidate
variants** are 16 projectiles times two interpolants; their 3902 node entries
repeat the Al nodes twice and are not additional native calculations.
All selected HZE species have both materials, but only 15 of 84 positive GCR
species and 15 of 24 positive SEP species are represented. Thus 69 GCR and
nine SEP positive species remain missing. In each source roster five selected
masses are numerically equal and ten differ; **zero** mass policies are qualified.
The seven GCR and three SEP source-declared zeros are retained separately.
The failed H/Al 6000 MeV control remains in the parent ledger without numerical
substitution. Neither interval observation nor source presence establishes
full source-spectrum or shielded-transport coverage.

### Bounded transport checks

`scripts/build_srim_range_candidate.py` constructs both Al candidates and checks
the six specified shields: 0, 0.01, 0.1, 1, 3 and 10 g/cm2. This is **not** a
GCR/SEP/ERB spectrum convolution, orbit calculation, mission-time integration,
LET production adapter, nuclear-reaction transport, or release calculation.

All 23412 combinations of candidate, native incident-energy node and shield
are retained, including unresolved exits. There are also 3870 geometric
midpoint forward/inverse checks. Their maximum scaled roundtrip discrepancy
is about 9.54e-80, against the explicitly numerical 1e-60 check; that threshold
is not a physical acceptance tolerance. Composition-of-shields and exact anchor
boundaries are additionally tested on synthetic tables.

| Shield, g/cm2 | Candidate supported exits | Candidate unresolved exits |
| ---: | ---: | ---: |
| 0 | 3902 | 0 |
| 0.01 | 2212 | 1690 |
| 0.1 | 1634 | 2268 |
| 1 | 1142 | 2760 |
| 3 | 916 | 2986 |
| 10 | 655 | 3247 |

These are **node counts across two interpolation candidates**, not flux
fractions or numbers of stopped particles. The source distribution has not been
convolved. Nuclear attenuation and fragmentation are not included.

The comparison route uses linearly interpolated native projected-range
differences after subtracting its own first-node range. It therefore differs
from the candidate in both range construction and inter-node interpolation;
differences must not be attributed solely to projected-versus-CSDA physics.
Each route's support status remains separate. At He, 400 MeV total incident
energy and 10 g/cm2, the log-log stopping candidate yields about 6.237389 MeV,
while the projected comparison lies below its observed anchored support.
That one support-mask disagreement is retained, not zero-filled or treated as
evidence of physical stopping. On jointly supported nodes the largest relative
exit-energy difference is about +119.477%: H at 27.5 MeV behind 1 g/cm2 gives
1.224891 MeV (log-log candidate) versus 0.558096 MeV (projected comparison).
This near-boundary point sensitivity is **not** an error estimate for SRIM,
the mission spectrum, dose or SEE rate. It does not override the earlier
small integrated range/stopping differences.

### Reproducibility and implementation checks

Final and replay reports are byte-identical:
`outputs/srim_range_candidate_final_20261101.json` and
`outputs/srim_range_candidate_replay_20261101.json`, SHA256
`f423c5da9eb45e8489787257a84479a0489f406c866ed12802a97313fb065b53`.

```powershell
.venv/Scripts/python.exe scripts/build_srim_range_candidate.py --output outputs/srim_range_candidate_new_replay.json
```

The runner pins the preceding range/stopping report, the source-mass manifest,
their implementations and source files, then validates every parent native
report and retained input/output/command identity. It checks identities again
after construction. Required missing controls cause failure; failed unselected
controls remain explicit. The output path must be new, including under a
concurrent writer. The candidate/projected comparison requires identical raw
and semantic native identities.

New implementation hashes:

- `validation/srim_anchored_range.py`:
  `214df435b5c37f81ecf2a3e10504d9268be3983ee81d8c73421394cca3cfe7fa`;
- `validation/srim_native_coverage.py`:
  `423ef13b43dfb44be7065fbdebda4e0d90882453fe9d927e4f76043d54d7b099`;
- `scripts/build_srim_range_candidate.py`:
  `569b425f69e3b868783b405ec85852e49a63ec80ab8ce561e2dae2e1adac0078`.

The frozen focused suite passes **141 new tests**: 85 anchored-range, 30
source/native coverage, and 26 runner tests. Ruff and strict mypy pass all six
new implementation/test files. Read-only review independently checked source
rows against the existing mass audit, all retained provenance identities,
700 additional inverse and 2100 sequential-shield checks per interpolant, and
the permanent `INCOMPLETE`/false qualification flags. The representability and
cross-native-identity findings were fixed and covered by regression tests.

An independent read-only numerical review manually decoded all six columns and
original tokens of 3902 native Al/Si rows across the 32 selected controls,
and checked 192 retained input/output/command identities for their repeats.
Separate 200-digit analytic formulas recomputed the 3902 candidate-node entries,
3870 off-node checks and 23412 shield-path rows. All 57386 numeric checks agreed
within a maximum scaled difference of 4.913e-80. All 12951 unresolved exits
remained explicit, and all 236 source/material rows and energy/mass quotients
matched the pinned 118-row source manifest. This review applies unchanged to
the byte-identical development, final and replay reports. It does not establish
physical accuracy or full source-spectrum coverage.

The next bounded work is to extend native input coverage to the missing OST
species and explicitly declared source-mass parameters, retaining those as
diagnostic model parameters rather than asserting an isotope distribution.
Before any production connection, the remaining energy-domain limitations and
supported-spectrum balance must also be evaluated. The current candidate does
not justify replacing a CSV, silently truncating a source, or narrowing the
mission/grid to make coverage pass. Strict OST remains five errors across
15 resources; global audit remains ten across 23, manifest `.2` unchanged.

The frozen broad regression completed: **4796 passed in 454.59 seconds**,
zero failures/errors/skips in the selected set. The report is
`outputs/pytest_srim_candidate_broad_20261101.xml`, SHA256
`39833c52dea1a6c333e6f02eeac78e5bd40cb14d65475068a576df9d4f61a674`.
It includes all 141 new tests; focused counts are not additive. The same 17
upper-domain tests in `test_gcr_upper_domain.py` and `test_sep_upper_domain.py`
were explicitly excluded, not claimed rerun or passed. After the broad run,
all 13 candidate/source pins, all 18 distinct parent protected-file pins,
the three native reports and every retained native input/output/command hash
were rechecked unchanged. Stage 4 remains INCOMPLETE.

## SRIM literal-source roster campaign (2026-11-01)

This block extends **observed native species and numerical mass-parameter
coverage**, not the production transport domain or physical qualification.
No production CSV, source spectrum, coefficient, interpolation policy, or
integrity-manifest entry was changed. The earlier diagnostic Al range model
and native-output parser were reused without modification.

### Fixed inputs and execution gate

`validation/srim_source_catalog.py` independently reconstructs all 118 source
rows from the pinned OST implementations: 91 GCR and 27 SEP HZE rows, including
their ten declared-zero contributions (seven GCR and three SEP). The union of
positive contributions contains 84 GCR species and includes all 24 positive
SEP species. The source-mass manifest remains
`outputs/stopping_mass_manifest_20261031.json`, SHA256
`f367a1b5f047451e40ea37d4cb03139ff6cbdb7c7eb9333888b44c11d46250fc`.
Both the manifest and its six source dependencies are checked before and after
catalog construction. All source fields, including separate SEP A/Q, are
compared; positive and declared-zero contributions are not conflated.

The fixed catalog has 168 controls: each of the 84 positive species in Al and
Si, with the literal source A passed as a **diagnostic numerical amu input**.
Fractional parameters are neither rounded nor presented as selected isotopes.
In particular, the printed-source values Tl=200.6, Pb=200.4 and Bi=207.2 are
retained, not silently replaced by periodic-table values. This campaign does
not resolve their physical interpretation or establish a documentary erratum.

Every control requests the normal SRModule mode over 1 to 12000 MeV **total ion
kinetic energy**, output selector 5, and the same single-element targets as the
preceding controls. SR.IN sets energy endpoints, not an explicit node list;
validation requires the exact expected 107-node native grid. The source folder is the
user-supplied SRIM-2013 package; native output continues to identify itself as
SRIM-2006 / SRIM-2012.01. Package naming is not substituted for that executable
identification. Only the five previously pinned normal-mode files and the
previously approved private MSVBVM50.dll are copied into isolated directories.
No installer, system dependency, locale, special energy-list file, or original
input is changed. Native working copies are private evidence, not production
distribution assets or a new redistribution-license claim.

`scripts/run_srim_source_campaign.py` first runs F19, Ne20.2, Fe55.8 and U238
in both materials twice. Full mode requires the SHA-pinned, complete eight-case
pilot, unchanged source/implementation pins, exact canonical inputs and grids,
and distinct retained repeat directories. It reuses those eight pilot records
once and executes only the other 160 cases twice. Each native child has its own
30-second bound; failure remains explicit and does not become a zero-valued
particle contribution. Files are checked during and after execution, including
copied files on a failed attempt. Outputs and checkpoints are exclusive new
files. A successful native status is only an execution/control gate.

### Retained native results

- Pilot: `outputs/srim_ost_source_pilot_20261101/report.json`, SHA256
  `3563c374050ecc38593d9b701fca352e66fd9655c8e438d8063f0874308ed739`.
  Eight complete cases, 16 new successful attempts.
- Full roster: `outputs/srim_ost_source_full_20261101/report.json`, SHA256
  `0302ffd3f4de1acd6f5e687d85128e1641e93c5570f6ebb0aec2af2f1506279b`.
  168 complete cases, eight reused pilot cases and 320 new successful attempts.

Thus there are **336 actual native attempts in total**, not 352, and 168 unique
ion/material controls. All repeat pairs have identical semantic and raw-byte
outputs. There are 17976 unique native energy rows (8988 in each material), or
35952 rows when both repeats are counted. No native timeout, invalid output or
repeat mismatch occurred in this fixed campaign. Ne20.2 and Fe55.8 are retained
exactly in both native ion headers; accepting the header parameter is not an
independent proof of the executable's internal mass dependence.

The pilot was also read independently without importing the campaign runner
or output parser: all 1712 raw rows / 10272 numerical cells, MeV/GeV and A/um
conversions, input parameters, output units, footer factors, command ledgers,
56 retained identities/checkpoints and 35 protected-file pins agreed. This was
the gate used before starting the remaining cases, not an after-the-fact waiver.
The subsequent full read-only review checked all 35952 raw rows / 215712
numerical cells, literal masses of all 84 species, input and command records,
36 protected-file pins, 1176 retained identities, and isolated normal-file/DLL
copies. All 168 repeat pairs agreed. All ten common He/C/N/O/Cr controls in
Al/Si also matched the preceding legacy baseline on all 107 nodes, all six
native columns, semantic hashes and raw bytes. This agreement tests replay
consistency, not independent physical accuracy.

`scripts/summarize_srim_source_campaign.py` performs read-only replay without
importing or launching a native runner. It authenticates the complete pilot
and full-campaign ledgers, canonical inputs, raw outputs, commands, checkpoints
and copied normal files; reconstructs statuses, counters and source coverage;
then derives the same two declared Al stopping-interpolation candidates.
Only complete matching repeat pairs contribute numerical observations. The
summaries are byte-identical:

- `outputs/srim_ost_source_full_summary_20261101.json`;
- `outputs/srim_ost_source_full_summary_replay_20261101.json`;
- SHA256 `06011df28f365fc93a13917099d435bff95dbede36c120de8539b62daea50f1b`,
  18731713 bytes each, 3566 evidence-file identities unchanged before/after.

Both materials now have numerical source-mass matches for all 84 positive GCR
and 24 positive SEP species, with no missing positive species or numerical mass
differences. The seven/three declared zeros remain separate. There are 168 Al
candidates (84 species times two interpolants) and 84 Si native controls.
**Zero species have a physically qualified mass policy.** Si stopping is
retained as native evidence; no production LET adapter was created.

An additional read-only check across the 168 Al interpolation candidates
independently evaluated direct Decimal
logarithm/power formulas rather than the existing interval-integral helper:
17976 positive stopping nodes and exact node inverses, 17808 strictly
increasing range intervals and analytic integrals, and 17808 midpoint
roundtrips. With numerical scale `max(1, abs(reference))`, maximum integral
discrepancy was below `4.851e-80` and midpoint inverse discrepancy below
`9.524e-80`, against a `1e-70` tolerance. These decimal arithmetic residuals
are not physical uncertainty estimates. The range anchor and both declared
interpolation choices remain diagnostic.

Read-only reproduction (the destination must not already exist):

```powershell
.venv/Scripts/python.exe scripts/summarize_srim_source_campaign.py --campaign-report outputs/srim_ost_source_full_20261101/report.json --report-sha256 0302ffd3f4de1acd6f5e687d85128e1641e93c5570f6ebb0aec2af2f1506279b --output outputs/srim_ost_source_new_replay.json
```

### Energy-domain boundary remains open

For a source parameter A, this campaign supports only
`[1/A, 12000/A] MeV/u`. The unchanged default GCR evaluation grid has 81 centers
from 1 through `10^4.9` MeV/u, with integral products closing at 100000 MeV/u.
The unchanged SEP HZE grid has 165 centers from 1 through `10^4.1` MeV/u.

| Source parameter | Observed native support, MeV/u | Supported GCR centers | Supported SEP centers |
| --- | --- | --- | --- |
| He, A=4.0 | 0.25 to 3000 | 57 / 81 | 140 / 165 |
| Fe, A=55.8 | 5/279 to 20000/93 (about 0.01792115 to 215.05376) | 38 / 81 | 94 / 165 |
| U, A=238.0 | 1/238 to 6000/119 (about 0.00420168 to 50.42017) | 28 / 81 | Not an SEP source |

Every one of the 84 positive GCR and 24 positive SEP species still has an
unsupported upper-source interval. Supported-center counts range from 28 to
57 for GCR and 93 to 140 for SEP; these are **sampling counts, not flux
fractions or estimates of neglected physical contributions**. All first source
centers lie inside the native interval, but shielded exits below `1/A` and
continuation to physical zero energy are separate unresolved questions. The
range-coordinate anchor at the first native energy does not answer them.

The next bounded gate is an explicit source-to-transport energy-support ledger
and an evidence-backed strategy for the missing intervals. No extrapolation,
source-grid narrowing, mass substitution or physical stopping classification
is justified by this campaign. Native projected range remains separate from
the integral of electronic plus elastic nuclear stopping; nuclear-reaction
attenuation is not supplied by that stopping sum. A complete observed roster
does not replace the quarantined Si LET table or qualify a production adapter.

### Frozen implementation and software checks

- `validation/srim_source_catalog.py`:
  `ea57ea410ce10fc9f2fc317d4ec967cfdb7531eecfcd6ada9c069c1308f1982a`.
- `scripts/run_srim_source_campaign.py`:
  `d676df7b61a0eca5c1f75756aa390d060d1d8af57d165a1be55556993924ec59`.
- `scripts/summarize_srim_source_campaign.py`:
  `7ad98463fea65bcfb4f3919a28928c923ff1c14e204e472f85eac2ba2a43cf9f`.

The catalog and runner passed 104 tests together before the full native
campaign; the standalone summary has 53 additional synthetic replay tests.
Ruff and strict mypy pass all six new implementation/test files. Runner tests
never launch SRIM. They cover failed/incomplete pilots, repeat reuse, exact
fractional masses and grids, source/copy mutation, command ledger corruption,
checkpoint disagreement and final revalidation of earlier artifacts.

The frozen broad regression completed: **4953 passed in 514.35 seconds**,
zero failures/errors/skips in the selected set. It includes all 157 new tests;
the focused counts are not additive. The report is
`outputs/pytest_srim_source_broad_20261101.xml`, SHA256
`bf9ceb5ff33149cfd269acc21f2913ccb074637d5f0c1d98cdc896aeeb55ff8f`.
As in the preceding baseline, the 17 tests in `test_gcr_upper_domain.py` and
`test_sep_upper_domain.py` were explicitly excluded, not claimed rerun or
passed. This software suite is not the final Stage-4 spectrum-acceptance matrix.
After native execution and summary replay, the main review independently
rehashed all 36 campaign-protected paths and 3566 summary evidence paths with
no changes. The original SR.IN remains at SHA256
`5eca8ae70716f9c2396d36e7daa3224f8575fbbebbc564ef2771f84a9b31af9d`.

Strict profile/global audits still report respectively five errors across
15 resources and ten across 23, with `release_ready=false`. The unchanged
integrity manifest remains `radar-normative-2026-11-01.2`, SHA256
`b5f1b04cd3da03ac458bfbbb85a3716622debfe234aabbd933f9d3ea2bd87592`.
Stage 4, physical mass qualification and full source-energy coverage remain
INCOMPLETE; normative export is not enabled.

## Explicit SRIM/source energy-support ledger (2026-11-01)

`validation/srim_energy_support.py` and
`scripts/audit_srim_energy_support.py` now turn the preceding domain warning
into a per-source, per-ion, per-material record. This is read-only replay of
the retained native campaign, not another SRIM run, new transport physics, or
an assertion that the current source-grid endpoints are prescribed by OST.

The runner authenticates the fixed native replay implementation at SHA256
`7ad98463fea65bcfb4f3919a28928c923ff1c14e204e472f85eac2ba2a43cf9f`, replays
the pilot and all full-campaign artifacts, and records hashes of its own
implementation and the unchanged runtime grid definitions. Only complete,
matching native repeats become observations; failed controls remain in the
parent ledger and absent tables produce explicit UNKNOWN support.

The audit retains all 118 source species rows in both materials: **236 rows**,
of which 216 describe positive source contributions and 20 describe declared
zeros. Declared zeros are not missing calculations. Exact source/native
numerical mass equality is required for this comparison; a different mass,
duplicate table, or contradictory Al/Si mass is not silently reconciled.

Native support is the closed interval `[T_min/A, T_max/A]`. Both the original
numerator/denominator and a display quotient are retained. Runtime float grid
values are recorded as `Decimal(str(value))`; classification uses exact
rational arithmetic on those recorded coordinates, not the rounded display
quotient. Open/closed gap endpoints are explicit. This precision is for
unambiguous interval bookkeeping, not a physical uncertainty claim.

The evaluated-center span and the finite GCR integral domain are separate:
GCR ends its evaluation grid at `10^4.9` but closes the declared integral at
100000 MeV/u. SEP records only its evaluated span ending at `10^4.1`; no
physical or infinite-source upper bound is inferred from that endpoint.
Residual energies below native support and absolute stopping at zero remain
separate unresolved questions.

| Source, in each material separately | Positive species | Center/species entries | Supported | Above native support | Missing-table UNKNOWN |
| --- | --- | --- | --- | --- | --- |
| GCR | 84 | 6804 | 2997 | 3807 | 0 |
| SEP | 24 | 3960 | 2515 | 1445 | 0 |

Al and Si have the same interval counts because the campaign requested the
same total-energy bounds, not because their stopping powers are equal. No
source centers are below native support in this fixed case. Every positive
species retains an upper gap. These counts are **not fractions of fluence,
dose, LET-spectrum area, or SEE rate**; no spectral convolution was performed.

Final and replay reports are byte-identical:
`outputs/srim_energy_support_20261101.json` and
`outputs/srim_energy_support_replay_20261101.json`, SHA256
`de2d5ad77468ba9dc726eee8f521ab8f84ba0f3b3596f26b87875e5d3aefae8f`.

```powershell
.venv/Scripts/python.exe scripts/audit_srim_energy_support.py --campaign-report outputs/srim_ost_source_full_20261101/report.json --report-sha256 0302ffd3f4de1acd6f5e687d85128e1641e93c5570f6ebb0aec2af2f1506279b --output outputs/srim_energy_support_new_replay.json
```

New implementation hashes:

- `validation/srim_energy_support.py`:
  `aed71f95c32507ac9cb85443c7210febbae96152e9522a7eae9bfb9240c26cea`;
- `scripts/audit_srim_energy_support.py`:
  `e417eed9f515d6420fab4523c3aa7955ea77275e4a262eb6d1c38ad29fbcb9ea`.

The focused regression passed **317 tests in 36.50 seconds**, including all
103 new tests (69 interval/roster tests and 34 replay-runner tests). Report:
`outputs/pytest_srim_energy_support_20261101.xml`, SHA256
`19141cb36411ae4031a03c922c38ea553bd39459217d9b14ba715f734d1110b4`.
Ruff and strict mypy pass all four new files. The earlier 4953-test run is
historical evidence, not a full-suite rerun for this continuation. Independent
interval review checked 4356 domain/native interval pairs and 226512 point
memberships, including disjoint intervals and endpoint-only intersections.
Independent cross-products `T_min <= E_per_u*A <= T_max` reproduce the four
summary counts without using the new ledger's division/classification helper.

The final independent artifact review matched every one of the 236 rows to
the source-mass manifest, manually read the metadata and 107-node energy grids
of all 168 selected native outputs, and recomputed each exact boundary,
center partition and open/closed upper gap. Both materials and all four
summaries agreed. All 3571 evidence hashes were checked unchanged by that
review and again by the main review; final/replay bytes are identical.

### What the OST-profile findings mean (scope clarification)

The current five strict OST-profile findings are **not five proven errors in
the OST text**. They concern four additional RADAR data files:

1. Proton/Al range: authoritative upstream source/version remains unconfirmed.
2. Proton/Si LET: upstream provenance and the meaning/completeness of the
   legacy stopping columns remain unresolved; collisional/radiative stopping
   is not evidence of an elastic nuclear-stopping column.
3. HZE/Al range: the legacy 15-ion table has unconfirmed upstream provenance
   and incomplete positive-source species coverage.
4. HZE/Si LET: the damaged high-energy data are quarantined. A separate
   discontinuity diagnostic on that same file is the fifth finding.

These data-quality findings are separate from mathematical inconsistencies
in a printed expression and from an unresolved implementation convention.
The supplied OST PDF was visually rechecked on PDF pages 122-124, printed
pages 116-118, in this continuation. File SHA256 remains
`232c84bf7fb6c2d3aec11c44827bcf01a6b6964366a8d0059ff36cd5b143a9e1`.
The E.4 above-threshold definition and printed E.6 again show the missing
minus sign; E.11-E.13 again show the missing `2E` factor for differentiation
of `b1*exp(-b2*E^2)`. Their existing implementation treatment is documented
in `ost_appendix_e_formula_audit.md`; no formula was changed in this
continuation. This is a mathematical consistency finding, not a newly found
official errata sheet.

The agreed B.6 gamma2 multiplier 1.68 remains unchanged. The printed NEK He
6.6e-2 remains selected; its incorrectness is not established and 6.6e-3 is
not substituted. Fractional elemental A is not itself an OST defect. The
internal unresolved mass-policy flag must not be interpreted as requiring
isotope-resolved source spectra, experimental validation of every isotope,
or a 3D geometry. Within the agreed COSRAD-like scope, the required next
decision is a documented and checked source-A/transport-parameter convention
and applicable energy coverage. An isotope-distribution model is not added
merely to clear a diagnostic flag. The 1% acceptance criterion is numerical
convergence, not a promise of 1% physical accuracy.

No production tables, source grids, formulas, coefficients or release flags
were changed. The next computational work is to establish an evidence-backed
coverage strategy for the explicitly identified gaps, then validate the
supported spectrum transformations. Blanket extrapolation, silent source
truncation or zero filling is not an admissible substitute.

## Bounded native energy extension for the source roster (2026-11-01)

This continuation extends actual native stopping outputs without changing
source parameters, production tables, spectral grids, or release policy.
It does not infer high-energy data by extrapolation or rescale a second model
to agree with SRIM.

### Bounds and primary implementation evidence

The local documentation was rechecked: `SRIMHelp/SRTrouble.rtf` describes a
10 GeV/u table-program limit (SHA256
`cb3bf6c46e1ab7f00314ab90c9f3413d5871240e80ae915fa5032ae783910d19`).
The caution in `SRIMHelp/help2.rtf` above 1 GeV/u is explicitly about TRIM,
not a proven SRModule accuracy interval (SHA256
`1ff12022ac7485fa47caeb2f51d29535026551bdd4a94459175b678582cb8254`).
Being below that caution is not an accuracy guarantee.

Fresh primary implementation evidence was retrieved on 2026-11-01 from the
[SR-NIEL developers' residual-fluence calculator description](https://www.sr-niel.asi.it/index.php/sr-niel-web-calculators/web-calculator-for-determine-the-residual-spectral-fluence-for-a-particles-traversing-a-shielding-material/residual-spectral-fluence-for-particles-taversing-a-shielding-material).
They distinguish their SRModule upper energy of **5 GeV/amu** from the
**10 GeV/amu** available in the SRIM-2013 package. This is consistent with the
retained H/He execution observations, but does not identify the exact limit of
our pinned binary or explain its timeout mechanism. No new failing H/He
boundary search or undocumented special energy-list mode was attempted.

The fixed campaign chooses the largest normal-grid node no greater than
`1000 * literal_source_A` MeV total, and only when that node exceeds the
previous 12000 MeV endpoint. This **diagnostic experiment ceiling** is neither
an OST requirement nor a claim of physical validity. Normal-mode energy
spacing is retained, so arbitrary per-u endpoints are not fabricated.
All inputs remain below an explicit signed-32-bit-magnitude engineering
guard in keV; the guard is not a claim about SRModule internal variable types.

The extension roster has **79 species / 158 ion-material cases**. The ten
He/Li/Be/B/C material tables are left at their previous support, never
shortened to the new ceiling. Original lower endpoints, fractional source A,
separate SEP A/Q, targets and densities are unchanged.

| Pilot ion | New total-energy endpoint, MeV | Endpoint/source A, MeV/u | Native nodes |
| --- | --- | --- | --- |
| F19 | 18000 | 947.368421 | 113 |
| Ne20.2 | 20000 | 990.099010 | 114 |
| Fe55.8 | 55000 | 985.663082 | 125 |
| U238 | 225000 | 945.378151 | 141 |

### Execution and retained data

`scripts/run_srim_source_extension.py` uses the already pinned hidden,
bounded native launcher and isolated copied dependencies. Pilot mode runs
the four ions above in Al and Si twice. Full mode requires a complete pinned
pilot and unchanged implementation/source dependencies, reuses its eight
records, and executes only the remaining 150 cases twice. Every output must
match the complete expected native grid and all **107 old prefix rows**, with
all six numerical columns and original printed tokens. Metadata and footer
conversion factors must also match the baseline; the calculation date need
not match. A missing row, changed prefix, failed attempt or mismatching repeat
cannot supply a complete extension.

- Pilot report: `outputs/srim_source_extension_pilot_20261101/report.json`,
  SHA256 `b68c477900c19e3bcc32b64a80b6ebc534d3442faa359e6d6a25e64ab0492d70`.
  Eight complete cases / 16 native attempts.
- Full report: `outputs/srim_source_extension_full_20261101/report.json`,
  SHA256 `65dfd1a31ddc3e522da08b174dd8248d0be2c4ce35b78ca9ffece38401031802`.
  158 complete cases, eight reused pilot records / 300 new native attempts.

There are **316 actual extension attempts including the pilot**, not 332.
All complete repeat pairs are semantically and byte-identical. No native
failure, invalid output or repeat mismatch occurred. The 158 unique outputs
contain 20522 rows: 16906 old-prefix rows and **3616 new rows**. The complete
selected roster has 168 tables and 21592 rows after retaining the ten light-ion
tables. The reports record source input and runtime copies without modifying
the original software directory or distributing its binaries as application
assets.

Before full execution, independent raw review checked all 16 pilot outputs,
1972 rows / 11832 numerical cells, including 1712 baseline-prefix rows.
After full execution, independent raw review decoded all 316 outputs without
using the working parser or runner: **41044 rows / 246264 numerical cells**,
109-141 nodes per table, and 33812 exact prefix-row matches. All 158 repeat
pairs agreed, as did masses, materials, units, commands and inputs. Separate
retained-ledger review independently reconstructed the case roster/endpoints,
confirmed exact pilot reuse, and checked 45 protected files, 1106 case/repeat
identities, 316 command/result pairs and 1896 copied normal/runtime files.
These are execution and reproducibility checks, not physical qualification.

### Replayed selection and coverage improvement

Read-only summary chooses a **whole successful wider table** for each extended
case, never a stitched segment. A failed extension would keep the whole
baseline observation, while retaining the failure and unresolved upper gap.
It does not turn that failure into a zero particle contribution.

The final and replay summaries are byte-identical:
`outputs/srim_source_extension_full_summary_20261101.json` and
`outputs/srim_source_extension_full_summary_replay_20261101.json`, SHA256
`658b48c80abfe752a0057cd644dfbfcd8af25d0905025d67f8c71943e6065ce5`.
All **6892 evidence-file identities** were rechecked unchanged. The pilot-only
summary is retained separately at
`outputs/srim_source_extension_pilot_summary_20261101.json`, SHA256
`b0f8a2677312189317a20a367434a46b22ff93f5197c73e449fe4fb9648e9963`.

| Source, per material | Before: supported center/species entries | After | Gain | Still above support |
| --- | --- | --- | --- | --- |
| GCR | 2997 / 6804 | 4053 / 6804 | 1056 | 2751 |
| SEP | 2515 / 3960 | 2893 / 3960 | 378 | 1067 |

These values match the independently computed pre-execution predictions.
They describe sampled grid positions, **not fractions of flux, dose or SEE**.
All positive species still have unsupported upper intervals; the finite GCR
closure and SEP evaluated-domain distinction remain explicit. Low residual
energies and the absolute range origin are not resolved by an upper extension.
The existing derived-range implementation and the prior candidate files were
not changed or silently rebuilt; this block supplies new native observations
and their coverage audit, not a new production transport adapter.

Read-only reproduction, with an exclusive new output path:

```powershell
.venv/Scripts/python.exe scripts/run_srim_source_extension.py --mode replay --report outputs/srim_source_extension_full_20261101/report.json --report-sha256 65dfd1a31ddc3e522da08b174dd8248d0be2c4ce35b78ca9ffece38401031802 --output outputs/srim_source_extension_new_replay.json
```

### Software checks and remaining scope

Frozen implementation hashes:

- `validation/srim_source_extension.py`:
  `5d5eaee0ccfd61c3f7f8ffdbeb6e418da087fe64a3a293133d75873bf53cbb5d`;
- `scripts/run_srim_source_extension.py`:
  `34f5120aac44e02c42048a2dc6c8f3393e8bf8f9492008e31dc391ce0fafc92f`.

**559 focused tests passed in 92.77 seconds**, including all 131 new tests
(51 catalog and 80 synthetic campaign tests); counts are not additive.
JUnit: `outputs/pytest_srim_extension_final_20261101.xml`, SHA256
`f7845e1f46fe25f03f26c93df2655bf462ec94e05cd3b7caea772542626a2428`.
Ruff and strict mypy pass all four new files. This is a focused regression,
not a new full-suite or six-scenario spectrum-acceptance run.

The high-energy source tails exceed both this bounded campaign and the
documented SRModule range. Closing them requires an explicitly justified
high-energy treatment and overlap checks, not merely increasing SR.IN's
maximum indefinitely. No such model is selected or normalized by this
continuation. All physical/normative qualification flags remain false;
Stage 4 remains INCOMPLETE. Production manifest `.2` and its five strict OST
data findings are unchanged.

## Exact-node SRIM / CATIMA stopping overlap (2026-11-05)

This continuation measures the disagreement between the retained SRIM outputs
and the existing diagnostic CATIMA configuration. It does not select a
high-energy replacement, join tables, fit a normalization, alter OST source A,
or qualify either calculator as experimental truth.

### Method and scope

`scripts/compare_srim_source_overlap.py` authenticates the original baseline
and extension campaigns through their frozen replay implementations, including
the copied runtime/input files. It reconstructs **158 complete wider tables +
10 unchanged light-ion baseline tables**, not spliced row segments. The pinned
full selection summary is checked against this reconstruction. The replay
inventory has 6893 file identities, including that summary.

`scripts/generate_srim_overlap_controls.py` uses the already retained isolated
CPython-3.12 CATIMA wheel. It samples each requested energy directly rather
than interpolating the previously exported candidate grid. CATIMA's own
internal interpolation and LS mass-correction extrapolation are still present;
"no external energy-grid interpolation" must not be read as their removal.

The checked transformations are:

- `e = E_total / literal_source_A`, using exact rational arithmetic before
  conversion to the backend's binary64 input;
- `S_SRIM[MeV cm2/g] = 1000 * S_SRIM[MeV cm2/mg]`, with **no additional A factor**;
- `S_CATIMA,e = S_CATIMA,nonradiative - S_CATIMA,nuclear`;
- signed residual `r = S_CATIMA / S_SRIM - 1`. SRIM is the comparison
  denominator, not an assertion of physical truth.

Electronic, nuclear and nonradiative-total stopping are kept separate.
Radiative stopping is computed as the full-minus-nonradiative difference and
is zero at every sampled node, as expected below the retained backend's
50000-MeV/u switch. At 12310 nodes at or below 10 MeV/u, the separate exposed
electronic function agrees with nuclear subtraction to at most
`1.6653345369377348e-15` relative discrepancy. This is an internal arithmetic
check, not independent physical validation.

The selected SRIM tables contain **21592 rows**. Of these, **21090** are inside
the fixed overlap 0.01..3000 MeV/source-A. The remaining **502** are explicitly
listed below 0.01; none are above 3000. Those exclusions concern this comparison
only, not truncation of source spectra. Original raw table SHA256, row index,
printed energy/stopping tokens and literal mass are retained. Projected ranges
and CSDA ranges are deliberately not compared or calculated here.

Default materials are not fitted: CATIMA Al uses A=26.9815 and density 2.699,
whereas the authenticated SRIM input uses 26.981539 and 2.70. Si A=28.0855
agrees, but densities are respectively 2.33 and 2.3290. CATIMA's internal
ionization potentials remain 166 eV for Al and 173 eV for Si, with no override.
The projectile charge input is Q=Z with the fixed Pierce-Blann configuration,
not the SEP geomagnetic A/Q parameter.

### Physics provenance, not a new OST prescription

Fresh review of the retained `catima-1.91/src/catima.cpp:29-48` and
`src/calculations.cpp:75-124` confirms low-energy SRIM85 at e<=10, a linear
blend for 10<e<30, and the corrected Bethe/Lindhard-Sorensen treatment at
e>=30 MeV/u. The LS correction uses its own energy table and mass adjustment
(`calculations.cpp:580-594`). This implementation must not be described as
an exact copy of ATIMA 1.41 or as an independent experimental reference.

The [GSI ATIMA method description](https://web-docs.gsi.de/~weick/atima/),
retrieved on 2026-11-05, corroborates the low-energy Ziegler / high-energy
Lindhard-Sorensen construction and explicitly notes substantial differences
from simple Bethe treatment for high-Z projectiles. Its ATIMA energy-range
claim is not transferred to this particular CATIMA wheel as an accuracy
guarantee. The [SR-NIEL developers' description](https://www.sr-niel.asi.it/index.php/sr-niel-web-calculators/electronic-stopping-power-and-ionizing-dose-converters-up-to-high-energy-for-proton-or-ion-spectral-fluence-in-elements-and-compounds)
provides a separate example of a high-energy treatment with overlap checks;
its reported agreement percentages are not RADAR acceptance tolerances.

### Results

The following values describe **electronic stopping**, with signed differences
in percent `100*r`. All bands are numerical summary bins, not exclusive backend
branch labels: exactly 10 MeV/u still uses the low-energy branch.

| E/source A, MeV/u | Nodes per material | Al median absolute difference | Si median absolute difference | Al largest absolute, signed | Si largest absolute, signed |
| --- | ---: | ---: | ---: | ---: | ---: |
| [0.01,1) | 3966 | 7.1034% | 8.9576% | +82.0607% | +129.2835% |
| [1,10) | 2184 | 3.9717% | 2.9420% | +15.0209% | +17.1658% |
| [10,30) | 1057 | 5.9998% | 3.4331% | -18.9241% | -18.6012% |
| [30,100) | 1127 | 2.4475% | 2.4872% | -17.6741% | -17.5700% |
| [100,1000) | 2184 | 3.8130% | 3.7479% | +16.0042% | +15.8762% |
| [1000,3000] | 27 | 0.3967% | 0.2778% | +5.0329% | +4.8833% |

These are node-count descriptions, not source-abundance-, flux-, dose- or
SEE-weighted statistics. The last band contains only light ions (Z=2..8),
including nodes exactly at 1000; it does not establish heavy-ion upper-tail
accuracy. The below-1-MeV/u maxima occur for Xe, at different energies in the
two materials. They are a separate low-energy model disagreement, not evidence
that the high-energy extension caused a low-energy defect.

At the retained upper nodes, the Si electronic differences are +5.6278% for
Fe55.8 at 985.663 MeV/u, +15.8762% for Th232 at 969.828, and +13.7448% for
U238 at 945.378. The corresponding Al differences are +5.7406%, +16.0042%
and +13.9514%. Nuclear stopping contributes only approximately 0.011..0.015%
of nonradiative stopping at these six nodes, and the nonradiative residuals
are nearly identical. Thus these particular residuals are not explained by
confusing electronic stopping with nonradiative total stopping.

No one model has been established as correct by this comparison. Agreement
with a second model is neither a 1% numerical-convergence check nor proof of
physical accuracy. In particular, no energy-independent scale factor or
automatically selected joining energy is justified by these results.

### Retained evidence and verification

| Artifact in outputs/ | SHA256 |
| --- | --- |
| srim_catima_overlap_requests_20261105.json | d4bdec9546a825a05090d16f82574a61144d7d038af6ce7972937834c6045d52 |
| srim_catima_overlap_native_20261105.json | f839ef25be45dd8a7dfc0a51806a78437668be0a4929e139c5cae4ce0ee17441 |
| srim_catima_overlap_native_repeat_20261105.json | f839ef25be45dd8a7dfc0a51806a78437668be0a4929e139c5cae4ce0ee17441 |
| srim_catima_overlap_comparison_20261105.json | 34e36a5b9fec1b4bae07146c1661b392c14260e16b8a8b2283dfc1bcf89d654d |
| srim_catima_overlap_comparison_replay_20261105.json | 34e36a5b9fec1b4bae07146c1661b392c14260e16b8a8b2283dfc1bcf89d654d |

Both complete native executions are byte-identical. Both read-only comparison
reports are also byte-identical. Independent review, without importing the
new comparator or the SRIM parser, manually decoded all 168 selected native
text files, reconstructed all 21592 rows, verified request/exclusion identities,
and recomputed every matched stopping conversion/residual and all **182 summary
groups**, with exact agreement. This is a separate arithmetic/evidence review,
not a third physical model.

Frozen implementation SHA256:

- `scripts/compare_srim_source_overlap.py`:
  `0d39b1768fc721317968923ee87d1453571065787ee56c9742f28814b65a3f67`;
- `scripts/generate_srim_overlap_controls.py`:
  `d0293df5d5b2a82857ac36d873c5832b95a9e4a8f5f443d891c902ef757b6ce5`.

**944 focused regression tests passed in 80.69 seconds**, including **215 new
tests** (82 comparison/selection tests and 133 worker/replay tests); these
counts are not additive. Ruff and strict mypy pass all four new code/test
files. JUnit: `outputs/pytest_srim_catima_overlap_final_20261105.xml`, SHA256
`ba10991874dbfa9cc50a0e3cc83ae16eb4fa936f0a001bf783ad806a4c7507d9`.
This is not a full-suite or six-scenario spectral-acceptance run.

Example read-only replay (use a previously nonexistent destination):

```powershell
.venv/Scripts/python.exe scripts/compare_srim_source_overlap.py --mode compare --requests outputs/srim_catima_overlap_requests_20261105.json --requests-sha256 d4bdec9546a825a05090d16f82574a61144d7d038af6ce7972937834c6045d52 --native outputs/srim_catima_overlap_native_20261105.json --native-sha256 f839ef25be45dd8a7dfc0a51806a78437668be0a4929e139c5cae4ce0ee17441 --output outputs/srim_catima_overlap_new_replay.json
```

### Next bounded step and unchanged release boundary

Before selecting any high-energy continuation, attribute the disagreement at
fixed existing Fe/Th/U nodes near 30, 100, 300 and 950 MeV/u in both materials.
Keep the current full configuration as the baseline; inspect LS, Barkas and
shell contributions separately, and effective-charge influence separately.
Such switched-off variants are diagnostics, not preferred production models;
the variant closest to SRIM is not automatically the physically correct one.
Then select applicable independent experimental/theoretical controls. Neither
the source tail above native coverage nor the radiative switch at 50000 MeV/u
is validated by the present overlap test.

Production tables, source spectra/grids, coefficients, formulas, runtime
dependencies and release policy remain unchanged. A fresh strict OST audit
still reports **15 resources, five data findings, release_ready=false**, with
manifest SHA256 `b5f1b04cd3da03ac458bfbbb85a3716622debfe234aabbd933f9d3ea2bd87592`.
Those findings concern RADAR's unresolved tables, not five newly established
mistakes in OST. Stage 4 remains **INCOMPLETE**; dose and SEE stages are not
started by this continuation.

## Opt-in SRIM source-to-shield-to-LET integration (2026-11-05)

### Scope correction and implemented result

Following the user's request to return to practical Stage 4 work, the detailed
LS/Barkas/shell attribution proposed above is **not a mandatory prerequisite**
for continuing integration. The completed comparison remains diagnostic
evidence; it does not select a preferred model or establish mistakes in OST.

The restored native SRIM data now feed an opt-in computational path, rather
than only comparison scripts:

- `radar.shielding.srim_dataset.load_srim_hze_dataset`: authenticated, bounded,
  text-only ZIP loading; exact ion/material/source-mass pairing and member hashes;
- `radar.shielding.srim_hze.transport_srim_primary`: conservative primary-ion
  transport through Al using the observed stopping data;
- `radar.shielding.primary_let.convert_primary_electronic_let`: conversion of
  the original source measure to Si electronic LET, including all branches of
  a nonmonotone LET curve;
- `radar.pipelines.srim_hze.calculate_srim_hze_products`: ordinary typed
  GCR/SEP products in, per-ion energy and LET products for one selected mode out;
- `scripts/build_srim_hze_bundle.py` and `scripts/run_srim_hze_products.py`:
  reproducible data packaging and an executable external-source integration run.

The shared LET operator preserves the existing candidate diagnostic wrapper.
Four existing validation runners now also fingerprint its implementation file.
This does not change the meaning of previously frozen reports or silently
update their historical implementation fingerprints.

### Physical and numerical boundaries

Al transport integrates electronic plus elastic-nuclear stopping analytically
under piecewise log-log interpolation. It uses a path coordinate anchored at
the first observed energy, not SRIM's projected range as a substitute for CSDA
range. The additive anchor cancels when both incident and exit energies are
inside the supported interval. Exits below that interval remain **unresolved**,
not automatically stopped or represented by an extrapolated curve.

Si LET uses electronic stopping only. Every output-bin contribution is obtained
from preimages in the original source measure; it is not inferred by subdividing
an intermediate energy histogram. Literal source masses are preserved, including
the existing floating-point serialization of their trace. There is no additional
mission-time, mass-number, density or solid-angle multiplier.

Missing ions/tables produce explicit incomplete cases, never an implicit legacy
fallback or fabricated zero. Zero Al thickness preserves the full input energy
spectrum; missing Si coverage is still reported separately for LET. All new
products remain `INCOMPLETE`, with normative export disabled. No nuclear reaction
attenuation or fragmentation model is added. No aggregate HZE LET product or GUI
connection is introduced in this step; the batch returns individual-ion products.

### Retained data and actual calculations

The data bundle contains **84 Al/Si pairs, 168 native text tables and 21592 rows**:
158 whole extended tables plus ten whole retained baseline tables. It contains
no executable. Native files were reused; SRIM was not launched in this step.

| Artifact in outputs/ | SHA256 |
| --- | --- |
| srim_hze_observed_20261105.zip | f2b523dfafbbe8bfbcf70b27a186f05ed009a19918b14042f97b39e37e7f0bb2 |
| srim_hze_products_full_20261105.json | 06d9345d09953b11796be624f7d05d785b0a3e2a0e9549020c1e866997cd45f7 |
| srim_hze_products_dense_20261105.json | 1488b658cee67ff268cb544ae3f5a46206399ca84ac9bbda2abb86e0885de0a3 |

Each run uses actual OST external source models and the OST solar timeline:
launch 2026, lifetime five years, mean cycle, SEP exceedance probability 0.1.
There are **216 positive source/ion/mode inputs and 1296 shield cases**:
84 GCR and 24 SEP ions, both modes, shields 0; 0.01; 0.1; 1; 3; 10 g/cm2.
Fourteen zero-sampled GCR source products are explicitly recorded separately.
Source generation, mode units, all energy/LET products and unsupported counts
are retained. These are **external-source integration runs, not LEO/GEO/HEO
orbital acceptance calculations**. Dose and SEE were not calculated.

The refined run inserts geometric midpoints into each source-energy interval:
GCR 81 to 161 nodes, SEP 165 to 329. Stopping data and interpolation are fixed.
Below, relative changes use the baseline as denominator, `abs(dense/base - 1)`:

| Per-ion integral | Maximum relative change |
| --- | ---: |
| Declared source integral | 0.287984% |
| Supported exit / represented energy / mapped LET | 0.370445% |
| Outside Al energy support | 0.862369% |
| Unresolved below-anchor exit | 1.651753% |

The largest change in a source/mode/shield aggregate mapped LET integral is
0.291216%. However, **78 below-anchor ledger comparisons exceed 1%**; the worst
is GCR peak Ga at 0.1 g/cm2. Therefore this does not establish the Stage 4
convergence criterion for all quantities, still less for the orbital matrix.
It is also not a refinement or validation of the physical stopping tables.

On the baseline source grid, the upper tail outside observed Al support is
19.801406% of GCR peak and 23.561437% of GCR cumulative source counts. On the
refined grid these become 19.702156% and 23.445620%. The corresponding SEP
fractions are approximately 4e-8 to 6e-8. The lower outside-Al contribution is
zero on these source grids. These fractions concern **particle counts over the
declared finite source interval**, not bounds on LET, dose or SEE errors. At
zero shield the energy spectrum is intact and the analogous LET limitation is
accounted for as outside-Si support.

Example run (choose a previously nonexistent output path; add
`--source-grid-refinement 2` for the refined source grid):

```powershell
.venv/Scripts/python.exe scripts/run_srim_hze_products.py outputs/srim_hze_observed_20261105.zip --bundle-sha256 f2b523dfafbbe8bfbcf70b27a186f05ed009a19918b14042f97b39e37e7f0bb2 --output outputs/srim_hze_products_new_run.json
```

### Verification and remaining work

**1039 unique focused regression tests passed in 12.30 seconds**, including
255 new tests for the transport, LET, dataset, pipeline and runner. Ruff and
strict mypy pass all 16 new/modified code and test files checked. JUnit:
`outputs/pytest_srim_integrated_close_20261105.xml`, SHA256
`205308f164a15ea363c39e1d212c670887ec15da93ca16c7b12fd7f53d458faa`.
This is not a full-suite run.

A separate arithmetic review reconstructed both reports from the original
source spectra and 168 native SRIM files without importing the new operators:
373032 numerical checks per report, all bins, ledgers and 24 aggregate groups.
Maximum residuals scaled by source integral were 3.10e-16 and 2.62e-16. This
checks implementation arithmetic, not independent physical validity of SRIM,
the OST source model, or orbital averaging.

The next bounded tasks are to resolve the low-energy range origin / stopping
classification, select and substantiate upper-energy coverage where needed,
and then connect qualified data to the normal pipeline and repeat the orbital
acceptance matrix. Unsupported contributions must remain explicit until then.
Detailed inter-model correction attribution is not an additional release gate.

Default profiles and normative tables were not replaced. The current strict
OST audit still has **15 resources, five findings, release_ready=false**;
manifest SHA256 remains
`b5f1b04cd3da03ac458bfbbb85a3716622debfe234aabbd933f9d3ea2bd87592`.
The new path is usable through its API/runner but is not yet the default
calculation backend. Stage 4 remains incomplete.

## Lower-energy SRIM pilot and parser correction (2026-11-05)

The next practical step extends whole native tables downward, without choosing
an extrapolation to rest. For He4, Fe55.8 and U238 in Al and Si,
`scripts/run_srim_low_energy.py` requests **10 eV total ion energy**, keeping
each previous upper endpoint, material, source mass and normal-mode coefficients.
This is a bounded engineering pilot, not a minimum prescribed by OST. The local
`HELP- SR Module.rtf` documents the input energy unit as keV; the input is 0.01 keV.
No installer, system change, alternate underscore data files or original SR.IN
modification was used. All executions use new isolated directories.

### A concrete reader defect, not an OST formula correction

The first native process succeeded but our reader rejected its output because
it required strictly increasing projected ranges. SRIM prints several adjacent
low-energy ranges as the same integer number of Angstroms (e.g. 4 A). The reader
now requires strictly increasing energies and **nondecreasing printed projected
ranges**. Negative/zero ranges and decreasing ranges remain rejected. Tokens,
raw hashes and stopping values are unchanged; projected ranges are still not
used as CSDA path ranges. A regression explicitly covers printed plateaus.

The native energy grid also contains representation effects: the requested
10 eV is printed as **9.99999 eV**, retained as 0.00000999999 MeV, not replaced
with an idealized node. Each low-energy node must correspond one-to-one to its
nominal grid node within one printed decimal quantum **and** 1e-5 relative;
the entire shared suffix from 1 MeV is exact. Each table has 50 such shifted
low-energy nodes, maximum relative departure 9.090909e-6 and maximum absolute
departure 1e-9 MeV. This check tolerates observed printing resolution, not a
change of the grid, source model or physics coefficients.

The rejected first attempt is preserved in `outputs/srim_lower_pilot_20261105`.
The corrected pilot is in `outputs/srim_lower_pilot_verified_20261105`:
six cases, two repetitions each, all valid and raw-byte-identical on repetition.
Each whole table adds **130 lower nodes**. All original shared rows, including
their projected ranges and straggling, are exactly preserved. Table sizes are
237 rows for He, 255 for Fe and 271 for U in each material (1526 rows total).

### Calculated impact and its limits

The new bundle was run through the existing typed energy/LET pipeline for
both modes and shields 0; 0.01; 0.1; 1; 3; 10 g/cm2: **60 cases**, comprising
GCR He/Fe/U and SEP He/Fe. Parent products were recomputed with the same current
implementation so comparison does not silently mix old and new code fingerprints.
`scripts/compare_srim_lower_support.py` checks matching source inputs, exact
shared stopping, range-coordinate translation, preserved old output-bin counts,
conservation of the newly supported contribution, time/unit semantics and
zero-shield identity. Nonfinite or duplicate JSON values are rejected.

The newly observed Al path increment between the new lower endpoint and 1 MeV
total is 9.14576817e-4 g/cm2 for He, 2.23609631e-4 for Fe and 5.49387450e-5
for U. It shifts the anchored coordinate by a constant on the common domain;
it does **not** change the range differences between previously supported energies.

The largest added contribution in this pilot occurs for cumulative SEP He at
0.01 g/cm2: **3.035648% of its finite declared source count**, or **5.141359% of
its previous mapped LET integral**. The increase is additional low-energy
support, not a fitted normalization of existing contributions. These are
particle-count integrals, not dose or SEE error estimates. This pilot is not
the full-ion or orbital acceptance matrix and is not a grid-refinement test.

A positive lower energy still leaves the integral from rest to that energy
unobserved. Consequently the remaining low-energy ledger means **stopped OR
exiting below the observed endpoint**, not proved stopped. No finite positive
table endpoint alone establishes the absolute CSDA range origin. That remaining
boundary must be addressed explicitly before normative qualification; lowering
the cutoff does not silently remove the release restriction.

### Evidence and verification

| Artifact in outputs/ | SHA256 |
| --- | --- |
| srim_lower_pilot_verified_20261105/report.json | b280096db02201fee59bc0dc9e7ba68daaa30cbd823158790fd80b45fddc9f2c |
| srim_lower_pilot_verified_20261105/observed_pilot.zip | 8ec49fef7e5c36fb7b2ea2acfad70fa6acab100588d6d1bc3a5ebb216347b078 |
| srim_parent_products_lowercheck_20261105.json | 111f03f4282d43b891fb1580763e480a22191deea6d2efe8143535f05509bde9 |
| srim_lower_products_pilot_20261105.json | 9f20899f2658c030144b768e60b525155aa092211d979e88c639c78c05293378 |
| srim_lower_impact_20261105.json | 23efd04904b8b00e840dba62d5d03d34816ed560a61edb69a2e97584598d01be |

**1176 focused tests passed in 16.17 seconds**, including 137 new lower-pilot,
comparison and analytic support-extension tests; counts are not additive.
Ruff and strict mypy pass the seven new/modified implementation/test files
(`MYPYPATH=src;tests`, explicit package bases for the combined mypy invocation).
JUnit `outputs/pytest_srim_low_all_20261105.xml` SHA256:
`b5f5c0ec533f24099694c77c52be33d23dfec514e7cb617fc666a080fe2c6a7a`.
This is not a full-suite run. The parser change leaves previously parsed
semantic payloads unchanged, but its implementation hash has changed; historical
code-pinned replay reports remain historical and are not rewritten as current.

Independent read-only review manually parsed the six native tables and
reconstructed all 60 new cases without importing RADAR operators: 16820
numerical checks, including 16460 energy/LET-bin checks, maximum residual
scaled by source integral 1.0101e-15. All twelve native repeat hashes, shared
rows, range offsets and unsupported-count balances agreed. This verifies
implementation arithmetic, not independent physical accuracy or a range origin.

Next: expand the reviewed lower-table procedure to the full ion roster, retain
an explicit below-cutoff policy, then resolve the independent upper-energy
coverage and repeat orbital acceptance. Normative tables/default profiles are
unchanged; strict OST audit remains **15 resources, five findings,
release_ready=false**. No dose/SEE implementation was started.

## Full lower-energy SRIM coverage (2026-11-05)

The reviewed lower-energy procedure is now applied to **all 84 positive OST
HZE source species in Al and Si**. `scripts/run_srim_low_energy_full.py` checks
the fixed source catalog and baseline bundle, authenticates both repetitions
of each of the six pilot cases, reuses those whole tables, and executes the
remaining **162 cases twice** in new isolated directories. A failure stops
the campaign; a full bundle is created only after all 168 cases succeed.
The pilot implementation and original campaign artifacts remain unchanged.

The complete campaign succeeded: 168 tables, 336 retained repetitions including
the twelve pilot executions, **43432 rows = 21592 unchanged old rows + 21840
new rows**. Each table adds 130 lower-energy nodes. Every common row is exactly
preserved, not just its stopping columns; every repeated output is byte-identical.
The actual first printed energy is 9.99999 eV total in all tables. Independent
read-only review confirmed the full inventory, all shared rows and repetitions,
and all 304 protected current-file hashes. No source model, coefficients, upper
endpoints, materials, original SRIM files or normative tables were changed.

The full text-only data bundle is
`outputs/srim_lower_full_20261105/observed_full.zip`, SHA256
`20c7e5686e61d14200dff807627009e0859f560c2adfdc34e37bc0f0d3314607`.
Native report `outputs/srim_lower_full_20261105/report.json`, SHA256
`9057ec24f594e64689f4c42dc23d35a6337383b08e2c530c3716355eab727c32`.
It retains the raw input/output/command fingerprints for each execution.

### Explicit below-cutoff interpretation and complete-inventory gate

The comparison report now includes, for every case, the classification
`stopped_or_exit_below_observed_energy`, the actual Al total-energy endpoint
`Tmin`, and the reported low-residual particle integral `U`. In the current
primary CSDA model (no secondary production or acceleration), the kinetic
energy of this subset obeys `K_low <= Tmin * U`. The product is retained as a
decimal value evaluated from the reported binary64 count and the native decimal
endpoint: PEAK units MeV cm^-2 s^-1, cumulative units MeV cm^-2. There is no
additional A, mission-time, density or angular factor. The angular normalization
remains that of the source; this is not energy crossing an arbitrary plane.

This bound covers **only** the unresolved below-anchor subset, not particles
outside the Al source-energy domain, absent ions or other unknown source tails.
It is not an error bound for LET, dose or SEE and does not establish stopped
particle classification, physical uncertainty, an absolute range origin or
permission for normative export. Energy bins entirely above Tmin/A receive no
contribution from this subset; no equivalent conclusion for LET follows solely
from the small residual energy. No extrapolation below the observed endpoint
has been introduced.

`scripts/compare_srim_lower_support.py --require-full-inventory` additionally
rejects a partial pilot: it requires identical 84-pair inventories, all 216
source/ion/mode inputs and the complete **1296-case** source/mode/shield matrix.
The generic pilot-compatible subset comparison remains available explicitly
without this flag. Both comparison reports remain non-normative.

### Code verification

**1240 focused tests passed in 21.40 seconds**, including 42 new full-campaign
tests and 22 additional complete-inventory/low-residual tests. Ruff and strict
mypy pass the four new/modified code and test files. JUnit:
`outputs/pytest_srim_lower_full_final_20261105.xml`, SHA256
`80ddefa13c835af41e54a056a4111bd16d0014bce01bd4667c186fcdbcb396ef`.
This is not a full-suite test run or completion of the orbital acceptance matrix.

### Full external-source calculations and refinement

The parent bundle, new full lower bundle, and new full bundle with doubled
source-grid density were calculated with identical current implementation and
normative input fingerprints. Each run has 216 positive source/ion/mode inputs
and 1296 cases; fourteen zero-sampled GCR source records remain explicit.
Launch is 2026, lifetime five years, mean cycle and SEP exceedance probability
0.1. These remain external-source transport calculations, **not** the orbital
LEO/GEO/HEO acceptance matrix.

| Artifact in outputs/ | SHA256 |
| --- | --- |
| srim_parent_products_full_lowercheck_20261105.json | 85e4d6a08cd9293fe49cf0b629fbdeec1f0b3eba8d8542ad236d62c865827ddb |
| srim_lower_products_full_20261105.json | be862d3a07a1b0c3e6321efebb6f10938de1bcdd9b10dacf28be3be7a82fec37 |
| srim_lower_products_dense_20261105.json | 4721300a73370a6d45119b2605d7cc1d4a403e233851ec13a90fbd21ca5987f9 |
| srim_lower_full_impact_20261105.json | b1de65596cf2fb5f66a8dba1f49aa7f3a7ded6f68c4b5c0587a1612e82dd53b7 |

The full-inventory comparison passes for all 84 pairs and all 1296 cases.
Old supported energy bins are preserved; new supported counts equal the
reduction in the below-anchor ledger, and no upper source-domain tail is
silently removed. The largest increase of a per-ion mapped LET integral is
**5.283979%**, cumulative SEP Ni at 0.01 g/cm2. Summing mapped LET counts over
all positive SEP ions for that mode/shield gives an increase of **5.107523%**.
The maximum added fraction of an individual finite source count is **3.035648%**,
cumulative SEP He at 0.01 g/cm2. These are particle-count integrals, not dose/SEE.

For source-grid refinement (GCR 81 to 161 nodes, SEP 165 to 329), using
`abs(dense/base - 1)`:

- maximum individual source-integral change: 0.287984%;
- maximum supported energy / mapped LET integral change: **0.370451%**;
- maximum source/mode/shield aggregate mapped LET change: **0.291183%**;
- maximum below-anchor integral change: **1.657262%**, GCR peak Ga, 0.1 g/cm2;
  76 such per-case ledger values exceed 1%.

Thus the supported output-integral refinement passes the 1% check in this
external-source run, but this does not close all Stage 4 convergence criteria.
The finite upper-energy gap is unchanged: 19.801406% of GCR peak and 23.561437%
of GCR cumulative source counts on the baseline grid. It is neither a bound
on output error nor an acceptable implicit zero tail.

The lower-energy native-table coverage subtask is complete for the declared
84-ion roster. The absolute range origin / below-cutoff release policy,
upper-energy treatment, normal-pipeline qualification and orbital acceptance
remain open. Default profiles and normative data remain unchanged; strict OST
audit still reports 15 resources, five findings and `release_ready=false`.

Independent arithmetic review, without importing RADAR operators, reconstructed
both new full reports from the raw native tables and original source measures:
370440 checks per report, including 355536 energy/LET-bin checks. Maximum
residuals scaled by source integral were 2.7984e-16 and 2.8054e-16. Mode units,
time semantics, all primary/LET ledgers and complete inventories agreed. This
is a separate implementation check, not physical qualification of SRIM or OST.

## Upper-energy source budget and bounded native pilot (2026-11-05)

The next step separates three questions: how much finite source measure is
outside native support, whether a wider whole table can actually be generated,
and whether the wider physical model is qualified. The first two are addressed
here; successful execution does not establish the third.

### Source-weighted coverage, without truncation or a dose/SEE claim

`scripts/audit_srim_upper_budget.py` authenticates the full lower SRIM bundle and
its external-product report, checks all 216 positive zero-shield source identities,
units, source mass traces and both modes, and integrates their piecewise-linear
densities. No extra time, density, mass-number or angular factor is introduced.
Prospective coverage uses the largest normal-grid node <=5000 times literal
source A, subject to the explicit 2147483647 keV engineering input guard.
It is a planning calculation, not a claim that all 168 wider tables exist.

| GCR cutoff / support scenario | PEAK fraction above it | CUMULATIVE fraction above it |
| --- | ---: | ---: |
| Current individual native upper endpoints | 19.801406% | 23.561437% |
| Common 3 GeV/source A | 16.468077% | 19.848628% |
| Common 5 GeV/source A | 8.652015% | 10.586748% |
| Prospective normal nodes <=5 GeV/source A | 8.667365% | 10.605228% |
| Common 10 GeV/source A | 3.190796% | 3.952675% |

The denominator is the declared finite source integral over 1..79432.823472
MeV/source A, not an integral to infinity or even to the separate 100000 MeV/u
reference boundary. SEP has its own finite interval 1..12589.254118 MeV/source A.
The prospective SEP tail fractions at the <=5 GeV/source-A normal nodes are
2.19805e-9 PEAK and 1.40502e-9 cumulative. These are **particle counts**, not LET,
dose or SEE error bounds. Unknown source tails beyond the declared intervals
are neither included nor set to zero.

He supplies 74.17% / 75.13% of the presently unsupported GCR count (peak/cumulative),
and approximately 89.20% of the prospective remaining count after the <=5 GeV/A
extension. Small counts of rarer heavy ions do not bound their SEE contribution.
Thus a wider native table is useful but cannot by itself close the high-energy
problem; even the prospective remaining count is far above 1%.

The final budget report is `outputs/srim_upper_source_budget_final_20261105.json`,
SHA256 `9228d31665d37a2297bd81d82aded598605fb4df05329299996088af7f6ac81e`.
Independent reconstruction from raw native metadata and zero-shield spectra,
without importing the audit algorithm, agrees for all 216 cases / 4536 values.

### Native pilot and physical boundary of its interpretation

The [SR-NIEL developers' current description](https://www.sr-niel.org/index.php/sr-niel-web-calculators/web-calculator-for-determine-the-residual-spectral-fluence-or-the-residual-energy-up-to-high-energy-for-protons-and-ions-traversing-a-shielding-material),
rechecked on 2026-11-05, distinguishes the 5 GeV/amu upper limit of their SRModule
use from the 10 GeV/amu package limit. This motivates a bounded experiment,
not an OST-prescribed cutoff or an accuracy guarantee for our pinned binary.
No extrapolation, adjustment to another model, or automatic boundary search
was used. The four pilot masses retain the source A, not rounded isotopes.

| Pilot | Total-energy upper node, MeV | Upper node / source A, MeV |
| --- | ---: | ---: |
| He4 | 20000 | 5000 |
| Ne20.2 | 100000 | 4950.495050 |
| Fe55.8 | 275000 | 4928.315412 |
| U238 | 1100000 | 4621.848739 |

`scripts/run_srim_upper_pilot.py` generated all four ions in both Al and Si,
twice, in isolated copies with the previous requested 10 eV lower endpoint.
All **8 cases / 16 executions passed**. The 2014 old rows are exactly preserved,
including original tokens, projected ranges and straggling; 118 upper rows
were added, for 2132 rows total. Every repeated raw output is byte-identical.
The observed lower energy remains 9.99999 eV total. All 194 protected current
files, including original source/runtime inputs and normative data, are unchanged.

| Artifact in outputs/ | SHA256 |
| --- | --- |
| srim_upper_pilot_20261105/report.json | 5955844fd96615eddbd251ddf1748d12e2a7614bd777c923e8307d46e9e1ce3c |
| srim_upper_pilot_20261105/observed_upper_pilot.zip | aef824e81874d71592b6f630aa30fe62779dae1fd7c0773f76fa0727702b90c6 |
| srim_upper_products_pilot_20261105.json | fd0c4c013800a4fc4c196bd5778fa689f244d0a95aadbf420e97414d75d46e2d |

The wider pilot bundle feeds the same primary energy / Si electronic-LET
pipeline: **84 cases**, GCR He/Ne/Fe/U and SEP He/Ne/Fe, both modes and all six
shields. Independent arithmetic review manually decoded the native files and
reconstructed all bins and ledgers without importing RADAR operators: 24150
checks including 23044 energy/LET bins; maximum residual scaled by source
integral 2.49e-16. The zero-shield energy identity, temporal/unit conventions,
lower residual and conservation of recovered upper contributions all agree.
This is not the full-ion or orbital acceptance matrix, physical model validation,
or grounds to remove `INCOMPLETE`.

### Comparison, regression checks and remaining work

`scripts/compare_srim_upper_support.py` authenticates the parent/pilot bundles
and product reports and requires exactly the declared 84 cases. It checks the
whole old table prefix, unchanged source identities and mode/unit conventions,
nondecreasing supported bins and the balance between recovered upper source
counts, supported output and the below-anchor residual. Zero-shield energy
remains identical; zero-shield LET may gain supported counts when the Si table
is extended. Remaining unsupported Al/Si contributions stay explicit.
The actual comparison passes:
`outputs/srim_upper_pilot_impact_20261105.json`, SHA256
`cbbdc9189e6f6e0af57677e242feac56974f83781babfc30f161e9791dfcefef`.

Repeating the source-budget audit on the doubled source grids gives prospective
normal-node <=5 GeV/source-A GCR tail fractions 0.08606932985 PEAK and
0.10531607137 CUMULATIVE. The largest relative change of the four source/mode
tail fractions is 0.697239%. This checks only the source-budget quadrature;
it is not a refinement test of the extended transport or the complete orbital
calculation. The dense audit is
`outputs/srim_upper_source_budget_dense_20261105.json`, SHA256
`512b4825339a7755c13fa8c3915a16d7ec66d025ffeab141543ed7e4c7b2eadc`.

The combined targeted regression run passes **1420 tests**, with no failures,
errors or skips (JUnit duration 24.759 s). Its record is
`outputs/pytest_srim_upper_final_20261105.xml`, SHA256
`1c36fc7f705570366e36270ef3aa0eb7652a892b566e1daa0dea103e81ad33ae`.
Ruff and mypy pass for all six new audit/pilot/comparison implementation and
test files. A final independent hash check finds all 194 protected files
unchanged. No default model, source coefficient, normative table or production
transport formula was replaced. The strict OST audit remains at 15 resources,
five findings and `release_ready=false`; those data-readiness findings are not
newly asserted errors in OST.

The next bounded implementation step is to apply the proven native upper-table
procedure to the full declared ion roster, retaining per-table failure gates
and exact parent-prefix checks, then rerun the complete spectral/refinement
matrix. This still cannot close the upper-energy domain: a separately justified
high-energy treatment is needed up to the declared source endpoints. Automatic
SRIM/CATIMA splicing, extrapolation or treating the remaining tail as zero is
not authorized by these results. Physical qualification, absolute range-origin
policy, the outstanding lower-residual refinement and orbital acceptance remain
separate open gates. Stage 4 and normative export remain open/disabled.

## Full-roster native upper extension (2026-11-05)

The bounded upper experiment now covers the entire fixed positive OST HZE
roster: **84 ions, 168 Al/Si tables**. This closes the execution/coverage
subtask for normal nodes <=5000 MeV per literal source A, not the full source
energy domain or the physical qualification of the stopping model.

### Controls and native results

`scripts/run_srim_upper_full.py` reauthenticates all eight reviewed pilot cases,
including both retained inputs, commands, raw outputs and source/runtime hashes.
It executes the remaining **160 cases twice** in isolated working directories.
An unsuccessful native execution, incomplete grid, changed parent row or
disagreeing repeat stops the campaign; there is no automatic boundary search.
The endpoint calculation is independent of ambient decimal precision. Source
masses, target definitions and the requested 10 eV total-energy lower endpoint
are retained. No program is installed and the original SRIM directory is not
modified.

All **168 cases passed**, with **336 matching repeated outputs** in total
(320 new executions, 16 authenticated pilot executions). The full bundle has
**46444 rows: 43432 exactly retained parent rows plus 3012 new upper rows**.
The observed lower endpoint remains 9.99999 eV total ion energy. Individual
upper normal nodes range from 4405.286344 to 5000 MeV/source A; the minimum is
Dy158.9 at 700000 MeV total, not an arbitrary change of its source mass.
A final hash check found all **342 protected files unchanged**.

| Artifact | SHA256 |
| --- | --- |
| scripts/run_srim_upper_full.py | c0ed6b58a41745b9282468d389c79d015f56c5dc05e0d12a8e7de15dae8d4f6f |
| outputs/srim_upper_full_20261105/report.json | f0e1f991bab45b0e51df20a43df1dba12fbdb8136a70877b4a10a3be451327ff |
| outputs/srim_upper_full_20261105/observed_upper_full.zip | d4754dce59446a44d8c96ffe9853feb4102a70ade2e3e839737eaff63289434c |

### Comparison and regression contract

`scripts/compare_srim_upper_support.py --require-full-inventory` now checks
the independently reconstructed fixed source roster, exact native masses,
84 GCR / 24 SEP identities, both modes and all six shields: 216 positive inputs
and **1296 cases** in each report. Merely having the right number of ions is
insufficient. Source grids, units, temporal semantics and implementation/input
fingerprints must agree with the parent report.

The conservative relation is `recovered_upper = supported_gain + below_anchor_gain`.
It does not assume that the below-anchor residual must
always remain unchanged for arbitrary shielding. Both reports must also satisfy
supported = represented + below-output-window + above-output-window. Zero-Al
energy remains identical, while an extended Si table can increase mapped LET
coverage even at zero Al. Unsupported tails are never silently set to zero.

The combined targeted regression run passes **1525 tests**, with no failures,
errors or skips (JUnit duration 43.797 s). This includes 59 new native-campaign
tests and 46 new full-inventory comparison tests. Ruff and strict mypy pass for
the four added/modified implementation and test files. Test record:
`outputs/pytest_upper_full_final_20261105.xml`, SHA256
`fd62c8dd6d7a30fcfcba60f9f2c3df10c1e64f2ea488f482933d904824ef5b41`.

### Actual full spectra and source-grid refinement

Both external-source product runs are complete for the declared inventory:
launch 2026, five years, mean solar activity, SEP exceedance probability 0.1,
both assessment modes and shields 0, 0.01, 0.1, 1, 3, 10 g/cm2. Each report
contains 216 positive inputs and 1296 transport/LET cases; 14 exactly zero
sampled GCR products remain explicitly listed separately. No orbital
penetration calculation is part of this particular matrix.

Both authenticated full-inventory parent comparisons pass. Every previous
supported contribution is retained; added high-energy source measure is
conserved. In this particular shield matrix the below-anchor residual remains
unchanged. Zero-Al energy products remain exactly identical to the parent.

| Artifact in outputs/ | SHA256 |
| --- | --- |
| srim_upper_products_full_20261105.json | d35e107ac7b6d406ea8e31d3dca0b6558d138cc5dfd698742d44c6c68f33d786 |
| srim_upper_products_dense_20261105.json | 6329581471422c66bc9510b438d16e22217802536147207d78cf4a1679b53c4b |
| srim_upper_full_impact_20261105.json | 6ec070d12ae0f5a114e0823c0f2a14dad0dc8d750e8769a4dd50209937023c09 |
| srim_upper_dense_impact_20261105.json | 0d3b6824ba89dd09d7fdd7e5b0d08a7ab8bdfdce2c4c956d7c08d8ce0597e3cf |

With GCR 81 -> 161 and SEP 165 -> 329 source nodes and the baseline denominator
`abs(dense/base - 1)`, the maximum changes are:

| Integral | Maximum relative change | Location / qualification |
| --- | ---: | --- |
| Finite source integral | 0.287984% | GCR peak Co |
| Supported energy / mapped LET | 0.429850% | GCR peak U, 10 g/cm2 |
| Represented energy output | 0.429851% | GCR peak U, 10 g/cm2 |
| Source/mode/shield aggregate mapped LET | 0.291204% | SEP cumulative, 10 g/cm2 |
| Outside-Al upper tail | 1.147292% | GCR peak N; 115 positive-shield cases exceed 1% |
| Outside-Si tail at zero Al | 1.147292% | 23 ion/mode cases exceed 1% |
| Below-anchor residual | 1.657262% | GCR peak Ga, 0.1 g/cm2; 76 cases exceed 1% |
| Below displayed energy window | 12.245244% | GCR peak Ni, 0.1 g/cm2; 47 cases exceed 1% |

The last maximum concerns a baseline contribution of 2.8666373e-7 of that
ion's finite source count, with an absolute change of 3.5102674e-8 of the
source count. Its small size must not be mistaken for a <1% relative result.
The above-energy and outside-LET-display contributions are zero in these runs.
Thus supported output-integral convergence passes this check, but a blanket
claim that every integral or all Stage 4 criteria pass would be incorrect.

Actual remaining finite GCR source fractions above the native upper endpoints
are **8.667365% PEAK / 10.605228% CUMULATIVE** on the baseline grid and
**8.606933% / 10.531607%** on the dense grid. These match the prior prospective
normal-grid budget. SEP fractions are respectively 2.19805e-9 / 1.40502e-9
on the baseline grid. At positive shielding these are outside-Al contributions;
at zero Al the energy operator is an identity and missing Si LET support is
accounted separately. These fractions are particle-count coverage, never a
dose/SEE error bound or an assigned zero contribution.

Independent review reconstructed both reports without importing RADAR transport
operators, using its own raw SRIM parser, analytic log-log range/inverse,
piecewise-linear source integrals and all Si LET branches. Each report passes
371520 arithmetic checks including 355536 energy/LET-bin checks. Maximum
residuals scaled by source integral are 2.6175e-16 and 2.7881e-16. Units, source
identities, temporal conventions and both parent comparisons agree. The review
also independently confirmed the native rows, repeats and protected hashes.

Default profiles, source formulas and normative tables are unchanged. The
strict OST audit remains at 15 resources, five data-readiness findings and
`release_ready=false`. Native repeatability and arithmetic checks do not remove
the separate physics, source-domain, range-origin or orbital acceptance gates.

Next: justify the treatment above the observed native upper endpoints through
the declared source endpoints, including overlap/units/physical-component
checks; do not automatically splice another model or extrapolate SRIM. The
full-roster native extension is now available as pinned diagnostic data, while
normative replacement remains prohibited. Residual/refinement policy and the
orbital acceptance matrix remain explicit outstanding work.

## Direct workbook comparison and identified tail-cell identity (2026-11-05)

The user's question about how much the replacement differs from the original
workbook was checked directly against `<LOCAL_DATA>/ЛПЭ.xlsx`, SHA256
`d4ed7928331a0b7abcdfe1b585504b136809a3f6823f220edfed8967c28ec684`.
The workbook was only read, never recalculated, edited or exported. Its raw
numeric XML tokens were retained rather than rounded through binary64.

### Concrete defect and comparison scope

All **1656 cells** in `i_Si_e!B107:CO124` exactly equal the corresponding cells
in `i_Al_n!B107:CO124`: 92 element columns and 18 energies, 11000..45000 MeV
of **total ion energy**, not MeV/nucleon. Independent inspection also confirms
byte-identical complete XML cell elements, identical column headers and energy
coordinates, and no formulas in the block. This establishes saved-content
identity between electronic-labelled Si and nuclear-labelled Al data. It does
not establish how or when the workbook acquired that content.

`scripts/audit_legacy_workbook_srim.py` authenticates the workbook, the retained
15-ion legacy-mass native controls, both repeats of their raw outputs and the
current full-upper bundle. It compares exact energy nodes, with a unique
eight-binary64-ULP allowance only for workbook energy-coordinate serialization.
This allowance is not a tolerance on stopping/range values. Duplicate energies,
misaligned electronic/nuclear coordinates and cached formula values fail closed.
The audit checks **5487 values**, 1829 per metric: 1575 at or below 10000 MeV
and 254 above it. The retained He control ends at 12000 MeV; its 16 unmatched
workbook tail nodes are outside this particular controlled comparison.

| Compared quantity | Maximum absolute difference through 10000 MeV total | Meaning |
| --- | ---: | --- |
| Si electronic stopping | 0.576923% | Same retained legacy projectile masses |
| Si nuclear stopping | 0.194363% | Kept separate from electronic LET |
| Al range versus native projected range | 0.980392% | Legacy physical range definition is unqualified |

The denominator is the original workbook value. These percentages are observed
data differences, not physical uncertainty or an acceptance threshold. For
example, Fe56 at 11000 MeV total has workbook Si electronic stopping
0.0007206263437152665 versus native 2.426 MeV cm2/mg, a factor 3366.52.
Above 10000 MeV the Al discrepancy is separate: C12 at 45000 MeV has workbook
range 3889.0437738455621 mm versus native projected range 2670 mm (-31.3456%).
This does not justify identifying projected range with a CSDA range or scaling
one to match the other.

The workbook contains 92 element columns in these sheets; the former RADAR
HZE resources selected only 15 of them. The present 84-ion roster comes from
positive OST source populations, not from an assertion that the workbook
contains only 15 ions. Ten of the old 15 integer mass parameters differ from
the literal source A used by the new pipeline. Those differences must not be
mislabelled as stopping-model error. For the five unchanged masses (He, C, N,
O, Cr), all **1198 matched native Al/Si rows** agree exactly with the current
full-upper bundle. The 0.576923% result is not a universal bound for all 84
current ions.

Final audit: `outputs/legacy_workbook_srim_audit_final_20261105.json`, SHA256
`174c0d0b6b863ebd871ddebd9edd97bf05864fe37097ad5615b265e87e01faab`.
Independent review, without importing this audit, verified all 5487 entries,
six summaries, input hashes and current same-mass rows directly from workbook
XML and raw SRIM text. Earlier exploratory audit files are not the final
complete-node comparison. Workbook and production-table hashes remain unchanged.

## Full-roster upper-band SRIM/CATIMA comparison (2026-11-05)

`scripts/compare_srim_upper_overlap.py` and
`scripts/generate_srim_upper_overlap_controls.py` extend the stopping comparison
to the new full-upper bundle without changing the earlier frozen workers.
The authenticated inventory contains **3066 exact native nodes**, 1533 per
material, for all 84 ions in Al and Si over **1000..5000 MeV/literal source A**.
The other 43378 native rows are explicitly outside this comparison band;
no source spectrum is truncated. Total energy divided by literal A is evaluated
as an exact fraction before conversion to the backend's binary64 coordinate.

The pinned CATIMA configuration, Pierce-Blann prescription, LS corrections,
elemental ionization potentials and default target constants are unchanged.
SRIM MeV cm2/mg values are multiplied by 1000, with no additional A factor.
Electronic, nuclear and nonradiative-total stopping are compared separately.
The radiative difference is exactly zero at these nodes, all below the backend's
50000 MeV/u radiative threshold. No range, spectral convolution, fitted factor
or joined transport operator is produced.

Both native runs are byte-identical, as are the two comparison replays.
Node-count statistics for electronic stopping use `100*(CATIMA/SRIM - 1)`:

| MeV/source A band | Nodes per material | Al median absolute difference | Si median absolute difference | Al maximum absolute | Si maximum absolute |
| --- | ---: | ---: | ---: | ---: | ---: |
| [1000,3000) | 1057 | 6.636664% | 6.536451% | 16.482984% | 16.418950% |
| [3000,4400) | 361 | 6.381483% | 6.372413% | 15.713569% | 15.734653% |
| [4400,5000] | 115 | 6.041166% | 6.044066% | 14.674444% | 14.655321% |

At each ion's actual upper node, Si electronic differences are +0.035916% for
He4, +3.381257% for Ne20.2, +5.850996% for Fe55.8, +14.655321% for Th232 and
+12.910557% for U238. These are model-to-model differences, not experimental
errors, source-weighted spectral effects or permission for automatic splicing.
This overlap does not validate continuation above 5000 MeV/source A.

| Artifact in outputs/ | SHA256 |
| --- | --- |
| srim_upper_overlap_requests_20261105.json | dadfd486b1dc7f62f42139d3d8282a5023be60f8bd0e4bb27154997ba66b216c |
| srim_upper_overlap_native_20261105.json and native_repeat | cb300846aad97322420532bd719f472e6f4cdf0335dce7aea4f690038216d657 |
| srim_upper_overlap_comparison_20261105.json and comparison_replay | 078b25d0aa08f968abd25c988f67e708c016bc2cf498f30ff0328b0498064404 |

The `py -3.12` launcher reported no installed interpreter in this turn; no
backend ran in those attempts. Both successful runs used the existing bundled
CPython 3.12 executable, whose version is retained in the reports. No package,
system runtime or source program was installed or modified.

Independent arithmetic review checked all 3066 raw SRIM node identities,
36792 component-field identities and 528 component-group summaries without
importing RADAR/CATIMA operators, including the nuclear-term formula replay,
unit conversion, component closure and repeat/evidence hashes.

The next bounded independent-code comparison can use the original
[GSI ATIMA V1.41 table interface](https://web-docs.gsi.de/~weick/atima/atima14.html),
which the [official method description](https://web-docs.gsi.de/~weick/atima/)
distinguishes from the extended CATIMA interface. Proposed anchors are
He/Ne/Fe/U in Al/Si at their individual SRIM upper node, 10000, 50000 and
79432.823472 MeV/source A. These 32 points have **not** been executed and CGI
availability/output conventions are not yet verified. This is a possible
comparison route, not a selected production replacement or independent
experimental truth. The radiative transition needs separate one-sided checks.

Combined targeted regression: **1993 passed**, no errors, failures or skips,
JUnit duration 45.504 s. This includes 92 workbook-audit and 73 upper-overlap
tests. Ruff passes all five new code/test files; strict mypy passes in the
respective existing script/package import layouts. JUnit record:
`outputs/pytest_workbook_upper_overlap_final_20261105.xml`, SHA256
`fa43aea9fc42cd70f50105f52992d7bf344bfbbe0456eb75134da250d0f22f2f`.

No OST coefficient, default transport profile or normative resource was changed.
Strict OST still reports 15 resources, five data-readiness findings and
`release_ready=false`. Stage 4 remains incomplete; the high-energy treatment
and its qualification are not replaced by successful model comparisons.

## Original-GSI high-energy anchors and radiative threshold (2026-11-05)

The previously proposed independent-implementation comparison has now been
performed. This closes a bounded implementation/unit check, **not** the choice
or normative qualification of a high-energy transport model. No source
coefficient, production table, default profile, range operator or spectrum was
changed. In particular, this does not select or normalize an SRIM/CATIMA splice.

### Scope and source conventions

`scripts/generate_high_energy_anchors.py` evaluates 48 pinned CATIMA samples:
He4, Ne20.2, Fe55.8 and U238 in Al and Si, each at the actual final native SRIM
node, 10000, 50000 and **79432.82347242821 MeV/literal source A**, plus the two
adjacent binary64 values immediately below and above 50000. The 32 main points
and 16 one-sided checks are labelled separately. The final source-grid centre
is retained exactly; neither its previously printed rounded value nor the
100000-MeV reference boundary is substituted. Fractional source A is not
silently replaced by an isotope mass. Electronic, nuclear, nonradiative and
radiative stopping remain separate, with whole-ion units MeV cm2/g.

The 32 main points were queried through the original
[GSI V1.41 web interface](https://web-docs.gsi.de/~weick/atima/atima14.html),
not the extended CATIMA interface. The
[official description](https://web-docs.gsi.de/~weick/atima/) identifies the
former as a lookup using tables from the FORTRAN implementation. Four official
source pages and all 32 responses are retained as original bytes with hashes,
explicit text decoding, timestamps and full request URLs. No downloaded
executable, Linux wrapper or 1.5-GB spline archive was installed or executed.

The form's mass-edit handlers contain `parseInt` and inconsistent case in field
names. This is an unsafe input path, not evidence that an observed calculation
actually rounded A. Direct GET requests and exact checks on both input echoes
and calculated projectile mass establish that **20.2 and 55.8 were preserved**
in these runs. Natural targets were requested with `at=0`; observed masses are
26.981541 and 28.085501, with ionization potentials 166 and 173 eV. CATIMA's
corresponding target masses are 26.9815 and 28.0855. The CGI does not report
target density, so full material-constant equivalence is not established.

The interface says V1.41 but each calculation response prints **Atima 1.4**.
Both labels are preserved; the report does not silently upgrade the response
version. The printed `dE/dx` is generic stopping without a component breakdown.
It is therefore **not identified as electronic LET** or assumed to equal a
particular CATIMA component.

`scripts/gsi_atima_anchors.py` checks the unit convention independently using
`(E_in-E_out)*A/thickness` with the 1-mg/cm2 target and propagation of printed
rounding. All 32 finite-foil/entrance-stopping ratio intervals intersect the
declared [0.999,1.001] convention-check band. This is neither exact equality of
a derivative and finite difference nor a 0.1% physical-accuracy criterion.
The CGI's printed stopping already refers to the whole ion: only the factor
1000 from mg to g is applied, **no additional factor A**.

### Observed comparisons, not accuracy bounds

Every component is compared with the same generic GSI value as
`100*(CATIMA_component/GSI_printed_stopping-1)`. Maximum absolute electronic
differences over each ion's eight main points are:

| Literal source ion | Maximum absolute difference from generic GSI stopping |
| --- | ---: |
| He4 | 0.008831% |
| Ne20.2 | 0.016098% |
| Fe55.8 | 0.011486% |
| U238 | 0.353681% |

The overall nonradiative maximum is 0.353400%. These descriptive differences
are rounded upward in this table, not adopted as tolerances. Six-decimal CGI
stopping has half a printed unit of 0.0005 MeV cm2/g; the report propagates it
to both bounds of every comparison interval. For He this is about 0.0075%
near its native upper node, so many small differences are print-limited.
Shared LS theory and differing charge prescriptions remain relevant; this is
an independent code-path comparison, not an independent theory or experiment.

CATIMA's strictly `T > 50000` pair-production/bremsstrahlung condition produces
a nonzero jump at the next representable energy above the boundary. The
radiative contribution is **0.1503356..0.1687668% of nonradiative stopping**
there, and **0.3293618..0.3594014%** at the actual final GCR centre. It is
exactly zero at and below 50000 in the pinned backend. These observations do
not establish local energy deposition or authorize adding the radiative term
to electronic Si LET. The threshold has not been smoothed away.

Three GSI responses print Fe charge slightly above its atomic number:

| Target / energy MeV/A | Printed entrance / exit charge |
| --- | --- |
| Al / 4928.31541218638 | 26.000002 / 26.000000 |
| Al / 79432.82347242821 | 26.000002 / 25.999998 |
| Si / 50000 | 26.000006 / 26.000004 |

The parser retains the literal values, exact excess and an explicit physical
bound warning, without clamping or an arbitrary small-excess tolerance. Charge
is not used by our comparison arithmetic. The cause and influence on the
underlying GSI calculation are **unknown**, not dismissed as harmless rounding.
All results remain diagnostic. These are observations about external output,
not errors attributed to OST.

### Reproducibility and limits

Both local native runs and both offline GSI comparison replays are byte-identical.
The remote service was queried once per retained case; a remote reproducibility
campaign is not claimed. Transport timeouts and TLS EOFs interrupted acquisition.
`scripts/fetch_gsi_atima_evidence.ps1` completed the same bounded URL plan through
Windows TLS, without disabling certificate checks, then Python independently
validated every retained byte stream and input. The immutable per-URL cache
avoids re-requesting successful responses. Final assembly pins the current
collector/helper versions; timestamps and raw hashes, not a claim that all
earlier retrieval attempts used one code version, identify the HTTP evidence.
The PowerShell 128-KiB validation is post-download, not a hard network byte cap.

Independent review without importing the new operators checked all 48
coordinates, eight raw SRIM endpoint records, 197 implementation/production
hashes, six source/package pins and 35 unpacked wheel members. Direct replays
of the pinned nuclear and radiative formulas agree to relative differences
4.44e-16 and 8.33e-14 respectively. A separate raw-data review checked all 36
GSI byte/hash/decoding records, 32 request identities and unit checks, 96 point
differences, 192 rounding-interval bounds and the three charge warnings.
None of these checks removes the physical-qualification limitations.

| Artifact in outputs/ | SHA256 |
| --- | --- |
| high_energy_anchors_final_20261105.json and final_repeat | 466a975f47d66f83662a26a740ecab04488c947f688517525a4bcaf10398d604 |
| gsi_atima_fetch_plan_20261105.json | c4fc139b839cdac016f6be3d4b3918e0670189cc30caa8c53ca98e3d1c890d64 |
| gsi_atima_anchors_raw_20261105.json | c63d516407cdf9b37be3c51bdb4107e6e924a59e86a3263dc6c1123a78291b11 |
| gsi_atima_anchors_comparison_20261105.json and comparison_replay | 6ab87e063941513ef0c0b5640df2c24793d4231c00777d86ef7db7b78e17060a |
| pytest_gsi_high_verified_20261105.xml | fc89c2f8e80fa20c4ebb5026e3b98e7332ce33b6bfa761cb2b54f8605bb12fc8 |

The earlier `high_energy_anchors_20261105.json` is exploratory: its code snapshot
predates a final annotation correction. Use only `final`/`final_repeat` above.
Combined targeted regression: **2164 passed**, including 71 new local-anchor
and 100 GSI evidence/parser tests. Ruff and strict mypy pass the four new Python
code/test files. The PowerShell acquisition helper was also statically reviewed
and successfully exercised on the retained public requests.

Strict OST remains **15 resources, five findings, release_ready=false**; the
integrity manifest remains SHA256
`b5f1b04cd3da03ac458bfbbb85a3716622debfe234aabbd933f9d3ea2bd87592`.
This comparison supports continuing the high-energy assessment but does not
close Stage 4. The next substantive gate is an explicitly declared treatment
of the SRIM/relativistic-model overlap and its effect on protected energy/LET
spectra. No forced endpoint scaling, extrapolation or automatic model join is
authorized by the close ATIMA/CATIMA point values. Full ion coverage, remaining
support/residual checks and the orbital acceptance matrix remain separate gates.
