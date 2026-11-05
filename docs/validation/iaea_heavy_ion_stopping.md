# Stage 4: experimental heavy-ion stopping comparison (2026-10-31)

Absolute user-profile and attachment-directory identifiers are intentionally
replaced by local placeholders in this public note; source filenames and
hashes are preserved. Substitute the local interpreter/data locations before
running the example commands.

## Decision and scope

**INCOMPLETE; no production replacement or normative qualification.** The
candidate now has a reproducible comparison with experimental compilation
records, in addition to the previous inter-model and He/ASTAR comparisons.
The calculation uses the same pinned CATIMA configuration, without fitting
coefficients, changing material ionization potentials or rounding OST masses.
This is a stopping-power check, not validation of protected spectra, absolute
CSDA ranges or the full orbital acceptance matrix.

The Stage-4 criterion of less than 1% change on grid refinement is a numerical
convergence criterion for the specified integral results. It is not a universal
1% physical-accuracy requirement against experimental stopping data. Targeted
primary-source review informs the applicability and uncertainty of candidate
data; resolving every historical paper is not an additional mandatory gate for
Stage 4. The uploaded Ri1983 and Zhi1998 full texts have now been read in full;
the dated follow-up below supersedes their earlier access-limited status.

The IAEA landing page describes **electronic** stopping, assuming nuclear
stopping has been removed or is negligible. This is the compilation's stated
convention, not proof that every original experiment made the same correction.
The [database authors' review](https://arxiv.org/html/2402.03080v3), section 1,
also warns that experimental analyses can use stopping models for thickness,
normalization and nuclear/electronic separation. Experimental comparisons must
not automatically be called independent out-of-sample validation.

## Sources, selection and lossless handling

Raw files are retained unchanged in `research/iaea_stopping_20261031/`:

| Source | SHA256 |
|---|---|
| [IAEA data ZIP](https://nds.iaea.org/stopping/download/tuples), `data.zip` | `0813d4137aa571eb8d8d48e96ed7ca3dd2b7ef10e510d627e7d01029379ba01b` |
| [IAEA references ZIP](https://nds.iaea.org/stopping/download/refs), `references.zip` | `9d5b2bfec9aa4b51ad5a5fc64e1d6b34228d38f1e5c263f23b886159e03faed4` |
| [IAEA landing page](https://nds.iaea.org/stopping/), `index.html` | `61fa5a4ece771421fc376eca6e6dd772614129fa9051a522968c7f416b86338a` |
| [Javanainen author report, ESA 2009](https://escies.org/download/webDocumentFile?id=19698), `javanainen_esa_2009.pdf` | `fb9cca73e7992e1547afb7eccc3b22afbcd186f96acceb5639517f5134604cc9` |
| [Weick et al., NIMB 164-165 (2000), 168-179](https://web-docs.gsi.de/~wolle/EB_at_GSI/SLOWED-DOWN_BEAMS/PAPERS/weick.pdf), `weick_nimb_2000.pdf` | `945dd542e1a4616045b8ab22c7f495909505e3f83e2071bdfa3edd61e8727986` |

`scripts/iaea_stopping_reference.py` reads the single ZIP members without
extracting them. It retains every raw CSV field and logical record number
(header = record 1; quoted multiline fields do not create extra records).
Every selected observation includes the raw fields, bibliography, normalized
coordinates, uncertainty text and duplicate identity. Exclusions keep their
record-number ledgers, referring back to the pinned archive. This is read-only
tabular analysis: no cleaning, smoothing, silent deduplication or source edits.

The archive contains **274092 records**, not 274092 independent measurements.
It includes 208230 rows under five identified model references: `SRIM2013`,
`ICRU49`, `MSTAR2003`, `ESPNN`, `DPASS21.06`. Those are excluded. The remaining
65862 records differ by 593 from the landing page's advertised 65269 data
points for release 2026-09. The cause is not established: download bytes are
pinned, but equality to the advertised release is not asserted.

Selected conditions are elemental Al/Si, literal solid phase `s`, heavy ions
with Z = 3..92, explicitly given positive integral isotope labels, supported
native units and energies 0.01..100000 MeV/u. Fractional or missing projectile
masses are not rounded or inferred. Whole-ion energies in MeV or keV are divided
by the explicit A; `/u` coordinates are not divided again. Target molar mass
is separately used only when converting an atomic stopping cross section.

There are **2612 selected records**: Al 1583, Si 1029; 52 explicit isotopes,
79 isotope/material groups and 97 reference IDs. These are not 97 verified
independent experiments. There are 2611 distinct raw rows: the identical
`Jav2009` Si30 records 248699/248700 are both retained. Of the selected records,
69 have blank uncertainty; ranges such as `3-7` are kept as text. A scalar
percentage is not silently interpreted as a Gaussian standard deviation,
confidence interval or physical acceptance tolerance.

Independent conversion is `MeV cm2/mg * 1000 = MeV cm2/g`. For native
`1e-15 eV cm2/atom`, it is `value * 1e-15 * N_A / M_target / 1e6`, using
exact SI `N_A = 6.02214076e23 mol^-1`. No density or projectile-A factor enters
this conversion. One selected row (`Nte2023`, CSV record 200868, N14/Si,
153 keV total) differs from the provider's converted column: independently
1732.5582104611003 versus provider 1732.0774200020855 MeV cm2/g, or
`provider / independent - 1 = -0.0277503206595%`. Both are preserved; no
conversion constant is adjusted to make them agree.

## Native controls and result definition

`scripts/generate_iaea_stopping_controls.py` evaluates **2538 unique exact
IAEA isotope/material/energy keys**, preserving links to all 2612 source rows,
plus seven RADEF keys. The 2545 native calculations use explicit isotope A as
the numeric backend mass parameter in u, not measured isotope masses and not
the fractional OST source-mass convention. Native values are whole-projectile
MeV cm2/g. There is no extra multiplier A, time factor, angular integral,
shielding interpolation or spectrum interpolation in this comparison.

Full, electronic, nuclear and radiative losses and relative native ranges are
retained. Electronic stopping is nonradiative total minus the independently
replayed native nuclear term. At 2444 energies <=10 MeV/u it also agrees with
the backend's exposed electronic function to relative error <=2.23e-16. This
is arithmetic/internal closure, not experimental validation of nuclear loss.
The backend's absolute range origin remains unqualified.

`scripts/compare_iaea_stopping.py` rebuilds the selection and checks the complete
source, implementation, configuration, material, production and request
identity before and after calculation. Missing rows, relabelled units/masses,
changed inputs, duplicate JSON keys and nonfinite numbers fail closed. All
physical/release qualification flags must remain literally false.

For each observation the signed difference is
`CATIMA_electronic / independently_converted_experiment - 1`.
The provider-converted denominator is recorded separately, never substituted
silently. Descriptive summaries retain both all-record and exact-raw-unique
counts, grouped by reference, isotope/material and energy band. No result is
weighted by a guessed uncertainty or selected for agreement with the model.

## Observed differences and interpretation

The median absolute difference is **5.020277%** across all selected records
and **5.020901%** across distinct raw records. Neither is a model accuracy
bound. The detailed report retains 440 group summaries and every residual.
Examples with source-interpretation limits, without deleting or correcting them:

| Record / reference | Conditions | Signed difference | Why the interpretation remains open |
|---|---|---:|---|
| 16 / Ri1983 | Ag107/Al, 0.05607 MeV/u | -39.090558% | The compilation citation itself notes erroneous Table 1 values and personal communication. Whether the retained numbers already incorporate a correction is not established. |
| 20534 / Zhi1998 | Au197/Si, 0.01538 MeV/u | +133.254050% | Full-text review now confirms the source-table scale and units to <0.02%. Electronic stopping was inferred from projected ranges with fixed model nuclear stopping and a range/projection relation; it was not a direct model-free measurement. |
| 208120 / Tp1962 | Ne20/Al, 0.08384 MeV/u | +85.085697% | Multiple large deviations in the same historical series require review of its measurement and reduction method. |

At the original comparison checkpoint, direct DOI retrieval for the first two
papers was unavailable and their full texts had not been verified. This is
historical access information: user-supplied copies were subsequently read in
full on 2026-11-01, as recorded below. No fitted correction or normative physical
tolerance is inferred from either the residuals or the later source review.

The seven RADEF measurements were visually checked in the **measured** column
of page 6, not the adjacent SRIM/LETCalc columns. Page 5 identifies B-TOF plus
magnetic-dipole measurements with approximately 1 and 12 micrometre Si samples.
The slide alone does not explicitly establish nuclear subtraction, so both
electronic and nonradiative-total predictions are retained separately.

| Ion at 9.3 MeV/u in Si | Published measured LET, MeV cm2/mg | Electronic prediction difference |
|---|---:|---:|
| N15 | 1.88 | -3.909079% |
| Ne20 | 3.64 | -2.033255% |
| Si30 | 6.73 | -1.771408% |
| Ar40 | 10.1 | +1.772294% |
| Fe56 | 18.5 | +2.354031% |
| Kr82 | 30.2 | +3.697867% |
| Xe131 | 55.3 | +0.686069% |

Adding the native nuclear term changes these differences by only
0.0496..0.0846 percentage points. The reported `<3%` uncertainty is retained
as an upper-bound statement, not a universal acceptance threshold or exactly
3% one-sigma error. These seven points may overlap the IAEA compilation; they
are not counted as independent extra experiments.

## Energy coverage and what is not established

Selected measurements reach **985 MeV/u for Al and 90.63 MeV/u for Si**. Only
eight Al observations exceed 250 MeV/u. The Weick paper independently confirms
the experimental method for relativistic Au/Pb/Bi in solid targets, including
charge-state-equilibrium preparation, but its IAEA reference also includes a
revised dissertation. The exact origin of every compiled point is not thereby
established. No Si observations or high-energy tail are manufactured from that
paper, and these experimental data are not declared independent of historical
ATIMA development/calibration.

The attempted retained Datz high-energy download was an HTML challenge, not a
PDF; it is labelled `datz_prl_1996_response.html`. No numerical control was
derived from those bytes. Full candidate coverage to 100000 MeV/u remains a
separate open task. Matching GSI ATIMA alone would be a cross-implementation
check, not independent physical verification of shared model ingredients.

Data access is not a resolved redistribution/integration license for every
source or the backend. The research archives and diagnostic output do not
establish permission for a normative software release.

## Reproduction and checks

- Native output: `outputs/iaea_stopping_native_controls_20261031.json`, SHA256
  `994971e3fcd22ae360535ff166cb5a71cf00a32f84fbc14b38b9f75346cc1860`.
- Comparison: `outputs/iaea_stopping_comparison_20261031.json`, SHA256
  `4ea2168385dde6c28034a37b2aea4cb8c84d3f1883a151319dfb2e8c36683dea`.
- Both `*_replay_20261031.json` files reproduce their respective originals
  byte-for-byte. The retained comparison draft has the same bytes as the final.
- 272 new tests pass: parser 78, generator 69, comparison 125. Ruff and strict
  mypy pass all six new implementation/test files.
- Independent direct CSV/Decimal/native-JSON arithmetic reproduced all 2612
  residuals and 440 group summaries without differences. The independent
  checks did not call the comparison arithmetic or the parser conversions.
- The first combined test attempt had five temporary-directory setup errors
  from the sandbox's malformed user-temp path; rerunning with a fresh explicit
  workspace `--basetemp` passed. No test or assertion was disabled.
- All 166 production/source-resource hashes remain unchanged. No normative
  table, physical formula, coefficient or release blocker is changed here.

The final broad regression passed **3994 tests in 388.36 s**, with zero
failures, errors or skips (`outputs/pytest_iaea_broad_20261031.xml`). This
includes the 272 new tests; the two counts must not be added. The 17 expensive
tests in `test_gcr_upper_domain.py` and `test_sep_upper_domain.py` were explicitly
excluded, as in the preceding broad checkpoint, and were not rerun here.
Final focused results are retained in `outputs/pytest_iaea_all_20261031.xml`.
`git diff --check` and an explicit whitespace check of the new files pass.
The live strict OST audit still examines 15 resources and reports six blocking
findings in five production tables, with `release_ready=false`.

```powershell
& '<CPYTHON_3_12>\python.exe' -B scripts/generate_iaea_stopping_controls.py research/iaea_stopping_20261031/data.zip research/iaea_stopping_20261031/references.zip --output outputs/iaea_native_replay_new.json
.\.venv\Scripts\python.exe scripts/compare_iaea_stopping.py research/iaea_stopping_20261031/data.zip research/iaea_stopping_20261031/references.zip outputs/iaea_stopping_native_controls_20261031.json --native-sha256 994971e3fcd22ae360535ff166cb5a71cf00a32f84fbc14b38b9f75346cc1860 --output outputs/iaea_comparison_replay_new.json
```

The next bounded decision is to determine a defensible candidate-data domain
and explicit unresolved regions using the available source evidence, including
the follow-up below. It does not require resolving every historical outlier.
High-energy theory/implementation checks, actual interpolation/transport
effects, e/p tables, orbital issues and the final Stage-4 acceptance matrix
remain separate gates. A global descriptive median cannot replace any of them.

## Original-source follow-up (2026-11-01)

This is a source review, not a change to the model or the prior comparison.
The original numerical report remains byte-identical. No observations are
removed, relabelled as verified or used to fit a correction.

The table gives the latest status after receipt of the two uploaded PDFs.
Earlier access attempts are retained below explicitly as history, not as current
missing-full-text blockers.

| Series | Primary evidence obtained | Status after review |
|---|---|---|
| Tp1962 | Full official JETP PDF; relevant pages visually checked | Original measured-loss convention and units checked; exact compilation derivation and isotope assignment unresolved |
| Ri1983 | Full user-supplied PDF, all four pages read and visually checked; pinned IAEA correction warning | Published units, Table I and model-assisted electronic/nuclear separation checked; numerical correction provenance remains UNKNOWN |
| Zhi1998 | Full user-supplied PDF, all four pages read and visually checked; retained publisher metadata | Projected-range method and model-assisted electronic stopping checked; two selected values reproduce the source-unit conversion to <0.02%, without resolving the physical model discrepancy |

### Tp1962: full article available, but the discrepancy is not resolved

The [official JETP article](https://jetp.ras.ru/cgi-bin/dn/e_015_01_0031.pdf)
was retained as `research/iaea_source_review_20261101/tp1962_jetp_en.pdf`, SHA256
`81a267cf4aee0e8093e2a8debb207ed20c84d0e0e6ed7debbdc7e44f92aae98a`.
Printed pages 33, 35, 36, 38 and 39 were inspected visually, including captions.

Page 33 derives solid-film loss from air-equivalent thickness and range versus
velocity, assigning `Delta E / Delta x` to mean energy `(E + E1)/2`; the stated
absolute error is 10-15%. Pages 35-36 identify Ne in Fig. 4d, with filled
circles for Al, stopping in keV cm2/microgram and energy in keV/nucleon.
Pages 38-39 distinguish electronic, nuclear and charge-exchange contributions;
Table III has separate total and nuclear-subtracted columns, but no Ne row.
Thus the plotted measured loss is not itself a verified electronic-only table.
The historical three-part decomposition is not assumed identical to the
modern backend's electronic/nuclear split.

The unit identity is exact:
`1 keV cm2/microgram = 1 MeV cm2/mg = 1000 MeV cm2/g`.
No missing factor 1000, projectile A or density is indicated. The graph is
compatible with the scale of compiled Ne-Al values, but exact digitization,
subsequent curator processing and the explicit Ne20 assignment were not
independently established from the article.

**New arithmetic check, not an experimental correction:** all eight retained
Tp1962 Ne20/Al observations were matched to their exact native keys. For record
208120 at 0.08384 MeV/u, the retained experiment is 3070 MeV cm2/g, native
electronic stopping is 5682.130902814593 and nuclear is 91.76816197564719.
`Se / measurement - 1` is +85.085697%; `(Se + Sn) / measurement - 1` is
**+88.074888%**. Across all eight points these two ranges are +47.394..85.086%
and +53.086..88.075%, respectively. If the compilation retained total losses,
comparing native total rather than electronic stopping increases this gap.
This alternative therefore does not explain the disagreement. No nuclear term
has been subtracted from the observed values to manufacture agreement.

### Ri1983: full article reviewed; numerical correction still UNKNOWN

The [APS publisher abstract](https://doi.org/10.1103/PhysRevA.28.3234) identifies
*Stopping of silver ions in solids*, Phys. Rev. A 28, 3234-3237 (1983), and
describes recoil stopping inferred from Doppler shifts of gamma rays, with
electronic stopping cross sections compared to a semiempirical model. This
supports the subject and experimental method only, not the numerical table.

The later user-supplied full article is
`<LOCAL_ATTACHMENTS>/1-ribas1983.pdf`,
SHA256 `429a04c44a31ee74d6fa4d898bbe887736fe3ed41dbf330e8cabecb8f5936e67`.
All four pages (printed 3234-3237) were read and visually checked on 2026-11-01.
Page 3234 explicitly identifies 107Ag recoils produced using 16O beams, with
metal stoppers Al, Ti, V, Fe, Ni, Zn, Zr and Pd; there is no Si measurement.
The stated recoil-energy domain is approximately 50-200 keV/amu.

Page 3235 explains that a direct extraction of point stopping powers is not
possible because appreciable energy is lost in the foils. Electronic stopping
is fitted to Doppler-shift centroids using Monte Carlo and the modified LSS
form `-(d epsilon/d rho)_e = k * epsilon^p`. Nuclear stopping is included using
an LSS model with its multiplier `eta = 1`, justified by the authors' estimate
that nuclear stopping is less than 20% of the electronic contribution. This
is model-assisted separation, not subtraction of independently measured
nuclear stopping.

Table I on page 3236 gives `Se = k * (v/v0)^(2p)` in **MeV cm2/mg**; the
printed Al row is `k = 8.3(6), p = 0.7`. The exponent is visually confirmed as
`2p`, not `p`. The authors limit the fitted parameters to their stated roughly
5-20 MeV interval and estimate stopping uncertainties at about 7%; this is not
a general acceptance tolerance or an identified Gaussian standard deviation.
The paper supplies fitted parameters, not an eight-point Ag/Al stopping table.
Its four incident-beam settings and jointly fitted response do not justify
calling the eight selected compilation rows eight independent measurements.

The pinned IAEA reference record warns of erroneous Table 1 values based on
personal communication. It does **not** specify a replacement table, a
correction formula, or whether the retained compilation numbers already
incorporate that correction. In the selected Al subset, eight records remain
unchanged (CSV 16-23). Calling their residuals established CATIMA errors, or
calling all eight experimental values erroneous, would both exceed the evidence.

Historically, ordinary access to the APS PDF returned a sign-in/unauthorized
response, so that attempt obtained no full PDF. No access controls were
bypassed and no response HTML was presented as a paper. The uploaded PDF
resolves that access limitation, but contains no appended correction. A public
erratum was not located in the earlier bounded publisher/title search; that is
not proof that no correction exists. What remains UNKNOWN is the numerical
replacement and whether the IAEA rows incorporate it. The full text alone is
not evidence of either a corrected compilation or a demonstrated CATIMA error.

### Zhi1998: source units checked; electronic stopping is model-assisted

Crossref's publisher-deposited record and Elsevier metadata agree on
*Stopping power for MeV energy Au ions in silicon*, NIMB 135(1-4), 128-131
(1998), DOI `10.1016/S0168-583X(97)00586-7`. Inverted given/family-name fields
in deposited metadata are not evidence of a wrong article link.

Retained metadata from the initial access-limited review (not full text):

- `research/iaea_source_review_20261101/zhi1998_crossref.json`, from
  [Crossref](https://api.crossref.org/works/10.1016/S0168-583X(97)00586-7), SHA256
  `4d56741f6da60a7edce1fca62fd48023d48ea62aebeb65acd4fe4b73c45b9d28`.
- `research/iaea_source_review_20261101/zhi1998_elsevier.xml`, from the
  [publisher API](https://api.elsevier.com/content/article/PII:S0168583X97005867?httpAccept=text/xml), SHA256
  `0395994c3d9f998d490c4335232e45fe83947cc794a743ce28e82d05b531593c`.
- `research/iaea_source_review_20261101/zhi1998_openalex.json`, SHA256
  `324640a2c1dcbd6b18a11e59d901115ceb0e0e321bffc5ba5f05f77e2e7d1b4e`:
  discovery of an open copy only; not evidence of the experiment's physics.

At that initial checkpoint, Elsevier returned metadata with `openaccess=0`,
not article text, and no open full copy was found through the attempted
publisher and institutional routes. This is superseded by the user-supplied
full article
`<LOCAL_ATTACHMENTS>/2-yongjun1998.pdf`,
SHA256 `221eca8efccc4c278ae16f3b9bc15bd872ed8ede2d266f7da0bf0ea261a7edf9`.
All four pages (printed 128-131) were read and visually checked on 2026-11-01.

Page 129 specifies Au+ implantation into Si at **1.23, 2.13 and 3.03 MeV total
ion energy**, with the Si <100> axis 7 degrees from the beam to suppress
channeling and liquid-nitrogen cooling. The paper does not explicitly label
the Au isotope as 197. The isotope is explicit in the IAEA records; its use
for the following energy conversion must not be presented as a newly verified
isotope statement in the article.

Table 1 on page 130 contains **projected ranges** `Rp = 0.310, 0.551, 0.843`
micrometres, not CSDA ranges. Stopping powers are inferred from these ranges
using fixed Biersack nuclear stopping, the range integral in equation (4), a
model relation between mean and projected range, and the fitted electronic
form `Se = A * (E/E_Bohr)^P`. Page 131 gives `A = 185`, `P = 0.02` and Table 2
electronic stopping values **180.0, 182.0 and 183.2 eV/(10^15 atoms/cm2)**.
Thus this is not a pointwise, model-free electronic-stopping measurement or a
direct experimental measurement of the nuclear term. The reported 12% error
comes mainly from range measurement; no Gaussian/error-confidence convention
is imposed on it here. The same table already reports substantial differences
from TRIM-91 (275.1, 362.0 and 431.7 in the same units).

Using exact SI `N_A` and the retained Si target molar mass `28.085 g/mol`, the
source-unit conversion is `Se * 0.602214076 / 28.085` in MeV cm2/mg:

| Total energy, MeV | Table 2, eV/(10^15 atoms/cm2) | Independently converted, MeV cm2/mg | IAEA, MeV cm2/mg | `IAEA / converted - 1` |
|---|---:|---:|---:|---:|
| 2.13 | 182.0 | 3.902544484 | 3.902 | -0.013952% |
| 3.03 | 183.2 | 3.928275546 | 3.928 | -0.007014% |

Both differences are less than 0.02% and consistent with the compiled values'
precision. With the compilation's explicit `A = 197`, `E/A` is
0.01081218274 and 0.01538071066 MeV/u, matching the rounded compiled energies
0.01081 and 0.01538. The 1.23 MeV point corresponds to about 0.00624 MeV/u and
is below the diagnostic candidate's 0.01 MeV/u lower limit. These checks rule
out a missing factor 1000 or a second projectile-A conversion as an explanation
of the two selected residuals; they do not establish physical accuracy.

The page-130 assertion that the range integral necessarily diverges at zero
is not a universal mathematical result. For its fitted positive electronic
term proportional to `E^0.02` and nonnegative nuclear stopping, the inverse
total stopping is bounded above by a constant times `E^-0.02`, which is
integrable at zero. This checks the formal extrapolated model, not its physical
validity down to zero energy. The article's 0.1 keV integration cutoff is not
adopted as a RADAR rule or used to fix the candidate's range origin.

### Consequence for Stage 4 and reproducibility

The updated review does not establish a production formula or table correction.
The Ri1983 and Zhi1998 full texts are no longer missing. They clarify that the
published electronic stopping values depend on experimental reduction models;
Zhi1998's source units and two selected tabulated values are now cross-checked.
Ri1983's private numerical correction and its application to the compilation
remain UNKNOWN. Together with the Tp1962 convention issue, these findings
bound the interpretation of the unchanged diagnostic residuals rather than
turning them into unconditional model-error measurements. None of the three
sources alone provides unconditional calibration/acceptance of the candidate.
No automatic exclusion, smoothing, uncertainty reweighting or fitted scaling
was performed.

These individual articles and resolution of every discrepancy are not new
mandatory Stage-4 barriers. They are supporting evidence for the data actually
used by the selected profile and its stated domain. Required provenance,
physical definitions, complete spectra, the specified numerical convergence
checks and the full acceptance matrix remain distinct requirements. In
particular, the 1% grid-refinement criterion must not be reinterpreted as 1%
agreement with every experiment or as a universal physical accuracy claim.

On this review date the native-control SHA256 remains
`994971e3fcd22ae360535ff166cb5a71cf00a32f84fbc14b38b9f75346cc1860`
and comparison SHA256 remains
`4ea2168385dde6c28034a37b2aea4cb8c84d3f1883a151319dfb2e8c36683dea`.
All 166 production/resource pins were checked again and match. This turn
changes documentation and retains source evidence only; it does not claim a
new full test run. Stage 4 stays INCOMPLETE.

The [official IAEA version history](https://nds.iaea.org/stopping/versions)
also explicitly explains inclusion of calculated curves in the 2026-09 bulk
CSV and introduction of converted columns in 2026-01. This supports the
existing model-row exclusion, but does not resolve the separate 593-record
catalog discrepancy or establish corrections for these three experiments.
