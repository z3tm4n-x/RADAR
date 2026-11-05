# GCR source and shielding energy-domain contract

## Evidence for the production source domain

- OST 134-1044-2007 Appendix V defines the analytical differential GCR
  spectrum in MeV/nucleon and states that its base model comes from
  GOST 25645.150 and the cited international standard.  The appendix does not
  introduce either a 10 or 300 MeV/nucleon calculation boundary.
- The supplied final draft of the successor GOST GCR model states an energy
  domain of 0.01 to 100 GeV, or GeV/nucleon for nuclei.
- The installed COSRAD 4 `GFN.txt`, `GFX.txt` and `GFXM.txt` files use a
  0.1-decade GCR source grid starting at 1 MeV/nucleon and ending at the
  printed centre 79,400 MeV/nucleon.  The next centre is the 100,000
  MeV/nucleon model boundary.

RADAR therefore uses the unrounded COSRAD-compatible final centre
`10**4.9 = 79,432.82347242821 MeV/nucleon`.  The 81-point production grid
starts at 1 MeV/nucleon and retains the control centres 1, 3, 10, 30, 100,
300, 1,000, 3,000, 10,000 and 30,000 MeV/nucleon, with eight geometric
intervals per control segment.  Its last logarithmic bin edge remains below
the 100 GeV/nucleon reference boundary and the proton aluminium table limit.
This density is a numerical integration setting, not a change to any
normative spectral coefficient.

GCR integral products are integrated only to the declared finite boundary
100,000 MeV/nucleon.  The final interval from the last COSRAD-compatible
centre to that boundary uses the local slope of the final two calculated
samples.  The former generic power-law continuation to infinity is not used.
The same finite bound is recorded in the transformation chain and preserved
when the integral is recalculated after geomagnetic penetration.

The final centre is a calculation-grid decision backed by the local COSRAD
output, not a new normative coefficient.  A normative GCR model invoked with
a first centre above 1 MeV/nucleon or a final centre below 79.4328
GeV/nucleon is explicitly marked `INCOMPLETE` at both source and mission-product
levels.  A truncated custom grid is integrated only to its actual final
sample; it is never continued numerically from 300 MeV/nucleon to 100
GeV/nucleon.  The pipeline preserves the same status in its component audit.

## Why 300 MeV/nucleon was unsafe

For a 10 g/cm2 aluminium shield, the centre of the 300 MeV/nucleon output bin
requires source energies from about 327 MeV/nucleon for helium to about
571 MeV/nucleon for nickel.  With a source grid ending at 300 MeV/nucleon the
transport interpolator therefore returned zero outside its source grid.  In
particular, the 300 MeV/nucleon silicon and iron bins disappeared although
the source model is nonzero there.

The expanded source grid restores those bins.  If a caller supplies the old
truncated grid, the zero is retained only as a diagnostic value and the
spectrum carries an explicit source-domain `INCOMPLETE` reason.

## Current HZE table limit

The available aluminium range workbook covers only 15 ions.  Its per-ion
upper energies range from 762.711864 MeV/nucleon for cobalt to 11,250
MeV/nucleon for helium.  This does not cover the GCR source tail through
79.4 GeV/nucleon.

RADAR does not extrapolate these tables.  For positive shielding it reports
the affected bins as zero and attaches explicit `INCOMPLETE` reasons for:

- source energy outside the supplied spectrum;
- output energy outside the ion's range table;
- required range above the ion table's maximum range.

Thus the expanded source spectrum is physically adequate at its upper
boundary, while the current protected HZE and derived HZE LET spectra remain
diagnostic until a verified transport/LET source covers the required energy
domain.  Zero shielding is an exact identity and bypasses the stopping-table
round trip.

## Regression checks

The automated checks cover:

- source upper-bound extension from 79.4328 to 100 GeV/nucleon for H, He, Si
  and Fe;
- source lower-bound extension by one COSRAD step from 1 to 0.7943
  MeV/nucleon for H, He, Si and Fe;
- every Stage-4 shield depth 0.01, 0.1, 1, 3 and 10 g/cm2;
- the old 300 MeV/nucleon failure for Si and Fe at 10 g/cm2;
- exact zero-shield identity for every represented HZE ion, SEP/GCR and
  peak/cumulative quantities;
- midpoint density refinement separately from upper-bound extension.

Historical domain checks on the 41-centre grid found that, within the
declared finite 100 GeV/nucleon domain, extending the final centre
to the boundary changes representative source integrals by at most
`5.3e-7`; extending below 1 MeV/nucleon changes them by at most 0.325% (He).
Across the required shield set, the corresponding maximum changes are 0.040%
at the upper edge and 0.019% at the lower edge.  The HZE comparisons are
explicitly diagnostic because their missing table tail remains quarantined
rather than being inferred from numerical agreement of the tabulated subset.

## Density convergence and support-aligned transport

The first transport implementation rejected any complete output bin whose
upper edge crossed a range-table boundary.  That made the result depend on
whether grid refinement happened to move an edge just below or above the
boundary.  On the 41-to-81-centre comparison the resulting changes reached
23.7529% for Fe energy, 1.13907% for He energy and 1.14740% for canonical HZE
LET.  Global refinement needed 1,281 centres, compared with 2,561 centres, to
reduce the Fe change to 0.279636%; it was therefore a numerical workaround,
not an appropriate production design.

### Localisation

The Fe discrepancy was concentrated at the upper range-table boundary, not
in the finite source-spectrum integral.  The Fe table ends at 803.571
MeV/nucleon and 30.5588 g/cm2.  Inverting `R_max - t` gives the largest fully
transportable output energy at each required shield depth:

| Al, g/cm2 | Fe output cutoff, MeV/nucleon |
|---:|---:|
| 0.01 | 803.410648 |
| 0.1 | 801.963621 |
| 1 | 787.493356 |
| 3 | 755.337212 |
| 10 | 636.483625 |

At 0.01 through 3 g/cm2 the 41-centre grid has an output bin centred at
740.083 MeV/nucleon with edges 636.679 and 860.281 MeV/nucleon.  The former
operator rejected that whole bin.  At 10 g/cm2 it rejected the bin centred at
547.723 MeV/nucleon with edges 471.195 and 636.679 MeV/nucleon, although the
upper edge exceeds the physical cutoff by only 0.195 MeV/nucleon.  One such
boundary bin accounted for 101% to 106% of the signed 41-to-81-centre
difference; smaller supported-bin changes partially cancelled it.

### Implemented operator

HZE transport version 3 constructs a
per-ion, per-shield output grid.  It:

- inserts exact edges at the table-energy bounds, `R_max - t` range bound and
  mapped lower/upper source-support bounds;
- emits unsupported sub-bins as zero and attaches explicit `INCOMPLETE`
  reasons; it never publishes a clipped value as an average over the original
  full bin;
- maps every fully supported output interval into its input-energy interval;
- integrates the point-sample source exactly under a piecewise-linear law and
  divides the conserved count only by the matching output-bin width;
- returns explicit `BIN_AVERAGE` ordinates and the exact edges used by those
  values, without extrapolating any table or source spectrum.

### Historical matrix, superseded

An earlier 41-to-81-centre run reported the following values across both GCR
assessment kinds and all five required shields:

| Component | Worst relative change | Controlling case |
|---|---:|---|
| H energy | 0.251209% | 10 g/cm2 |
| He energy | 0.952649% | 10 g/cm2 |
| Fe energy | 0.144354% | 10 g/cm2 |
| Canonical HZE LET | 0.951861% | 10 g/cm2 |

These 40 historical comparisons and their 25.75 s execution time are
superseded by subsequent corrections to proton partial-support handling and
the TENDL secondary-particle operator.  They do not establish convergence of
the current implementation.  With those corrections, the old 41-centre grid
gave 1.02511% for H at 0.01 g/cm2 and 1.06284% at 10 g/cm2, exceeding the
unchanged strict 1% acceptance limit.

### Current proton density checks, 2026-10-02

The current primary proton operator conserves the piecewise-linear measure
of its point-sampled incident spectrum.  The separate finite-upper-bound
source integral uses local power-law interpolation.  Convergence of the
latter did not certify the actual source measure used by proton transport:
the 41-to-81-centre change of the linear source integral alone was 1.025%.

The production grid therefore now uses 81 centres.  The following results
were measured for launch year 2026 and five-year lifetime; the source-domain
endpoints were held fixed during density refinement:

| Calculation | Density comparison | Relative integral change |
|---|---|---:|
| H source, piecewise-linear reconstruction | 81 to 161 | 0.256008% |
| H source, piecewise-linear reconstruction | 161 to 321 | 0.063980% |
| H total behind 0.01 g/cm2 Al, primary + survival + secondary | 81 to 161 | 0.255988% |
| H total behind 10 g/cm2 Al, primary + survival + secondary | 81 to 161 | 0.276528% |

The two full transport comparisons above were evaluated for mission fluence;
the four transport calculations took approximately 313 s.  The fast source
regression checks both peak and cumulative products.  For this particular
reference mission their point-sample spectra are proportional at every node
(relative variation of the ratio below `4e-16`), but this observation does not
replace an explicit run of the full two-mode, five-shield, four-component
acceptance matrix. During the subsequent full regression run on 2026-10-02,
`test_gcr_protected_energy_and_canonical_hze_let_converge_below_one_percent`
passed all 40 comparisons on the current 81-to-161-centre grids. This checks
source-to-shield transport for H, He, Fe and canonical HZE LET, not a complete
LEO/GEO/HEO orbit-and-transport reference calculation or all-ion coverage.

The isolated secondary-proton integral at 0.01 g/cm2 still changes by
7.311354% between 81 and 161 centres.  Its share of the total is only about
`8e-6`, explaining why the total can pass while this component does not.
At 10 g/cm2 the isolated secondary change is 0.561496%.  The missing
incident-energy bin quadrature remains an explicit diagnostic blocker;
convergence of the total does not resolve it or the independent TENDL
reference-frame, angular-transport and higher-generation blockers.

### Partial-bin survival regression

The survival operator previously discarded a nonzero primary bin when its
nominal centre required a range outside the stopping table, even if a
supported fraction of that bin had already been retained by CSDA transport.
This also lost particles when the reaction cross section was zero.  The
representative attenuation path is now selected inside the supported
fraction, and the original domain-incompleteness reasons are preserved.

An independent analytic regression uses `R(E)=E`, `S(E)=1` and constant
cross section.  Every valid path through `t=10 g/cm2` then has the known
survival factor
`exp[-(N_A/A_Al) * sigma_barn * 1e-24 * t]`.
Four cases cover zero and nonzero cross sections and a last bin centred at
either 100 or 120 MeV with a table ending at 100 MeV.  They verify that the
supported part survives with that factor, including exact preservation when
the cross section is zero.  This regression validates the stated numerical
boundary behaviour; it is not a new normative transport prescription or a
claim that representative-path attenuation is exact for general varying
cross sections.

An all-ion smoke check also transports all 15 ions present in both the
aluminium-range and silicon-LET tables for SEP and GCR, in both assessment
modes.  Nickel is intentionally not silently coerced: the source metadata use
rounded `A=59`, while the range table declares `A=58`.  Positive-thickness Ni
products are produced only as explicit `INCOMPLETE` diagnostics carrying the
mass-mismatch reason.  A charge-number or symbol mismatch remains a hard
error, and zero shielding remains a clean identity without a table-derived
mass blocker.

The targeted proton checks and the updated 40-comparison representative matrix
pass the strict total-integral density criterion. The isolated
secondary-particle quadrature remains unresolved. None of
these comparisons makes the provisional HZE range or quarantined silicon LET datasets
release-eligible; their provenance and coverage remain independent Stage-4
release blockers.
