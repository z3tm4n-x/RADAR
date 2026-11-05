# SEP source and shielding energy-domain contract

## Evidence for the computational source domain

OST 134-1044-2007 Appendix B defines analytical proton and heavy-ion SEP
spectra but does not introduce the former RADAR cut-offs at 1,000 MeV for
protons or 500 MeV/nucleon for ions.  Those cut-offs were implementation
choices and removed non-zero particles before transport through the thicker
Stage-4 aluminium shields.

The installed COSRAD 4 `SFN.txt` result uses logarithmic centres from
1 MeV/nucleon through the printed 12,600 MeV/nucleon centre.  The retained
local validation run has 42 numerical rows at 0.1 decade spacing; its exact
last unrounded centre is `10**4.1 = 12589.254117941662 MeV/nucleon`.  The
source artifact used during the audit had:

- size: 14,073 bytes;
- SHA-256:
  `89e27e63de8ec6b554e5e42448a827a949722a2518a290a688d8fb333d91c6e0`;
- orbit: 36,000/36,000 km, inclination 0 degrees;
- mission: start in cycle year 1, duration 10 years, SEP probability 0.1.

This COSRAD output is implementation evidence for the evaluation domain, not
a normative coefficient table and not a numerical gold result.

RADAR therefore evaluates both proton and HZE analytical source models over
the same 1..`10**4.1` domain.  The production density is 0.05 decade for
protons (83 centres) and 0.025 decade for HZE (165 centres).  The different
densities are deliberate: the proton secondary-production operator is more
expensive, while the HZE CSDA and LET transformations required the finer grid
to satisfy the Stage-4 convergence criterion after 10 g/cm2 aluminium.

Any OST SEP source model whose first centre is above 1 MeV or whose last
centre is below `10**4.1` MeV is explicitly `INCOMPLETE`; a calculation may
remain available diagnostically, but it cannot pass the release gate.

## Independent convergence dimensions

The tests keep three questions separate:

1. endpoint coverage: the production domain reaches the COSRAD-observed last
   centre and a truncated custom source is rejected for release;
2. grid-density convergence: geometric midpoint refinement changes conserved
   protected energy and canonical HZE-only LET integrals by less than 1%;
3. transport support: a mapped input outside a stopping/range/TENDL table is
   recorded as `INCOMPLETE`, never silently converted to a valid zero.

The density check covers cumulative fluence and peak flux, aluminium depths
0.01, 0.1, 1, 3 and 10 g/cm2, representative p/He/Fe energy spectra, and the
aggregate HZE-only direct-ion LET spectrum.  Peak quantities remain per
second and are never multiplied by mission duration.

On 2026-10-02 the full regression run passed the current 40-comparison
protected-density matrix (both modes, five depths, p/He/Fe energy and
canonical HZE LET), as well as the separate p/He/Fe refinement checks behind
10 g/cm2. These are source-to-shield numerical checks for the represented
ions, not COSRAD agreement, complete all-ion transport coverage, or a
LEO/GEO/HEO orbit-and-transport reference validation. Provisional range data
and quarantined LET data retain their release blockers.

## Current release limits

Extending the analytical source domain does not repair the transport data.
The current HZE aluminium range tables cover only 15 ions and have
ion-dependent upper energies.  The HZE silicon LET table is quarantined due to
the confirmed discontinuity already present in its source workbook.  RADAR
does not extrapolate or smooth either gap.  Protected HZE energy and LET
results therefore remain diagnostic and carry their explicit table-domain and
data-eligibility reasons until verified replacements are regenerated from an
authoritative source.
