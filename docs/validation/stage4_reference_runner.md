# Stage 4 reference spectrum runner

`radar.validation.stage4_spectra` is the long-running diagnostic runner for the
approved Stage-4 matrix. It executes the normal OST 134-1044-2007 SEP, GCR and
ERB pipelines for:

- LEO: 400/400 km, 51.6 degrees;
- GEO: 35786/35786 km, 0 degrees;
- HEO: 500/39700 km, 63.4 degrees;
- launch year 2026, five mission years and SEP exceedance probability 0.1;
- aluminium depths 0.01, 0.1, 1, 3 and 10 g/cm2;
- both `peak` and `cumulative` assessment modes.

Run the complete six-case matrix from the repository root:

```powershell
.venv\Scripts\python.exe -m radar.validation.stage4_spectra `
  --output outputs\validation\stage4-spectrum-reference.json
```

The output is deterministic for a fixed program and data release: it contains
no timestamp, elapsed time or host-dependent path. A smaller developer smoke
run can be requested without changing the fixed full-matrix definition:

```powershell
.venv\Scripts\python.exe -m radar.validation.stage4_spectra `
  --scenario LEO --mode cumulative `
  --output outputs\validation\stage4-leo-cumulative-smoke.json
```

The JSON records the complete input contract, program and model versions,
strict profile-scoped normative-data audit, and every selected on-orbit and
behind-shield spectrum. Each spectrum entry includes product kind, physical
quantity, units, species, completeness, reasons, grid extent and a trapezoidal
or bin-conservative integral according to the ordinate basis. Every positive shield depth records its
required, observed and missing product kinds. It also records typed product
signatures so a kind is not considered complete unless all required particle
groups are present: proton and HZE for SEP/GCR energy spectra, proton and
electron for ERB, and the canonical aggregate HZE direct-ion LET product
behind shielding. Protons are not included in that canonical LET product.
The LET role, product enum, particle enum and `ParticleSpecies.is_aggregate`
identify it; display labels are never parsed.

Schema `radar.stage4.spectrum-reference.v4` checks the exact multiset of shield
depths against the calculation configuration, including missing, unexpected
and duplicate depths. It also verifies each spectrum's actual shield, source,
location, mode and typed contract instead of trusting its enclosing stage.
The exact OST ion roster is checked on orbit, at every positive depth and at
zero shielding: the positive-abundance B.9 entries for SEP and H through U for
GCR. A partial ion selection remains a diagnostic calculation.

Each spectrum carries the contract trace needed to reproduce and audit the
artifact: angular and temporal bases, mission duration, shielding descriptor,
complete model reference with document/dataset references, and every structured
transformation step with its parameters.

Zero shielding is checked separately through the low-level transport functions
at 0 g/cm2. It is never inserted into `ShieldingConfig`, whose public contract
continues to require positive thicknesses. The report records exact equality,
maximum differences, tolerances, missing products and explicit incomplete
reasons for these identities. Product keys must match one-to-one, without
missing, extra or duplicate spectra. Identity equality and release readiness
are separate gates: identical incomplete spectra still cannot be released.

## Status and limits

This artifact is always marked `diagnostic_only`, `numerical_gold: false` and
`normative_export_allowed: false`. Its calculated values are not frozen by unit
tests. A traceable COSRAD version, settings and exported reference results are
still required before numerical tolerances or gold values can be established.

Current unverified or incomplete normative resources and any pipeline
incompleteness are copied into `release_gate.blockers`; they are never hidden or
replaced with synthetic spectra. By default the command succeeds when the
diagnostic calculation itself completes. CI or a controlled release check may
add `--fail-on-release-blockers` to request exit code 1 whenever blockers remain.
