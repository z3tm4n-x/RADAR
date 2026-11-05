# Stage 4 development snapshot

This branch publishes the accumulated RADAR implementation, regression tests,
validation tools and project protocols. It is a development snapshot, **not a
normative release**. The strict OST audit still has five blocking data findings.
Publishing the code does not qualify diagnostic stopping models or complete
the orbital acceptance matrix, dose-rate calculation or SEE implementation.

## Included and excluded evidence

Included: authored Python/PowerShell tools, tests, small attributed parser
fixtures, bundled calculation resources, and reviewed validation protocols.
The benchmark JSON files under this directory are diagnostic records, not
independent reference solutions.

Excluded: `research/`, `outputs/`, `work/`, temporary directories, private
manuscripts/reviews, downloaded publications, native executable/runtime packages
and full SRIM/COSRAD calculation campaigns. Their historical hashes and source
references remain in the protocols. A clone alone cannot replay experiments
whose external inputs are not distributed.

Historical runner defaults may refer to the original local Windows installation.
These strings are retained because downstream evidence pins the scripts' exact
bytes; they do not grant access to those files. Use each runner's explicit input
options and restore authenticated inputs before running it. Personal attachment
directory paths have been removed from the public Markdown protocols.

## Tests

```console
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check src tests
python -m mypy src
```

Tests needing excluded artifacts are marked `external_data`. When those artifacts
are missing, only the dependent tests are skipped with explicit path reasons.
Synthetic tests in the same modules still run. A present but corrupt artifact
continues to fail its integrity tests; it is not skipped. To select the portable
subset explicitly, use `pytest -m "not external_data"`. Always report skipped
external-evidence checks separately from passing tests.

## Literal bytes

Stage 4 provenance includes SHA256 of raw source/output bytes. `.gitattributes`
therefore disables line-ending conversion for pinned production Python, scripts
and native parser fixtures. Some otherwise unchanged source files acquire
newline-only Git differences when this snapshot first preserves their actual
reviewed working bytes. This is intentional; no formula changes are implied by
those differences. Readable text diffs remain enabled.

## Third-party data notices

- TENDL-2025 proton-on-Al-27 ENDF source credits A. J. Koning and is distributed
  under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The retained
  ENDF file is unchanged; the MF3/MF6 CSV files are RADAR-derived conversions with
  their transformation/provenance recorded in the integrity manifest. Source:
  [TENDL-2025](https://tendl.imperial.ac.uk/tendl_2025/tendl2025.html).
- The NIST ESTAR fixture is generated output, not redistributed program code;
  see `tests/fixtures/nist_estar_NOTICE.txt` and the
  [NIST distribution terms](https://physics.nist.gov/PhysRefData/Star/Text/download.html).
- The small SRIM parser fixture is attributed numerical program output; see
  `tests/fixtures/srim_h_al_NOTICE.txt`. No SRIM program, runtime or blanket
  redistribution licence is conveyed by this snapshot.

No vendor endorsement, independent experimental validation or normative approval
is implied by these fixtures or by successful numerical tests.
