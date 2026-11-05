# OST Appendix V formula audit (2026-10-02)

Source: the supplied `ОСТ 134-1044-2007 изм. 2_хор.кач..pdf`, Appendix В,
PDF pages 108-114, printed pages 102-108. These pages were visually checked,
including tables В.1-В.3. The code uses the Latin transliteration `V`.
This is a limited formula/provenance audit, not validation of all RADAR or
an official errata sheet. No numerical formula, coefficient, angular factor,
rest-energy constant or gamma guard was changed by this audit.

## Formula-to-code checks

| Formula / printed page | Code mapping and evidence | Status / limits |
|---|---|---|
| В.1 / 102 | `ost_main_gcr_differential_flux_directional_cm2_s_sr_mev`: `1e-3 (A/Z) C beta^(alpha-1) R^(-gamma) [R^1.15/(R^1.15+R0)]^Delta`. | Algebra and explicit normalization match. The section states directional per-nucleon units; no GOST m^-2/GeV conversion is applied. |
| В.2-В.3 / 102 | `gcr_rigidity_gv_from_mev_per_nucleon`, `gcr_beta_from_mev_per_nucleon`: relativistic square root and A/Z match. | Code uses 938.2720813 MeV instead of the printed rounded 938 MeV; see below. |
| В.4 / 103 | `ost_delta_exponent`: `5.5 + 1.13 M x exp(-x)`, `x=beta R/R0`. | Algebra matches. |
| В.5-В.6 / 103 | `ost_gcr_drift_m`: parity `(-1)^(n-1)` and S before/after reversal match. | Code additionally clamps `(W-Wmin)/(Wmax-Wmin)` to [0,1]. Calendar discretization of the maximum is declared in the model metadata. |
| В.7 / 103 | `ost_modulation_potential_r0_gv`: `0.28 Wlag^0.09 + 4e-4 Wlag^1.45`. | Coefficients and powers match. |
| В.8-В.9 / 103,106 | `ost_gcr_delay_months`: `Tplus=16.5 R^-0.10`, `Tminus=7.5 R^-0.45`, tau parity `(-1)^n`, denominator Wmax. The timeline supplies W(t-16 months). | Algebra matches on the applicable nonnegative tau-base domain. Code clamps negative tau bases to zero; that is an implementation rule. |
| В.10-В.12 / 106 | `ost_nek_differential_flux_directional_cm2_s_sr_mev`: `ac exp(bc W) E^[-(ag-bg W)]`. | Algebra matches; helium bc remains pending COSRAD comparison. |
| В.13 / 107 | `ost_akl_differential_flux_directional_cm2_s_sr_mev`: `C exp(-0.08 t^2.5) exp(-gamma E)`, `gamma=0.15-4.5e-5 t^4.6`. | Algebra matches while gamma>0 under the current conventions. Units/time interpretation and the gamma<=0 suppression require confirmation. |
| В.14 / 108 | Printed interplanetary modulation factor `1-Rsun/Rb`, with Rb=90 au. | Not implemented in this Earth-orbit source branch. No general interplanetary qualification is claimed. |
| Tables В.1-В.3 / 104-107 | `parameters.py`: main-ion coefficients, NEK charge-dependent coefficients, and AKL He/C/N/O/Ne/Ar amplitudes. | Values match the supplied printing, including its unusual heavy-element mass entries. No mass-table correction was inferred. |

## Unresolved AKL conventions and release gate

Before В.13, and in the description of table В.3, the printed units are
`cm^-2 s^-1 MeV^-1`: they omit both sr^-1 and the per-nucleon energy basis.
В.2.1 explicitly specifies directional per-nucleon units for the main GCR
component, but does not unambiguously establish those units for В.13.
The table В.1 heading also omits sr^-1, so the printing is internally
inconsistent and does not justify silently removing or adding a 4*pi factor.
The text calls t time since solar-cycle start without explicitly naming a unit.

The retained implementation assumes directional input in
`cm^-2 sr^-1 s^-1 (MeV/nucleon)^-1`, energy in MeV/nucleon and t in years,
then multiplies by 4*pi. These assumptions are now recorded in the source
transformation metadata with `pending_source_or_reference_comparison`.
The AKL and total spectra for the six ions with nonzero table В.3 amplitudes
are `INCOMPLETE`. Their main and NEK components do not acquire this blocker;
H and other ions without an AKL contribution do not acquire it either.
The same localization applies to the experimental GOST branch using OST AKL,
while its pre-existing draft-source blocker is preserved.

The implementation sets AKL to zero when gamma<=0. This condition is absent
from the supplied В.13. With t interpreted in years, gamma crosses zero at
5.832344995392128 years. At t=6 and E=10, the literal He expression gives
1.2768720622387116e-6 before angular conversion, while the guarded code gives
zero. Removing the guard without establishing the applicability domain would
allow exponential growth with energy. The guard is therefore retained and,
when active, contributes a second explicit `INCOMPLETE` reason to AKL and
total. A zero guarded spectrum is not a confirmed absence of AKL.

Provenance versions are `ost_134_1044_2007_change_2_appendix_v_v3` and
`gost_gcr_draft_main_plus_ost_appendix_v_v2`; the numerical arrays are unchanged.

## Transparent remaining numerical assumptions

The common particle constant is 938.2720813 MeV, while the supplied В.2-В.3
text specifies 938 MeV. Across 1-1000 MeV/nucleon this changes R by about
+0.0095 to +0.0145 percent and beta by about -0.0046 to -0.0145 percent.
This review records the choice without changing the shared constant or
claiming a correction to the standard. Its influence on final fluxes has
not been bounded here.

The printed `bc(He)=6.60e-2` remains selected and its existing
`pending_cosrad_comparison` status is preserved. The alternative `6.60e-3`
has not been selected. The AKL checks are independent of that NEK question.

## Targeted verification

Source-level regressions cover all six AKL ions at t=2 and t=6, preserving
main/NEK statuses, retaining the helium NEK warning, propagating AKL reasons
to total, and confirming that H/Fe/U acquire no irrelevant AKL blocker.
Existing numerical source tests continue to check the same calculated values.
These checks validate the status/provenance change, not the unresolved
normalization, time convention or post-gamma-zero physics.

Mission aggregation previously inherited only the first time bin's status.
The version-3 aggregation operation now unions reasons and document/data
references from all bins, including a later zero or non-maximum contribution,
for all six mean/peak/cumulative differential/integral products. Its actual
time-bin count is recorded. A real carbon AKL/total t=2 to t=6 regression
confirms that the later gamma-guard warning survives aggregation. Synthetic
tests also verify order independence, distinct reasons and unknown legacy
completeness. The numerical sums, maxima and time weights are unchanged.
Legacy inputs without a first-bin model reference retain that missing
reference and remain ineligible for normative release.
