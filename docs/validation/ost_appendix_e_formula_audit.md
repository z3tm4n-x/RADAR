# OST Appendix E formula audit (2026-10-02)

Source: the supplied `ОСТ 134-1044-2007 изм. 2_хор.кач..pdf`, PDF pages
122-124, printed pages 116-118. The complete pages were visually checked.
This record distinguishes the printed expressions, mathematical identities,
and RADAR's numerical/model assumptions. It is not an official errata sheet.

## Formula-to-code checks

| Formula | Check and current implementation | Status |
|---|---|---|
| E.1 | The text describes a time-propagated trajectory of approximately ten revolutions and recommends `N_rev = 100 (H + R_E)/(1000 + R_E)`. `sample_erb_orbit_grid` instead samples independent mean anomaly and RAAN. Uniform mean anomaly has equal time weights on the two-body orbit, but the independent RAAN ensemble is not the stated time trajectory. | Explicit trajectory-equivalence release blocker, separate from density convergence. |
| E.2 | Three-point Lagrange coefficients match the printed expression. The two coordinate passes recover a synthetic log-quadratic surface to floating-point precision. Positive flux is interpolated logarithmically, as recommended in the text. | Formula checked; implementation rules for zero nodes are recorded separately below. |
| E.3 | `mean = sum(phi_n * dt_n) / sum(dt_n)`. Equal mean-anomaly intervals have equal `dt`. Failed field traces previously reduced the denominator to valid samples, yielding a conditional mean. Version `ost_appendix_a_v2` retains the whole denominator. | Fixed and tested for 0, 1 and 3 valid samples out of 4. Unknown contributions still make the result `INCOMPLETE`. |
| E.4 | Above-threshold integral `Phi(>E) = integral_E^infinity phi(E') dE'`. Finite tabulation, energy interpolation and a convergent fitted high-energy power-law tail are numerical assumptions, not additional printed coefficients. | Integral definition checked; the table tail and fit remain diagnostic. |
| E.5 | Linear interpolation between solar minimum and maximum. Averaging the interpolation fraction commutes with flux averaging when the same spatial model and the declared equal-month timeline are used. | Algebra checked; the time policy is declared in metadata. |
| E.6 | Printed `phi = dPhi/dE` lacks the minus sign for the above-threshold convention in E.4. The fundamental theorem of calculus gives `phi = -dPhi(>E)/dE`. | Internal mathematical inconsistency recorded; no claim of official correction. |
| E.7-E.10 | Proton integral `a1/(a3+E)^a2 + b1 exp(-b2 E)`, relative-error least squares, and the coefficient mapping to its negative derivative agree. | Independent finite-difference checks pass. |
| E.11-E.13 | The electron integral contains `b1 exp(-b2 E^2)`. Its negative derivative contains `2 E b1 b2 exp(-b2 E^2)`. Printed E.12 lacks `2E`, while E.13.3 sets `b1' = b1 b2`. | Differential helper corrected explicitly; literal printed expression retained for diagnostic comparison. |

## Electron derivative: independent evidence

With only the Gaussian amplitude nonzero (`b1 = b2 = 1`), the printed
expression gives `exp(-E^2)` whereas the negative derivative is
`2E exp(-E^2)`. At 1 MeV the values are 0.3678794412 and 0.7357588823;
at 2 MeV they are 0.01831563889 and 0.07326255555. The equality at
0.5 MeV is accidental and cannot validate the expression.

Dimensional analysis independently requires the energy factor: `b2` has units
MeV^-2, so `b1*b2` alone has flux/MeV^2 rather than flux/MeV. The factor `2E`
restores the correct dimension.

Numerical integration over 1-2 MeV provides a third check. The literal helper
integrates to 0.13525725795, whereas `Phi(>1)-Phi(>2)` is 0.34956380228.
The corrected helper reproduces that difference. Tests also compare the
negative derivative independently at five energies for proton, mixed-electron
and pure-Gaussian electron approximations.

`OstErbElectronIntegralApproximation.differential_flux` now returns the
derivative-consistent value. `printed_differential_flux` preserves E.12/E.13
literally for diagnostics. The E.13 coefficient mapping itself is retained.
Metadata records `negative_derivative_of_e11_with_2E_gaussian_factor` and
`internal_mathematical_consistency_correction_not_confirmed_official_errata`.

These fits do not generate the production source spectra. `OstErbModel`
constructs the differential spectra from Appendix A and attaches the fitted
integral approximation afterwards. A regression forbids calls to either
species' differential fit while calculating the tabulated model, ensuring
this separation remains true.

## Known-contribution semantics for failed field traces

The retained diagnostic value is the sum of computed nonnegative contributions
divided by the complete sampled duration. It is not a completed orbit mean.
All-invalid traces produce a zero computed contribution with `INCOMPLETE`,
not proof that the radiation flux vanishes. Mean and cumulative spectra retain
the full denominator; the sampled peak remains the maximum among known
points and is also incomplete. Cumulative fluence applies the declared mission
duration once. Peak flux has no mission-duration multiplication.

The experimental AE8/AP8 implementation still uses a conditional valid-sample
mean; this audit does not qualify that profile for normative release.

## Additional interpolation rules and unresolved work

At zero tabulated ordinates a logarithm does not exist. RADAR uses Lagrange
interpolation of linear flux and clamps a negative result to zero, even in
`STRICT` mode. For nodes `(1,2,3)` with values `(0,0,1)`, literal E.2 yields
-0.125 at 1.5; the implementation returns zero. This is a declared physical
nonnegativity rule, not a verbatim requirement of E.2. Missing table support
continues to be reported as incomplete.

The RAAN-ensemble assumption must be validated against a time trajectory, and
orbit quadrature/peak search must pass convergence checks. Increasing grid
density alone cannot prove equivalence of those two models. Neither the
formula fixes nor passing helper tests close these release blockers. The
existing numerical measurements remain in `erb_orbit_sampling_convergence.md`.
