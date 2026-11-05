# Stage 4 LET product semantics

The production LET product used by the direct-ion SEE calculation is the
HZE-only sum of the individually transported ion spectra. It is identified
structurally as:

- `particle = hze`;
- aggregate particle species;
- `let_role = hze_direct_ion`;
- a source- and mode-specific LET product kind.

No proton contribution is included in this product. Proton SEE calculations
use the separate shielded proton energy spectrum and the Bendel model. This
separation prevents the same proton population from entering both the Bendel
path and the direct-ion LET convolution.

The per-ion HZE LET spectra remain available for traceability. Their sum is
checked bin by bin against the HZE direct-ion aggregate. The proton LET
histogram remains a diagnostic control. A proton+HZE histogram is also
retained only as a diagnostic product and is typed as `particle = mixed_ion`,
`let_role = mixed_proton_hze_diagnostic`; it cannot satisfy the Stage 4
assessment signature or the heavy-ion SEE input selector.

The legacy project field `include_protons_in_let` is retained for snapshot
compatibility. It now controls publication of the mixed diagnostic products
only and is exposed in code as `include_mixed_let_diagnostic`. Switching it
does not change the HZE assessment LET spectrum.

Until Stage 5 is completed, the historical optional ion-LET dose control is
routed explicitly from the mixed diagnostic product and remains excluded from
the primary dose total. This preserves the existing diagnostic calculation
without silently treating the mixed histogram as the canonical HZE spectrum.
