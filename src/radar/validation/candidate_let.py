"""Compatibility wrapper for the candidate primary electronic-LET operator.

The shared implementation also accepts SRIM's analytic transport measure.
Candidate labels and the original strict candidate type boundary are preserved.
"""
from __future__ import annotations

from radar.shielding.primary_let import (
    ElectronicLetLedger as ElectronicLetLedger,
    ElectronicLetProvenance,
    ElectronicLetResult as DiagnosticElectronicLet,
    convert_primary_electronic_let,
)
from radar.validation.candidate_transport import DiagnosticPrimaryTransport

_CANDIDATE_PROVENANCE = ElectronicLetProvenance(
    model_suffix="candidate_electronic_let",
    transformation_name="candidate_electronic_let_pushforward",
    diagnostic_reason="Diagnostic candidate electronic LET: physical table accuracy is not qualified",
    outside_domain_reason="Supported primary exits outside candidate Si electronic-LET domain",
)


def convert_candidate_electronic_let(
    transport: DiagnosticPrimaryTransport, *, let_edges: tuple[float, ...],
) -> DiagnosticElectronicLet:
    """Preserve the candidate API while sharing conservative measure mapping."""
    if not isinstance(transport, DiagnosticPrimaryTransport):
        raise ValueError("A diagnostic primary transport result is required.")
    return convert_primary_electronic_let(
        transport, let_edges=let_edges, provenance=_CANDIDATE_PROVENANCE,
    )
