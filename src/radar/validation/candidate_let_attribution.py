"""Diagnostic electronic-LET tail attribution by supported EXIT energy.

This partitions the original transported source measure, not energy/LET display
histograms. It is not an uncertainty bound or a test of the Al slowing-down
path: different Al tables can move particles between the exit-energy bands.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from math import fsum, isfinite
from numbers import Real

from radar.validation.candidate_transport import DiagnosticPrimaryTransport


@dataclass(frozen=True)
class DiagnosticElectronicLetAttribution:
    """Counts/rates with the input source's time and angular normalization.

    Each matrix row is an exit-energy band; columns are inclusive LET tails.
    Outside-Si counts have unknown LET and do not enter threshold balances.
    The independent domain balance includes those counts. No source outside
    the supported primary exit measure is assigned a value here.
    """

    energy_edges: tuple[float, ...]
    thresholds: tuple[float, ...]
    band_tail_integrals: tuple[tuple[float, ...], ...]
    full_supported_tail_integrals: tuple[float, ...]
    below_energy_window_tail_integrals: tuple[float, ...]
    above_energy_window_tail_integrals: tuple[float, ...]
    outside_si_domain_integral: float
    balance_residuals: tuple[float, ...]
    supported_si_domain_integral: float
    supported_exit_integral: float
    si_domain_balance_residual: float
    normative_export_allowed: bool = field(default=False, init=False)
    physical_table_accuracy_verified: bool = field(default=False, init=False)
    error_bound_established: bool = field(default=False, init=False)
    al_path_accuracy_assessed: bool = field(default=False, init=False)


def _nonnegative(value: float) -> None:
    try:
        invalid = (isinstance(value, bool) or not isinstance(value, Real)
                   or not isfinite(value) or value < 0)
    except OverflowError:
        invalid = True
    if invalid:
        raise ValueError("Attribution values must be finite nonnegative real numbers, not bool.")


def _grid(values: tuple[float, ...], name: str, minimum: int) -> tuple[float, ...]:
    if not isinstance(values, tuple) or len(values) < minimum:
        raise ValueError(f"{name} must be an immutable tuple with at least {minimum} values.")
    for value in values:
        _nonnegative(value)
    normalized = tuple(float(value) for value in values)
    if any(left >= right for left, right in zip(normalized, normalized[1:])):
        raise ValueError(f"{name} must be strictly increasing.")
    return normalized


def _sum(values: Sequence[float]) -> float:
    for value in values:
        _nonnegative(value)
    try:
        result = fsum(values)
    except OverflowError as exc:
        raise ValueError("Attribution integral sum exceeds finite representation.") from exc
    _nonnegative(result)
    return result


def _balance(total: float, parts: Sequence[float]) -> float:
    _nonnegative(total)
    for value in parts:
        _nonnegative(value)
    try:
        residual = fsum((total, *(-value for value in parts)))
    except OverflowError as exc:
        raise ValueError("Attribution balance exceeds finite representation.") from exc
    if not isfinite(residual) or abs(residual) > 1e-10 * total:
        raise ValueError("Attribution integral balance failed.")
    return residual


def attribute_candidate_electronic_let(
    transport: DiagnosticPrimaryTransport,
    *,
    energy_edges: tuple[float, ...],
    let_thresholds: tuple[float, ...],
) -> DiagnosticElectronicLetAttribution:
    """Integrate every Si linear-segment preimage of L >= each threshold.

    Energy edges are MeV/nucleon; thresholds are MeV cm^2/mg. Rising and
    falling branches are both retained. A plateau equal to a threshold is
    wholly included. The supported exit measure has no energy point masses,
    so shared energy-band endpoints do not duplicate counts. The finite Si
    domain and exit support are intersected explicitly, without extrapolation.

    Threshold balances compare full supported Si tails with the requested
    bands and energy-window tails, not with the whole incident source. No
    time multiplication, angular factor, stopping correction or refit occurs.
    """
    if not isinstance(transport, DiagnosticPrimaryTransport):
        raise ValueError("A diagnostic primary transport result is required.")
    edges = _grid(energy_edges, "Energy edges", 2)
    thresholds = _grid(let_thresholds, "LET thresholds", 1)
    table, support = transport.table, transport.exit_support
    bands: list[list[list[float]]] = [[[] for _ in thresholds] for _ in edges[:-1]]
    full: list[list[float]] = [[] for _ in thresholds]
    below: list[list[float]] = [[] for _ in thresholds]
    above: list[list[float]] = [[] for _ in thresholds]
    outside: list[float] = []

    def count(lower: float, upper: float) -> float:
        if lower >= upper:
            return 0.0
        value = transport.integral_on_exit(lower, upper)
        _nonnegative(value)
        return value

    supported_si = 0.0
    if support is not None:
        low, high = support
        si_low, si_high = table.si_energy[0], table.si_energy[-1]
        outside.extend((count(low, min(high, si_low)), count(max(low, si_high), high)))
        supported_si = count(max(low, si_low), min(high, si_high))
        for e0, e1, l0, l1 in zip(
            table.si_energy, table.si_energy[1:],
            table.si_electronic_let, table.si_electronic_let[1:],
        ):
            if e1 <= low or e0 >= high:
                continue
            for column, threshold in enumerate(thresholds):
                if threshold > max(l0, l1):
                    continue
                left, right = max(e0, low), min(e1, high)
                if threshold > min(l0, l1):
                    # A plateau never reaches this branch. Exact endpoint
                    # roots avoid a rounding displacement at a table knot.
                    if threshold == l0:
                        root = e0
                    elif threshold == l1:
                        root = e1
                    else:
                        root = e0 + (threshold - l0) / (l1 - l0) * (e1 - e0)
                    if l1 > l0:
                        left = max(left, root)
                    else:
                        right = min(right, root)
                if left >= right:
                    continue
                full[column].append(count(left, right))
                below[column].append(count(left, min(right, edges[0])))
                above[column].append(count(max(left, edges[-1]), right))
                for row, (a, b) in enumerate(zip(edges, edges[1:])):
                    bands[row][column].append(count(max(left, a), min(right, b)))

    matrix = tuple(tuple(_sum(values) for values in row) for row in bands)
    full_counts = tuple(_sum(values) for values in full)
    below_counts = tuple(_sum(values) for values in below)
    above_counts = tuple(_sum(values) for values in above)
    outside_count = _sum(outside)
    supported = transport.ledger.supported_exit_integral
    domain_balance = _balance(supported, (supported_si, outside_count))
    balances = tuple(
        _balance(total, (below_counts[i], above_counts[i], *(row[i] for row in matrix)))
        for i, total in enumerate(full_counts)
    )
    if thresholds[0] == 0.0:
        _balance(supported_si, (full_counts[0],))
    return DiagnosticElectronicLetAttribution(
        edges, thresholds, matrix, full_counts, below_counts, above_counts,
        outside_count, balances, supported_si, supported, domain_balance,
    )
