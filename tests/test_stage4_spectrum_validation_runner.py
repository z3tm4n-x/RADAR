from __future__ import annotations

from dataclasses import replace
import json

import pytest

from radar.core.products import SpectrumProduct
from radar.core.profiles import MethodologyProfile
from radar.core.spectra import (
    ModelReference,
    ParticleSpecies,
    ShieldingDescriptor,
    Spectrum1D,
    TransformationStep,
)
from radar.core.types import (
    AngularBasis,
    AssessmentMode,
    LetSpectrumRole,
    Particle,
    RadiationProductKind,
    RadiationSource,
    SpectrumAxisKind,
    SpectrumCompleteness,
    SpectrumLocation,
    SpectrumOrdinateBasis,
    SpectrumQuantity,
    TemporalBasis,
)
from radar.core.units import Unit
from radar.data.audit import DataAuditReport, DataAuditScope
from radar.gcr.parameters import GCR_ELEMENT_SYMBOLS
from radar.sep.model import ost_sep_required_hze_ions
from radar.validation.stage4_spectra import (
    STAGE4_REFERENCE_REPORT_SCHEMA,
    STAGE4_REFERENCE_SCENARIOS,
    STAGE4_REFERENCE_SHIELD_THICKNESSES_G_CM2,
    Stage4CaseExecution,
    Stage4ShieldExecution,
    Stage4SourceExecution,
    Stage4ZeroShieldExecution,
    run_stage4_reference_validation,
    stage4_reference_validation_json,
)

_MISSION_DURATION_S = 5.0 * 365.25 * 24.0 * 60.0 * 60.0


def _fake_product(
    *,
    source: RadiationSource,
    kind: RadiationProductKind,
    thickness_g_cm2: float | None,
    particle: Particle = Particle.PROTON,
    ion: tuple[int, str] | None = None,
    let_spectrum: bool = False,
    aggregate_let: bool = True,
) -> SpectrumProduct:
    model = f"fake_{source.value}"
    let_role = None
    if let_spectrum:
        particle = Particle.HZE
        species = (
            ParticleSpecies(symbol="total", is_aggregate=True)
            if aggregate_let
            else ParticleSpecies(symbol="He", atomic_number=2, mass_number=4)
        )
        x = (1.0, 2.0, 4.0)
        x_unit = Unit.LET
        y_unit = Unit.DIFFERENTIAL_LET_FLUENCE
        quantity = SpectrumQuantity.LET_DIFFERENTIAL_FLUENCE
        axis_kind = SpectrumAxisKind.LINEAR_ENERGY_TRANSFER
        let_role = (
            LetSpectrumRole.HZE_DIRECT_ION
            if aggregate_let
            else LetSpectrumRole.HZE_SPECIES
        )
    elif particle is Particle.PROTON:
        species = ParticleSpecies(symbol="H", atomic_number=1, mass_number=1)
        x = (10.0, 20.0, 40.0)
        x_unit = Unit.MEV
        y_unit = Unit.DIFFERENTIAL_FLUENCE
        quantity = SpectrumQuantity.DIFFERENTIAL_FLUENCE
        axis_kind = SpectrumAxisKind.KINETIC_ENERGY
    elif particle is Particle.ELECTRON:
        species = ParticleSpecies(symbol="e-")
        x = (10.0, 20.0, 40.0)
        x_unit = Unit.MEV
        y_unit = Unit.DIFFERENTIAL_FLUENCE
        quantity = SpectrumQuantity.DIFFERENTIAL_FLUENCE
        axis_kind = SpectrumAxisKind.KINETIC_ENERGY
    else:
        atomic_number, symbol = (2, "He") if ion is None else ion
        species = ParticleSpecies(
            symbol=symbol,
            atomic_number=atomic_number,
            mass_number=max(2, 2 * atomic_number),
        )
        x = (10.0, 20.0, 40.0)
        x_unit = Unit.MEV_PER_NUCLEON
        y_unit = Unit.DIFFERENTIAL_FLUENCE_PER_NUCLEON_ENERGY
        quantity = SpectrumQuantity.DIFFERENTIAL_FLUENCE
        axis_kind = SpectrumAxisKind.KINETIC_ENERGY_PER_NUCLEON
    location = (
        SpectrumLocation.ON_ORBIT
        if thickness_g_cm2 is None
        else SpectrumLocation.BEHIND_SHIELD
    )
    shielding = (
        None
        if thickness_g_cm2 is None
        else ShieldingDescriptor(
            material="aluminium",
            areal_density_g_cm2=thickness_g_cm2,
        )
    )
    return SpectrumProduct(
        kind=kind,
        spectrum=Spectrum1D(
            x=x,
            y=(3.0, 2.0, 1.0),
            x_unit=x_unit,
            y_unit=y_unit,
            quantity=quantity,
            particle=particle,
            source=source,
            model=model,
            axis_kind=axis_kind,
            angular_basis=AngularBasis.OMNIDIRECTIONAL_4PI,
            temporal_basis=TemporalBasis.MISSION_TOTAL,
            location=location,
            species=species,
            shielding=shielding,
            assessment_mode=AssessmentMode.CUMULATIVE,
            mission_duration_s=_MISSION_DURATION_S,
            model_reference=ModelReference(
                name=model,
                version="1",
                document_refs=("fake standard",),
                dataset_refs=("fake.csv",),
            ),
            transformation_chain=(TransformationStep(name="fake", version="1"),),
            completeness=SpectrumCompleteness.COMPLETE,
            let_role=let_role,
        ),
    )


def _fake_source_execution(source: RadiationSource) -> Stage4SourceExecution:
    energy_kind = RadiationProductKind.MISSION_FLUENCE
    if source is RadiationSource.ERB:
        energy_species: tuple[tuple[Particle, tuple[int, str] | None], ...] = (
            (Particle.PROTON, None),
            (Particle.ELECTRON, None),
        )
    elif source is RadiationSource.SEP:
        energy_species = (
            (Particle.PROTON, None),
            *((Particle.HZE, ion) for ion in ost_sep_required_hze_ions()),
        )
    else:
        energy_species = tuple(
            (
                Particle.PROTON if atomic_number == 1 else Particle.HZE,
                (atomic_number, symbol),
            )
            for atomic_number, symbol in enumerate(GCR_ELEMENT_SYMBOLS, start=1)
        )
    on_orbit = tuple(
        _fake_product(
            source=source,
            kind=energy_kind,
            thickness_g_cm2=None,
            particle=particle,
            ion=ion,
        )
        for particle, ion in energy_species
    )
    shields = tuple(
        Stage4ShieldExecution(
            thickness_g_cm2=thickness,
            products=(
                *(
                    _fake_product(
                        source=source,
                        kind=energy_kind,
                        thickness_g_cm2=thickness,
                        particle=particle,
                        ion=ion,
                    )
                    for particle, ion in energy_species
                ),
                *(
                    ()
                    if source is RadiationSource.ERB
                    else (
                        _fake_product(
                            source=source,
                            kind=RadiationProductKind.MISSION_LET_FLUENCE,
                            thickness_g_cm2=thickness,
                            let_spectrum=True,
                        ),
                    )
                ),
            ),
        )
        for thickness in STAGE4_REFERENCE_SHIELD_THICKNESSES_G_CM2
    )
    return Stage4SourceExecution(
        source=source,
        model_name=f"fake_{source.value}",
        model_version="1",
        model_document="fake standard",
        on_orbit_products=on_orbit,
        shielded_by_thickness=shields,
        zero_shield=Stage4ZeroShieldExecution(
            products=(
                *(
                    _fake_product(
                        source=source,
                        kind=energy_kind,
                        thickness_g_cm2=0.0,
                        particle=particle,
                        ion=ion,
                    )
                    for particle, ion in energy_species
                ),
            )
        ),
    )


def _fake_execution() -> Stage4CaseExecution:
    return Stage4CaseExecution(
        sources=tuple(
            _fake_source_execution(source)
            for source in (RadiationSource.SEP, RadiationSource.GCR, RadiationSource.ERB)
        )
    )


def _clean_ost_audit() -> DataAuditReport:
    return DataAuditReport(
        release_id="test-release",
        scope=DataAuditScope.for_profile(MethodologyProfile.OST_134_1044_2007),
        audited_paths=(),
        findings=(),
    )


def test_stage4_reference_runner_has_deterministic_diagnostic_contract() -> None:
    def execute(config: object) -> Stage4CaseExecution:
        assert config is not None
        return _fake_execution()

    kwargs = {
        "scenarios": STAGE4_REFERENCE_SCENARIOS[:1],
        "modes": (AssessmentMode.CUMULATIVE,),
        "case_executor": execute,
        "audit_provider": _clean_ost_audit,
    }
    report = run_stage4_reference_validation(**kwargs)
    rendered = stage4_reference_validation_json(report)
    assert rendered == stage4_reference_validation_json(
        run_stage4_reference_validation(**kwargs)
    )

    payload = json.loads(rendered)
    assert payload["schema"] == STAGE4_REFERENCE_REPORT_SCHEMA
    assert payload["intended_use"] == {
        "diagnostic_only": True,
        "normative_export_allowed": False,
        "numerical_gold": False,
    }
    assert payload["matrix"] == {
        "case_count": 1,
        "full_reference_matrix_requested": False,
        "mode_count": 1,
        "scenario_count": 1,
    }

    case = payload["cases"][0]
    assert case["case_id"] == "leo_cumulative"
    assert case["inputs"]["methodology_profile"] == "ost_134_1044_2007"
    assert case["inputs"]["shielding"]["thicknesses_g_cm2"] == [
        0.01,
        0.1,
        1.0,
        3.0,
        10.0,
    ]
    assert case["complete"] is True
    assert case["zero_shield_identities_pass"] is True
    assert case["zero_shields_release_ready"] is True

    for source in case["sources"]:
        assert len(source["behind_aluminium_shield"]) == 5
        assert source["positive_shield_thicknesses"] == {
            "complete": True,
            "duplicates_g_cm2": [],
            "expected_g_cm2": [0.01, 0.1, 1.0, 3.0, 10.0],
            "extra_g_cm2": [],
            "incomplete_reasons": [],
            "missing_g_cm2": [],
            "observed_g_cm2": [0.01, 0.1, 1.0, 3.0, 10.0],
        }
        assert source["zero_shield_identity"]["identity_passed"] is True
        assert source["zero_shield_identity"]["release_ready"] is True
        zero_multiset = source["zero_shield_identity"]["product_multiset"]
        assert zero_multiset["complete"] is True
        assert zero_multiset["expected_count"] == zero_multiset["observed_count"]
        assert zero_multiset["missing"] == []
        assert zero_multiset["extra"] == []
        assert zero_multiset["duplicates"] == []
        assert source["zero_shield_identity"]["public_shielding_config_contains_zero"] is False
        proton_spectrum = next(
            spectrum
            for spectrum in source["on_orbit"]["spectra"]
            if spectrum["particle"] == "proton"
        )
        assert proton_spectrum["y_unit"] == "cm^-2 MeV^-1"
        assert proton_spectrum["integral"] == {
            "method": "trapezoidal_over_reported_grid",
            "unit": "cm^-2",
            "value": 55.0,
        }
        assert proton_spectrum["angular_basis"] == "omnidirectional_4pi"
        assert proton_spectrum["mission_duration_s"] == _MISSION_DURATION_S
        assert proton_spectrum["model_reference"]["dataset_refs"] == ["fake.csv"]
        assert proton_spectrum["transformation_chain"] == [
            {"name": "fake", "parameters": [], "version": "1"}
        ]
        shield_stage = source["behind_aluminium_shield"][0]
        assert shield_stage["spectra"][0]["shielding"] == {
            "areal_density_g_cm2": 0.01,
            "areal_density_unit": "g/cm^2",
            "material": "aluminium",
        }
        expected_signature_count = 2 if source["source"] == "erb" else 3
        assert len(shield_stage["required_product_signatures"]) == expected_signature_count
        expected_roster_size = {
            "sep": len(ost_sep_required_hze_ions()),
            "gcr": len(GCR_ELEMENT_SYMBOLS),
            "erb": 0,
        }[source["source"]]
        roster = source["on_orbit"]["normative_ion_roster"]
        assert roster["complete"] is True
        assert len(roster["required"]) == expected_roster_size
        assert len(roster["observed"]) == expected_roster_size

    release_gate = payload["release_gate"]
    assert release_gate["calculation_products_release_ready"] is True
    assert release_gate["report_eligible_for_normative_export"] is False
    assert {item["code"] for item in release_gate["blockers"]} == {
        "diagnostic_not_numerical_gold"
    }


@pytest.mark.parametrize(
    "target_source",
    (RadiationSource.SEP, RadiationSource.GCR, RadiationSource.ERB),
)
@pytest.mark.parametrize(
    ("mutation", "expected_missing", "expected_extra", "expected_duplicates"),
    (
        ("missing", [0.01], [], []),
        ("extra", [], [20.0], []),
        ("duplicate", [], [0.01], [0.01]),
    ),
)
def test_stage4_runner_requires_exact_positive_shield_thickness_multiset(
    target_source: RadiationSource,
    mutation: str,
    expected_missing: list[float],
    expected_extra: list[float],
    expected_duplicates: list[float],
) -> None:
    execution = _fake_execution()
    source_index = next(
        index
        for index, source in enumerate(execution.sources)
        if source.source is target_source
    )
    source_execution = execution.sources[source_index]
    shields = source_execution.shielded_by_thickness
    if mutation == "missing":
        changed_shields = shields[1:]
    elif mutation == "extra":
        changed_shields = (
            *shields,
            replace(shields[-1], thickness_g_cm2=20.0),
        )
    else:
        assert mutation == "duplicate"
        changed_shields = (*shields, shields[0])
    changed_sources = list(execution.sources)
    changed_sources[source_index] = replace(
        source_execution,
        shielded_by_thickness=changed_shields,
    )

    report = run_stage4_reference_validation(
        scenarios=STAGE4_REFERENCE_SCENARIOS[:1],
        modes=(AssessmentMode.CUMULATIVE,),
        case_executor=lambda _config: replace(
            execution,
            sources=tuple(changed_sources),
        ),
        audit_provider=_clean_ost_audit,
    )

    case = report["cases"][0]
    source_summary = next(
        source for source in case["sources"] if source["source"] == target_source.value
    )
    thicknesses = source_summary["positive_shield_thicknesses"]
    assert thicknesses["complete"] is False
    assert thicknesses["missing_g_cm2"] == expected_missing
    assert thicknesses["extra_g_cm2"] == expected_extra
    assert thicknesses["duplicates_g_cm2"] == expected_duplicates
    assert source_summary["complete"] is False
    assert case["complete"] is False
    assert report["release_gate"]["calculation_products_release_ready"] is False
    blocker = next(
        item
        for item in report["release_gate"]["blockers"]
        if item["code"] == "positive_shield_thickness_multiset_mismatch"
        and item["source"] == target_source.value
    )
    assert blocker["missing_g_cm2"] == expected_missing
    assert blocker["extra_g_cm2"] == expected_extra
    assert blocker["duplicates_g_cm2"] == expected_duplicates


@pytest.mark.parametrize(
    "target_source",
    (RadiationSource.SEP, RadiationSource.GCR, RadiationSource.ERB),
)
@pytest.mark.parametrize("mutation", ("duplicate", "extra"))
def test_zero_identity_requires_exact_product_key_multiset(
    target_source: RadiationSource,
    mutation: str,
) -> None:
    execution = _fake_execution()
    source_index = next(
        index
        for index, source in enumerate(execution.sources)
        if source.source is target_source
    )
    source_execution = execution.sources[source_index]
    zero_products = source_execution.zero_shield.products
    if mutation == "duplicate":
        changed_zero_products = (*zero_products, zero_products[0])
    else:
        assert mutation == "extra"
        changed_zero_products = (
            *zero_products,
            _fake_product(
                source=target_source,
                kind=RadiationProductKind.MISSION_LET_FLUENCE,
                thickness_g_cm2=0.0,
                let_spectrum=True,
            ),
        )
    changed_sources = list(execution.sources)
    changed_sources[source_index] = replace(
        source_execution,
        zero_shield=replace(
            source_execution.zero_shield,
            products=changed_zero_products,
        ),
    )

    report = run_stage4_reference_validation(
        scenarios=STAGE4_REFERENCE_SCENARIOS[:1],
        modes=(AssessmentMode.CUMULATIVE,),
        case_executor=lambda _config: replace(
            execution,
            sources=tuple(changed_sources),
        ),
        audit_provider=_clean_ost_audit,
    )

    case = report["cases"][0]
    source_summary = next(
        source for source in case["sources"] if source["source"] == target_source.value
    )
    zero = source_summary["zero_shield_identity"]
    product_multiset = zero["product_multiset"]
    assert product_multiset["complete"] is False
    assert product_multiset["cardinality_matches"] is False
    assert product_multiset["key_multiset_matches"] is False
    assert product_multiset["observed_count"] == product_multiset["expected_count"] + 1
    assert len(product_multiset["extra"]) == 1
    assert len(product_multiset["duplicates"]) == (1 if mutation == "duplicate" else 0)
    assert zero["identity_passed"] is False
    assert zero["release_ready"] is False
    assert case["zero_shield_identities_pass"] is False
    assert case["zero_shields_release_ready"] is False
    assert report["release_gate"]["calculation_products_release_ready"] is False
    blocker = next(
        item
        for item in report["release_gate"]["blockers"]
        if item["code"] == "zero_shield_product_multiset_mismatch"
        and item["source"] == target_source.value
    )
    assert blocker["observed_count"] == blocker["expected_count"] + 1
    assert len(blocker["extra_products"]) == 1
    assert len(blocker["duplicate_products"]) == (
        1 if mutation == "duplicate" else 0
    )


def test_stage4_runner_reports_missing_normative_ion_at_every_stage() -> None:
    execution = _fake_execution()
    sep = execution.sources[0]

    def without_ni(
        products: tuple[SpectrumProduct, ...],
    ) -> tuple[SpectrumProduct, ...]:
        return tuple(
            product
            for product in products
            if not (
                product.spectrum.species is not None
                and product.spectrum.species.atomic_number == 28
                and product.spectrum.species.symbol == "Ni"
                and not product.spectrum.species.is_aggregate
            )
        )

    incomplete_sep = replace(
        sep,
        on_orbit_products=without_ni(sep.on_orbit_products),
        shielded_by_thickness=tuple(
            replace(shield, products=without_ni(shield.products))
            for shield in sep.shielded_by_thickness
        ),
        zero_shield=replace(
            sep.zero_shield,
            products=without_ni(sep.zero_shield.products),
        ),
    )
    report = run_stage4_reference_validation(
        scenarios=STAGE4_REFERENCE_SCENARIOS[:1],
        modes=(AssessmentMode.CUMULATIVE,),
        case_executor=lambda _config: replace(
            execution,
            sources=(incomplete_sep, *execution.sources[1:]),
        ),
        audit_provider=_clean_ost_audit,
    )

    case = report["cases"][0]
    sep_summary = case["sources"][0]
    expected_missing = [{"atomic_number": 28, "symbol": "Ni"}]
    assert sep_summary["complete"] is False
    assert sep_summary["on_orbit"]["normative_ion_roster"]["missing"] == expected_missing
    assert all(
        shield["normative_ion_roster"]["missing"] == expected_missing
        and shield["complete"] is False
        for shield in sep_summary["behind_aluminium_shield"]
    )
    zero = sep_summary["zero_shield_identity"]
    assert zero["identity_passed"] is True
    assert zero["release_ready"] is False
    assert zero["normative_ion_roster"]["missing"] == expected_missing

    roster_blockers = [
        blocker
        for blocker in report["release_gate"]["blockers"]
        if blocker["code"] == "normative_ion_roster_incomplete"
        and blocker["source"] == "sep"
    ]
    assert len(roster_blockers) == 7
    assert {blocker["stage"] for blocker in roster_blockers} == {
        "on_orbit",
        "behind_aluminium_shield",
        "zero_shield",
    }
    assert all(blocker["missing_ions"] == expected_missing for blocker in roster_blockers)


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    (
        ("wrong_depth", "spectrum aluminium depth does not match its reported shield depth"),
        ("wrong_source", "spectrum source does not match its reported source stage"),
        ("wrong_location", "spectrum location does not match its reported stage"),
        ("duplicate_proton", "spectral stage contains duplicate product identities"),
        ("missing_contract", "angular_basis is required"),
    ),
)
def test_stage4_runner_checks_actual_products_inside_valid_depth_matrix(
    mutation: str,
    expected_reason: str,
) -> None:
    execution = _fake_execution()
    sep = execution.sources[0]
    shield = sep.shielded_by_thickness[0]
    product = shield.products[0]
    spectrum = product.spectrum
    if mutation == "wrong_depth":
        assert spectrum.shielding is not None
        changed_spectrum = replace(
            spectrum, shielding=replace(spectrum.shielding, areal_density_g_cm2=10.0)
        )
    elif mutation == "wrong_source":
        changed_spectrum = replace(spectrum, source=RadiationSource.GCR)
    elif mutation == "wrong_location":
        changed_spectrum = replace(spectrum, location=SpectrumLocation.ON_ORBIT, shielding=None)
    elif mutation == "missing_contract":
        changed_spectrum = replace(spectrum, angular_basis=None)
    else:
        assert mutation == "duplicate_proton"
        changed_spectrum = spectrum
    changed_products = (replace(product, spectrum=changed_spectrum), *shield.products[1:])
    if mutation == "duplicate_proton":
        changed_products = (*changed_products, product)
    changed_sep = replace(
        sep,
        shielded_by_thickness=(replace(shield, products=changed_products),
                               *sep.shielded_by_thickness[1:]),
    )
    report = run_stage4_reference_validation(
        scenarios=STAGE4_REFERENCE_SCENARIOS[:1],
        modes=(AssessmentMode.CUMULATIVE,),
        case_executor=lambda _config: replace(execution, sources=(changed_sep, *execution.sources[1:])),
        audit_provider=_clean_ost_audit,
    )
    source = report["cases"][0]["sources"][0]
    assert source["positive_shield_thicknesses"]["complete"] is True
    assert source["behind_aluminium_shield"][0]["complete"] is False
    assert any(expected_reason in reason
               for reason in source["behind_aluminium_shield"][0]["incomplete_reasons"])
    assert report["release_gate"]["calculation_products_release_ready"] is False


@pytest.mark.parametrize("empty_field", ("scenarios", "modes"))
def test_stage4_runner_rejects_empty_validation_matrix(empty_field: str) -> None:
    with pytest.raises(ValueError, match="at least one scenario and mode"):
        run_stage4_reference_validation(
            scenarios=() if empty_field == "scenarios" else STAGE4_REFERENCE_SCENARIOS[:1],
            modes=() if empty_field == "modes" else (AssessmentMode.CUMULATIVE,),
            case_executor=lambda _config: _fake_execution(),
            audit_provider=_clean_ost_audit,
        )


def test_stage4_reference_runner_preserves_explicit_incomplete_reason() -> None:
    execution = _fake_execution()
    sep = execution.sources[0]
    first_shield = replace(
        sep.shielded_by_thickness[0],
        incomplete_reasons=("confirmed table is unavailable",),
    )
    incomplete_execution = replace(
        execution,
        sources=(
            replace(
                sep,
                shielded_by_thickness=(first_shield, *sep.shielded_by_thickness[1:]),
            ),
            *execution.sources[1:],
        ),
    )

    report = run_stage4_reference_validation(
        scenarios=STAGE4_REFERENCE_SCENARIOS[:1],
        modes=(AssessmentMode.CUMULATIVE,),
        case_executor=lambda _config: incomplete_execution,
        audit_provider=_clean_ost_audit,
    )
    payload = json.loads(stage4_reference_validation_json(report))

    sep_summary = payload["cases"][0]["sources"][0]
    assert sep_summary["complete"] is False
    assert sep_summary["behind_aluminium_shield"][0]["incomplete_reasons"] == [
        "confirmed table is unavailable"
    ]
    assert payload["release_gate"]["calculation_products_release_ready"] is False
    assert "incomplete_spectral_products" in {
        item["code"] for item in payload["release_gate"]["blockers"]
    }


def test_zero_identity_separates_inherited_blocker_from_identity_failure() -> None:
    execution = _fake_execution()
    sep = execution.sources[0]
    reason = "source coefficient pending"

    source_product = sep.on_orbit_products[0]
    zero_product = sep.zero_shield.products[0]
    incomplete_source = replace(
        source_product,
        spectrum=replace(
            source_product.spectrum,
            completeness=SpectrumCompleteness.INCOMPLETE,
            incomplete_reasons=(reason,),
        ),
    )
    incomplete_zero = replace(
        zero_product,
        spectrum=replace(
            zero_product.spectrum,
            completeness=SpectrumCompleteness.INCOMPLETE,
            incomplete_reasons=(reason,),
        ),
    )
    incomplete_sep = replace(
        sep,
        on_orbit_products=(incomplete_source, *sep.on_orbit_products[1:]),
        zero_shield=replace(
            sep.zero_shield,
            products=(incomplete_zero, *sep.zero_shield.products[1:]),
            incomplete_reasons=(reason,),
        ),
    )
    report = run_stage4_reference_validation(
        scenarios=STAGE4_REFERENCE_SCENARIOS[:1],
        modes=(AssessmentMode.CUMULATIVE,),
        case_executor=lambda _config: replace(
            execution,
            sources=(incomplete_sep, *execution.sources[1:]),
        ),
        audit_provider=_clean_ost_audit,
    )

    zero_summary = report["cases"][0]["sources"][0]["zero_shield_identity"]
    assert zero_summary["identity_passed"] is True
    assert zero_summary["release_ready"] is False
    assert zero_summary["incomplete_reasons"] == [reason]
    assert "zero_shield_identity_failed" not in {
        item["code"] for item in report["release_gate"]["blockers"]
    }
    assert "zero_shield_release_not_ready" in {
        item["code"] for item in report["release_gate"]["blockers"]
    }


def test_zero_identity_stage_reason_blocks_release_without_identity_failure() -> None:
    execution = _fake_execution()
    sep = execution.sources[0]
    reason = "zero-shield diagnostic table pending"
    changed_sep = replace(
        sep,
        zero_shield=replace(
            sep.zero_shield,
            incomplete_reasons=(reason,),
        ),
    )
    report = run_stage4_reference_validation(
        scenarios=STAGE4_REFERENCE_SCENARIOS[:1],
        modes=(AssessmentMode.CUMULATIVE,),
        case_executor=lambda _config: replace(
            execution,
            sources=(changed_sep, *execution.sources[1:]),
        ),
        audit_provider=_clean_ost_audit,
    )

    case = report["cases"][0]
    zero_summary = case["sources"][0]["zero_shield_identity"]
    assert zero_summary["identity_passed"] is True
    assert zero_summary["release_ready"] is False
    assert case["zero_shield_identities_pass"] is True
    assert case["zero_shields_release_ready"] is False
    assert report["release_gate"]["calculation_products_release_ready"] is False
    blockers = report["release_gate"]["blockers"]
    release_blocker = next(
        item for item in blockers if item["code"] == "zero_shield_release_not_ready"
    )
    assert release_blocker["reasons"] == [reason]
    assert "zero_shield_identity_failed" not in {
        item["code"] for item in blockers
    }


def test_zero_identity_detects_added_shielding_dataset_reference() -> None:
    execution = _fake_execution()
    sep = execution.sources[0]
    zero_product = sep.zero_shield.products[0]
    reference = zero_product.spectrum.model_reference
    assert reference is not None
    changed_zero = replace(
        zero_product,
        spectrum=replace(
            zero_product.spectrum,
            model_reference=replace(
                reference,
                dataset_refs=(*reference.dataset_refs, "stopping/unexpected.csv"),
            ),
        ),
    )
    changed_sep = replace(
        sep,
        zero_shield=replace(
            sep.zero_shield,
            products=(changed_zero, *sep.zero_shield.products[1:]),
        ),
    )
    report = run_stage4_reference_validation(
        scenarios=STAGE4_REFERENCE_SCENARIOS[:1],
        modes=(AssessmentMode.CUMULATIVE,),
        case_executor=lambda _config: replace(
            execution,
            sources=(changed_sep, *execution.sources[1:]),
        ),
        audit_provider=_clean_ost_audit,
    )

    zero_summary = report["cases"][0]["sources"][0]["zero_shield_identity"]
    assert zero_summary["identity_passed"] is False
    changed_product = next(
        item
        for item in zero_summary["products"]
        if item["contract_checks"].get("dataset_refs") is False
    )
    assert changed_product["particle"] == "proton"
    assert "zero_shield_identity_failed" in {
        item["code"] for item in report["release_gate"]["blockers"]
    }


def test_stage4_reference_runner_requires_each_energy_particle_signature() -> None:
    missing_particles = (
        (RadiationSource.SEP, Particle.PROTON),
        (RadiationSource.GCR, Particle.HZE),
        (RadiationSource.ERB, Particle.ELECTRON),
    )
    for target_source, missing_particle in missing_particles:
        execution = _fake_execution()
        source_index = next(
            index
            for index, item in enumerate(execution.sources)
            if item.source is target_source
        )
        source_execution = execution.sources[source_index]
        first_shield = source_execution.shielded_by_thickness[0]
        remaining_products = tuple(
            product
            for product in first_shield.products
            if not (
                product.kind is RadiationProductKind.MISSION_FLUENCE
                and product.spectrum.particle is missing_particle
            )
        )
        updated_source = replace(
            source_execution,
            shielded_by_thickness=(
                replace(first_shield, products=remaining_products),
                *source_execution.shielded_by_thickness[1:],
            ),
        )
        updated_sources = list(execution.sources)
        updated_sources[source_index] = updated_source
        incomplete_execution = replace(execution, sources=tuple(updated_sources))

        report = run_stage4_reference_validation(
            scenarios=STAGE4_REFERENCE_SCENARIOS[:1],
            modes=(AssessmentMode.CUMULATIVE,),
            case_executor=lambda _config, result=incomplete_execution: result,
            audit_provider=_clean_ost_audit,
        )
        payload = json.loads(stage4_reference_validation_json(report))
        source_summary = next(
            item
            for item in payload["cases"][0]["sources"]
            if item["source"] == target_source.value
        )
        first_shield_summary = source_summary["behind_aluminium_shield"][0]

        assert first_shield_summary["complete"] is False
        assert first_shield_summary["missing_product_kinds"] == []
        assert {
            "axis_kind": (
                "kinetic_energy_per_nucleon"
                if missing_particle is Particle.HZE
                else "kinetic_energy"
            ),
            "particle": missing_particle.value,
            "product_kind": "mission_fluence",
            "quantity": "differential_fluence",
            "semantic_role": "energy_species",
            "species_scope": "non_aggregate",
            "temporal_basis": "mission_total",
        } in first_shield_summary["missing_product_signatures"]


def test_stage4_reference_runner_requires_typed_aggregate_combined_let() -> None:
    execution = _fake_execution()
    sep = execution.sources[0]
    first_shield = sep.shielded_by_thickness[0]
    individual_hze_let = _fake_product(
        source=RadiationSource.SEP,
        kind=RadiationProductKind.MISSION_LET_FLUENCE,
        thickness_g_cm2=first_shield.thickness_g_cm2,
        let_spectrum=True,
        aggregate_let=False,
    )
    products_without_aggregate = (
        *(
            product
            for product in first_shield.products
            if product.kind is not RadiationProductKind.MISSION_LET_FLUENCE
        ),
        individual_hze_let,
    )
    incomplete_execution = replace(
        execution,
        sources=(
            replace(
                sep,
                shielded_by_thickness=(
                    replace(first_shield, products=products_without_aggregate),
                    *sep.shielded_by_thickness[1:],
                ),
            ),
            *execution.sources[1:],
        ),
    )

    report = run_stage4_reference_validation(
        scenarios=STAGE4_REFERENCE_SCENARIOS[:1],
        modes=(AssessmentMode.CUMULATIVE,),
        case_executor=lambda _config: incomplete_execution,
        audit_provider=_clean_ost_audit,
    )
    payload = json.loads(stage4_reference_validation_json(report))
    shield_summary = payload["cases"][0]["sources"][0]["behind_aluminium_shield"][0]

    assert shield_summary["missing_product_kinds"] == []
    assert {
        "axis_kind": "linear_energy_transfer",
        "particle": "hze",
        "product_kind": "mission_let_fluence",
        "quantity": "let_differential_fluence",
        "semantic_role": "hze_direct_ion",
        "species_scope": "aggregate",
        "temporal_basis": "mission_total",
    } in shield_summary["missing_product_signatures"]
    assert shield_summary["complete"] is False


def test_stage4_integral_only_product_cannot_satisfy_differential_signature() -> None:
    execution = _fake_execution()
    sep = execution.sources[0]
    first_shield = sep.shielded_by_thickness[0]
    differential = next(
        product
        for product in first_shield.products
        if product.kind is RadiationProductKind.MISSION_FLUENCE
        and product.spectrum.particle is Particle.PROTON
    )
    integral = SpectrumProduct(
        kind=differential.kind,
        spectrum=replace(
            differential.spectrum,
            quantity=SpectrumQuantity.INTEGRAL_FLUENCE,
            y_unit=Unit.INTEGRAL_FLUENCE,
        ),
    )
    products = tuple(
        integral if product is differential else product
        for product in first_shield.products
    )
    execution = replace(
        execution,
        sources=(
            replace(
                sep,
                shielded_by_thickness=(
                    replace(first_shield, products=products),
                    *sep.shielded_by_thickness[1:],
                ),
            ),
            *execution.sources[1:],
        ),
    )

    report = run_stage4_reference_validation(
        scenarios=STAGE4_REFERENCE_SCENARIOS[:1],
        modes=(AssessmentMode.CUMULATIVE,),
        case_executor=lambda _config: execution,
        audit_provider=_clean_ost_audit,
    )
    shield = report["cases"][0]["sources"][0]["behind_aluminium_shield"][0]

    assert shield["complete"] is False
    assert any(
        signature["particle"] == "proton"
        and signature["quantity"] == "differential_fluence"
        for signature in shield["missing_product_signatures"]
    )
    integral_summary = next(
        spectrum
        for spectrum in shield["spectra"]
        if spectrum["particle"] == "proton"
    )
    assert integral_summary["quantity"] == "integral_fluence"
    assert integral_summary["integral"] is None


def test_stage4_bin_average_summary_uses_conserved_bin_sum() -> None:
    execution = _fake_execution()
    sep = execution.sources[0]
    first_shield = sep.shielded_by_thickness[0]
    proton = next(
        product
        for product in first_shield.products
        if product.spectrum.particle is Particle.PROTON
    )
    bin_average = SpectrumProduct(
        kind=proton.kind,
        spectrum=replace(
            proton.spectrum,
            ordinate_basis=SpectrumOrdinateBasis.BIN_AVERAGE,
            bin_edges=(5.0, 15.0, 30.0, 50.0),
        ),
    )
    products = tuple(
        bin_average if product is proton else product
        for product in first_shield.products
    )
    execution = replace(
        execution,
        sources=(
            replace(
                sep,
                shielded_by_thickness=(
                    replace(first_shield, products=products),
                    *sep.shielded_by_thickness[1:],
                ),
            ),
            *execution.sources[1:],
        ),
    )

    report = run_stage4_reference_validation(
        scenarios=STAGE4_REFERENCE_SCENARIOS[:1],
        modes=(AssessmentMode.CUMULATIVE,),
        case_executor=lambda _config: execution,
        audit_provider=_clean_ost_audit,
    )
    shield = report["cases"][0]["sources"][0]["behind_aluminium_shield"][0]
    summary = next(
        spectrum
        for spectrum in shield["spectra"]
        if spectrum["particle"] == "proton"
    )

    assert summary["integral"] == {
        "method": "bin_sum_over_explicit_edges",
        "unit": "cm^-2",
        "value": 80.0,
    }
