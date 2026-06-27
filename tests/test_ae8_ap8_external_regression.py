import pytest

from radar.core.project import CalculationConfig, MissionConfig, OrbitConfig
from radar.core.types import Particle, SolarActivityLevel, SpectrumQuantity
from radar.erb.constants import ERB_SECONDS_PER_YEAR
from radar.erb.model import Ae8Ap8ErbModel, ErbModelInput


EXTERNAL_REFERENCE_PROTON_MEAN_FLUX = (
    2847.9842091987402, 2843.2960462967467, 2837.4052046954062, 1780.8323121123506,
    1767.4803296681484, 1762.9356309721275, 1933.3061038360242, 1932.3039154380263,
    1935.9235730674386, 1597.255031358528, 1741.8933099496044, 1722.854718387081,
    1699.1940432008355, 1654.1283684763036, 1619.4377653282909, 1577.0120984202874,
    1753.8617892066445, 1827.9233477295097, 2150.2310438227382, 1714.8263086881186,
    1636.1400938478394, 1276.3324257090303, 768.20348462252423, 418.22961057947458,
    336.1682839323355, 205.07482969285789, 182.72321247875709, 138.74434035090101,
    117.67605995356269, 96.333012544198766, 74.738058020840043, 55.758103468785698,
    38.588543134807693, 22.862672692028539, 13.374925345373036, 6.8320285263563578,
    2.9417371185571852, 1.0232612190578485,
)

EXTERNAL_REFERENCE_PROTON_PEAK_FLUX = (
    4150.9244848308454, 4144.6920100911211, 4136.8591517312962, 2816.4458189325487,
    2798.5843757116518, 2791.4193161087824, 3715.718529123124, 3520.9401518222289,
    2889.1853193296811, 2457.5248767113258, 3160.8335577450198, 3123.4345491050212,
    3076.9832610943463, 2828.63877488761, 2764.7320195793527, 2688.4922788782337,
    3097.4233389952674, 3232.3047729843152, 3831.3818248099928, 2960.8985996954307,
    2659.6862334054667, 2079.2216856866326, 1310.8959552299295, 618.33531846446829,
    494.14896609684916, 301.70865424307141, 266.9145380878947, 193.96790078795223,
    162.83356278163564, 132.87313978853379, 101.56068122981139, 75.138728322538384,
    51.447654288750634, 29.759368632483522, 17.243085232697442, 8.702968520805765,
    3.6905523738998007, 1.2589423840484339,
)

EXTERNAL_REFERENCE_PROTON_INTEGRAL_FLUX = (
    45563.589607943206, 45489.911097575758, 45397.328533078231, 45305.073209362054,
    45213.429671648708, 45098.62782300065, 44947.144519544752, 44747.912842202786,
    44496.913755205525, 44210.249858092619, 43866.510033603328, 43418.053395327399,
    42860.471315509669, 42172.817256883427, 41327.590862857272, 40288.709366682015,
    38923.56517028599, 37076.311326899755, 34492.721384556455, 31362.546444177191,
    27920.285936881744, 24186.806566967331, 20955.875934444237, 18621.125226795055,
    16688.249275181857, 14979.373510079253, 13397.005774872336, 11759.22049662584,
    10104.421826386293, 8368.765439522007, 6627.4228938013639, 4959.3625976480898,
    3449.4824293363549, 2228.9121637966005, 1323.979087102671, 698.57926057394479,
    326.56720474021085, 143.00631749989668,
)

EXTERNAL_REFERENCE_ELECTRON_MEAN_FLUX = (
    269655972.44462049, 249516196.67708844, 226342389.8716628, 289941062.63900161,
    230833562.54722703, 173028256.93382612, 121197328.99227525, 90608956.76246345,
    46305515.55094476, 20268403.687432893, 5171333.993624025, 2407509.668203969,
    869521.51404784643, 346884.91371213744, 177016.15639610309, 73464.082153964788,
    33481.595422754639, 19386.461738080347, 1703.2773398155423, 0.0,
    0.0, 0.0, 0.0, 0.0,
)

EXTERNAL_REFERENCE_ELECTRON_PEAK_FLUX = (
    441176784.51345855, 411363491.48707664, 376690067.50576949, 464667352.19238448,
    382593436.16773599, 298520655.23875558, 219408571.09675846, 192996513.8768234,
    95822279.200729579, 39807746.31521786, 8317871.1312648589, 4032082.701845604,
    1534421.1414559705, 573408.98710795096, 296163.34190542891, 124964.87201298162,
    56787.048890679638, 33809.648002688045, 2758.7883033806247, 0.0,
    0.0, 0.0, 0.0, 0.0,
)

EXTERNAL_REFERENCE_ELECTRON_INTEGRAL_FLUX = (
    45807714.391046464, 42453725.7508137, 38586166.126486525, 33291381.79358463,
    26623465.684832141, 20137800.516229279, 14218410.400122473, 8828466.4058227949,
    4600790.1996829435, 2066715.255678525, 958842.62918647355, 497605.51776567817,
    256174.64923542869, 141499.96109740919, 77109.398726023559, 39453.820393997441,
    18932.901947627684, 5776.2510998496482, 1257.6186816753941, 561.95384332977449,
    561.95384332977449, 561.95384332977449, 561.95384332977449, 561.95384332977449,
)


def _external_reference_case_result():
    config = CalculationConfig(
        mission=MissionConfig(
            launch_year=2027,
            lifetime_years=2,
            solar_activity_level=SolarActivityLevel.MEAN,
        ),
        orbit=OrbitConfig.circular(altitude_km=2_000.0, inclination_deg=0.0),
    )
    model = Ae8Ap8ErbModel(
        anomaly_samples=2,
        node_samples=2,
        solar_mode="cosrad",
        solar_reference_start_year=2027,
        igrf_epoch=1985.0,
        peak_state="worst",
    )

    return model.calculate(ErbModelInput(config=config))


@pytest.fixture(scope="module")
def external_reference_case_result():
    return _external_reference_case_result()


def test_ae8_ap8_model_matches_external_reference_proton_fluxes(
    external_reference_case_result,
) -> None:
    result = external_reference_case_result

    proton_mean = result.spectra[0]
    proton_peak = result.spectra[1]
    proton_fluence = result.spectra[2]

    assert proton_mean.particle is Particle.PROTON
    assert proton_mean.quantity is SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX
    assert proton_peak.quantity is SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX
    assert proton_fluence.quantity is SpectrumQuantity.DIFFERENTIAL_FLUENCE
    assert proton_mean.y == pytest.approx(
        EXTERNAL_REFERENCE_PROTON_MEAN_FLUX,
        rel=1.0e-12,
    )
    assert proton_peak.y == pytest.approx(
        EXTERNAL_REFERENCE_PROTON_PEAK_FLUX,
        rel=1.0e-12,
    )


def test_ae8_ap8_model_matches_external_reference_electron_fluxes(
    external_reference_case_result,
) -> None:
    result = external_reference_case_result

    electron_mean = result.spectra[3]
    electron_peak = result.spectra[4]
    electron_fluence = result.spectra[5]

    assert electron_mean.particle is Particle.ELECTRON
    assert electron_mean.quantity is SpectrumQuantity.MEAN_DIFFERENTIAL_FLUX
    assert electron_peak.quantity is SpectrumQuantity.MAXIMUM_DIFFERENTIAL_FLUX
    assert electron_fluence.quantity is SpectrumQuantity.DIFFERENTIAL_FLUENCE
    assert electron_mean.y == pytest.approx(
        EXTERNAL_REFERENCE_ELECTRON_MEAN_FLUX,
        rel=1.0e-12,
        abs=1.0e-12,
    )
    assert electron_peak.y == pytest.approx(
        EXTERNAL_REFERENCE_ELECTRON_PEAK_FLUX,
        rel=1.0e-12,
        abs=1.0e-12,
    )


def test_ae8_ap8_model_matches_external_reference_integral_fluxes(
    external_reference_case_result,
) -> None:
    result = external_reference_case_result

    proton_integral = result.integral_spectra[0]
    electron_integral = result.integral_spectra[1]

    assert proton_integral.particle is Particle.PROTON
    assert electron_integral.particle is Particle.ELECTRON
    assert proton_integral.integral_flux_gt_e == pytest.approx(
        EXTERNAL_REFERENCE_PROTON_INTEGRAL_FLUX,
        rel=1.0e-12,
    )
    assert electron_integral.integral_flux_gt_e == pytest.approx(
        EXTERNAL_REFERENCE_ELECTRON_INTEGRAL_FLUX,
        rel=1.0e-12,
    )


def test_ae8_ap8_model_external_reference_fluence_is_mean_flux_times_duration(
    external_reference_case_result,
) -> None:
    result = external_reference_case_result
    mission_seconds = 2 * ERB_SECONDS_PER_YEAR

    assert result.spectra[2].y == pytest.approx(
        tuple(value * mission_seconds for value in EXTERNAL_REFERENCE_PROTON_MEAN_FLUX),
        rel=1.0e-12,
    )
    assert result.spectra[5].y == pytest.approx(
        tuple(value * mission_seconds for value in EXTERNAL_REFERENCE_ELECTRON_MEAN_FLUX),
        rel=1.0e-12,
        abs=1.0e-12,
    )
