"""Semi-analytical Lee/Sambuca forward model."""

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray

from ..constants import REFRACTIVE_INDEX_SEAWATER


@dataclass
class ForwardModelResults:
    """Results from the forward model calculations.

    Attributes:
        r_substratum: The combined substrate, or substrate1 if the
            optional second substrate was not provided.
        rrs: Modelled remotely-sensed reflectance.
        rrsdp: Modelled optically-deep remotely-sensed reflectance.
        r_0_minus: Modelled remotely-sensed closed reflectance (R(0-)).
        rdp_0_minus: Modelled optically-deep remotely-sensed closed reflectance (Rdp(0-)).
        kd: Diffuse attenuation coefficient.
        kub: Bottom upwelling attenuation coefficient.
        kuc: Water column upwelling attenuation coefficient.
        a: Modelled total absorption (water + phyto + CDOM + NAP)
        a_ph_star: Specific absorption of phytoplankton.
        a_cdom_star: Modelled specific absorption of CDOM.
        a_nap_star: Modelled specific absorption of NAP.
        a_ph: Modelled absorption of phytoplankton.
        a_cdom: Modelled absorption of CDOM.
        a_nap: Modelled absorption of NAP.
        a_water: Absorption coefficient of water.
        bb: Modelled total backscatter (water + phyto + NAP).
        bb_ph_star: Modelled specific backscatter of phytoplankton.
        bb_nap_star: Modelled specific backscatter of NAP.
        bb_ph: Modelled backscatter of phytoplankton.
        bb_nap: Modelled backscatter of NAP.
        bb_water: Modelled backscatter of water.
    """

    r_substratum: NDArray[np.float64]
    rrs: NDArray[np.float64]
    rrsdp: NDArray[np.float64]
    r_0_minus: NDArray[np.float64]
    rdp_0_minus: NDArray[np.float64]
    kd: NDArray[np.float64]
    kub: NDArray[np.float64]
    kuc: NDArray[np.float64]
    a: NDArray[np.float64]
    a_ph_star: NDArray[np.float64]
    a_cdom_star: NDArray[np.float64]
    a_nap_star: NDArray[np.float64]
    a_ph: NDArray[np.float64]
    a_cdom: NDArray[np.float64]
    a_nap: NDArray[np.float64]
    a_water: NDArray[np.float64]
    bb: NDArray[np.float64]
    bb_ph_star: NDArray[np.float64]
    bb_nap_star: NDArray[np.float64]
    bb_ph: NDArray[np.float64]
    bb_nap: NDArray[np.float64]
    bb_water: NDArray[np.float64]


def forward_model(
    chl: float,
    cdom: float,
    nap: float,
    depth: float,
    substrate1: Sequence[float],
    wavelengths: Sequence[float],
    a_water: Sequence[float],
    a_ph_star: Sequence[float],
    num_bands: int,
    substrate_fraction: float = 1.0,
    substrate2: Optional[Sequence[float]] = None,
    a_cdom_slope: float = 0.0168052,
    a_nap_slope: float = 0.00977262,
    bb_ph_slope: float = 0.878138,
    bb_nap_slope: Optional[float] = None,
    lambda0cdom: float = 550.0,
    lambda0nap: float = 550.0,
    lambda0x: float = 546.0,
    x_ph_lambda0x: float = 0.00157747,
    x_nap_lambda0x: float = 0.0225353,
    a_cdom_lambda0cdom: float = 1.0,
    a_nap_lambda0nap: float = 0.00433,
    bb_lambda_ref: float = 550,
    water_refractive_index: float = REFRACTIVE_INDEX_SEAWATER,
    theta_air: float = 30.0,
    off_nadir: float = 0.0,
    q_factor: float = np.pi,
) -> ForwardModelResults:
    """Semi-analytical Lee/Sambuca forward model.

    The forward model calculates the spectra for a water column with specific optical properties.

    Args:
        chl: Concentration of chlorophyll (algal organic particulates) [mg/m³].
        cdom: Concentration of coloured dissolved organic particulates (CDOM) [1/m].
        nap: Concentration of non-algal particulates (NAP) [mg/L].
        depth: Water column depth [m].
        substrate1: A benthic substrate reflectance spectrum.
        wavelengths: Central wavelengths of the modelled spectral bands [nm].
        a_water: Absorption coefficient of pure water [1/m].
        a_ph_star: Specific absorption of phytoplankton [m²/mg].
        num_bands: The number of spectral bands.
        substrate_fraction: Substrate proportion, used to generate a
            convex combination of substrate1 and substrate2.
        substrate2: An optional second benthic substrate reflectance spectrum.
        a_cdom_slope: Slope of CDOM absorption [1/nm].
        a_nap_slope: Slope of NAP absorption [1/nm].
        bb_ph_slope: Power law exponent for the phytoplankton backscattering coefficient.
        bb_nap_slope: Power law exponent for the NAP backscattering coefficient.
            If no value is supplied, the bb_ph_slope value is used.
        lambda0cdom: Reference wavelength for CDOM absorption [nm].
        lambda0nap: Reference wavelength for NAP absorption [nm].
        lambda0x: Backscattering reference wavelength [nm].
        x_ph_lambda0x: Specific backscatter of chlorophyl at lambda0x [m²/mg].
        x_nap_lambda0x: Specific backscatter of NAP at lambda0x [m²/g].
        a_cdom_lambda0cdom: Absorption of CDOM at lambda0cdom [1/m].
        a_nap_lambda0nap: Absorption of NAP at lambda0nap [m²/g].
        bb_lambda_ref: Reference wavelength for backscattering coefficient [nm].
        water_refractive_index: Refractive index of water.
        theta_air: Solar zenith angle in degrees.
        off_nadir: Off-nadir angle in degrees.
        q_factor: Q value for producing the R(0-) values from
            modelled remotely-sensed reflectance (rrs) values.

    Returns:
        A ForwardModelResults object containing the model outputs.

    Raises:
        AssertionError: If input arrays have inconsistent lengths.
    """
    # Ensure inputs have the expected sizes
    assert len(substrate1) == num_bands, "substrate1 length must match num_bands"
    if substrate2 is not None:
        assert len(substrate2) == num_bands, "substrate2 length must match num_bands"
    assert len(wavelengths) == num_bands, "wavelengths length must match num_bands"
    assert len(a_water) == num_bands, "a_water length must match num_bands"
    assert len(a_ph_star) == num_bands, "a_ph_star length must match num_bands"

    # Convert inputs to numpy arrays
    wavelengths_arr = np.asarray(wavelengths, dtype=np.float64)
    a_water_arr = np.asarray(a_water, dtype=np.float64)
    a_ph_star_arr = np.asarray(a_ph_star, dtype=np.float64)
    substrate1_arr = np.asarray(substrate1, dtype=np.float64)
    substrate2_arr = None if substrate2 is None else np.asarray(substrate2, dtype=np.float64)

    # Sub-surface solar zenith angle in radians
    inv_refractive_index = 1.0 / water_refractive_index
    theta_w = math.asin(inv_refractive_index * math.sin(math.radians(theta_air)))

    # Sub-surface viewing angle in radians
    theta_o = math.asin(inv_refractive_index * math.sin(math.radians(off_nadir)))

    # Calculate derived SIOPS, based on
    # Mobley, Curtis D., 1994: Radiative Transfer in natural waters.
    bb_water = (0.00194 / 2.0) * np.power(bb_lambda_ref / wavelengths_arr, 4.32)
    a_cdom_star = a_cdom_lambda0cdom * np.exp(
        -a_cdom_slope * (wavelengths_arr - lambda0cdom)
    )
    a_nap_star = a_nap_lambda0nap * np.exp(-a_nap_slope * (wavelengths_arr - lambda0nap))

    # Calculate backscatter
    backscatter = np.power(lambda0x / wavelengths_arr, bb_ph_slope)
    # Specific backscatter due to phytoplankton
    bb_ph_star = x_ph_lambda0x * backscatter

    # Specific backscatter due to NAP
    # If a bb_nap_slope value has been supplied, use it.
    # Otherwise, reuse bb_ph_slope.
    if bb_nap_slope is not None:
        backscatter = np.power(lambda0x / wavelengths_arr, bb_nap_slope)
    bb_nap_star = x_nap_lambda0x * backscatter

    # Total absorption
    a_ph = chl * a_ph_star_arr
    a_cdom = cdom * a_cdom_star
    a_nap = nap * a_nap_star
    a = a_water_arr + a_ph + a_cdom + a_nap

    # Total backscatter
    bb_ph = chl * bb_ph_star
    bb_nap = nap * bb_nap_star
    bb = bb_water + bb_ph + bb_nap

    # Calculate total bottom reflectance from the two substrates
    r_substratum = substrate1_arr
    if substrate2_arr is not None:
        r_substratum = (
            substrate_fraction * substrate1_arr + (1.0 - substrate_fraction) * substrate2_arr
        )

    # Calculate optical coefficients
    kappa = a + bb
    u = bb / kappa

    # Optical path elongation for scattered photons
    # Elongation from water column
    du_column = 1.03 * np.power(1.00 + (2.40 * u), 0.50)
    # Elongation from bottom
    du_bottom = 1.04 * np.power(1.00 + (5.40 * u), 0.50)

    # Remotely sensed sub-surface reflectance for optically deep water
    rrsdp = (0.084 + 0.17 * u) * u

    # Common terms in the following calculations
    inv_cos_theta_w = 1.0 / math.cos(theta_w)
    inv_cos_theta_0 = 1.0 / math.cos(theta_o)
    du_column_scaled = du_column * inv_cos_theta_0
    du_bottom_scaled = du_bottom * inv_cos_theta_0

    # Calculate diffuse and attenuation coefficients
    kd = kappa * inv_cos_theta_w
    kuc = kappa * du_column_scaled
    kub = kappa * du_bottom_scaled

    # Remotely sensed reflectance
    kappa_d = kappa * depth
    rrs = rrsdp * (1.0 - np.exp(-(inv_cos_theta_w + du_column_scaled) * kappa_d)) + (
        (1.0 / math.pi)
        * r_substratum
        * np.exp(-(inv_cos_theta_w + du_bottom_scaled) * kappa_d)
    )

    # Create and return results
    return ForwardModelResults(
        r_substratum=r_substratum,
        rrs=rrs,
        rrsdp=rrsdp,
        r_0_minus=rrs * q_factor,
        rdp_0_minus=rrsdp * q_factor,
        kd=kd,
        kub=kub,
        kuc=kuc,
        a=a,
        a_ph_star=a_ph_star_arr,
        a_cdom_star=a_cdom_star,
        a_nap_star=a_nap_star,
        a_ph=a_ph,
        a_cdom=a_cdom,
        a_nap=a_nap,
        a_water=a_water_arr,
        bb=bb,
        bb_ph_star=bb_ph_star,
        bb_nap_star=bb_nap_star,
        bb_ph=bb_ph,
        bb_nap=bb_nap,
        bb_water=bb_water,
    )
