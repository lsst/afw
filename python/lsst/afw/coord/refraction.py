from __future__ import absolute_import, division, print_function
#
# LSST Data Management System
# Copyright 2008-2018 LSST/AURA.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#

from astropy import units
from astropy.units import cds
import numpy as np

from lsst.afw.geom import Angle
from lsst.afw.coord import Weather

__all__ = ["refraction", "differentialRefraction"]

# The differential refractive index is the difference of the refractive index from 1.,
#    multiplied by 1E8 to simplify the notation and equations.
deltaRefractScale = 1.0E8


def refraction(wavelength, elevation, weather, observatory):
    """Calculate overall refraction under atmospheric and observing conditions.

    The calculation is taken from Stone 1996
    "An Accurate Method for Computing Atmospheric Refraction"
    Parameters
    ----------
    wavelength : `float`
        wavelength is in nm (valid for 230.2 < wavelength < 2058.6)
    elevation : `lsst.afw.geom.Angle`
        Elevation of the observation, as an Angle.
    weather : `lsst.afw.coord.Weather`
        Class containing the measured temperature, pressure, and humidity
        at the observatory during an observation
    observatory : `lsst.afw.coord.Observatory`
        Class containing the longitude, latitude,
        and altitude of the observatory.

    Returns
    -------
    `lsst.afw.geom.Angle`
        The angular refraction for light of the given wavelength,
        under the given observing conditions.
    """
    if wavelength < 230.2:
        raise ValueError("Refraction calculation is valid for wavelengths between 230.2 and 2058.6 nm.")
    if wavelength > 2058.6:
        raise ValueError("Refraction calculation is valid for wavelengths between 230.2 and 2058.6 nm.")
    latitude = observatory.getLatitude()
    altitude = observatory.getElevation()
    if weather is None:
        weather = defaultWeather(altitude*units.meter)
    reducedN = deltaN(wavelength, weather)/deltaRefractScale
    temperature = extractTemperature(weather, useKelvin=True)
    atmosScaleheightRatio = 4.5908E-6*temperature/units.Kelvin

    # Account for oblate Earth
    # This replicates equation 10 of Stone 1996
    relativeGravity = (1. + 0.005302*np.sin(latitude.asRadians())**2. -
                       0.00000583*np.sin(2.*latitude.asRadians())**2. - 0.000000315*altitude)

    # Calculate the tangent of the zenith angle.
    tanZ = np.tan(np.pi/2. - elevation.asRadians())
    atmosTerm1 = reducedN*relativeGravity*(1. - atmosScaleheightRatio)
    atmosTerm2 = reducedN*relativeGravity*(atmosScaleheightRatio - reducedN/2.)
    result = Angle(atmosTerm1*tanZ + atmosTerm2*tanZ**3.)
    return result


def differentialRefraction(wavelength, wavelengthRef, elevation, weather, observatory):
    """Calculate the differential refraction between two wavelengths.

    Parameters
    ----------
    wavelength : `float`
        wavelength is in nm (valid for 230.2 < wavelength < 2058.6)
    wavelengthRef : `float`
        Reference wavelength, typically the effective wavelength of a filter.
    elevation : `lsst.afw.geom.Angle`
        Elevation of the observation, as an Angle.
    weather : `lsst.afw.coord.Weather`
        Class containing the measured temperature, pressure, and humidity
        at the observatory during an observation
    observatory : `lsst.afw.coord.Observatory`
        Class containing the longitude, latitude,
        and altitude of the observatory.

    Returns
    -------
    `lsst.afw.geom.Angle`
        The refraction at `wavelength` - the refraction at `wavelengthRef`.
    """
    refractionStart = refraction(wavelength, elevation, weather, observatory)
    refractionEnd = refraction(wavelengthRef, elevation, weather, observatory)
    return refractionStart - refractionEnd


def deltaN(wavelength, weather):
    """Calculate the differential refractive index of air.

    The differential refractive index is the difference of
    the refractive index from 1., multiplied by 1E8 to simplify
    the notation and equations. Calculated as (n_air - 1)*10^8

    This replicates equation 14 of Stone 1996

    Parameters
    ----------
    wavelength : `float`
        wavelength is in nanometers
    weather : `lsst.afw.coord.Weather`
        Class containing the measured temperature, pressure, and humidity
        at the observatory during an observation

    Returns
    -------
    `float`
        The difference of the refractive index of air from 1.,
        calculated as (n_air - 1)*10^8
    """
    waveNum = 1E3/wavelength  # want wave number in units 1/micron
    dryAirTerm = 2371.34 + (683939.7/(130. - waveNum**2.)) + (4547.3/(38.9 - waveNum**2.))
    wetAirTerm = 6487.31 + 58.058*waveNum**2. - 0.71150*waveNum**4. + 0.08851*waveNum**6.
    return (dryAirTerm*densityFactorDry(weather) +
            wetAirTerm*densityFactorWater(weather))


def densityFactorDry(weather):
    """Calculate dry air pressure term to refractive index calculation.

    This replicates equation 15 of Stone 1996

    Parameters
    ----------
    weather : `lsst.afw.coord.Weather`
        Class containing the measured temperature, pressure, and humidity
        at the observatory during an observation

    Returns
    -------
    `float`
        Returns the relative density of dry air
        at the given pressure and temperature.
    """
    temperature = extractTemperature(weather, useKelvin=True)
    waterVaporPressure = humidityToPressure(weather)
    airPressure = weather.getAirPressure()*units.pascal
    dryPressure = airPressure - waterVaporPressure
    eqn = (dryPressure/cds.mbar)*(57.90E-8 - 9.3250E-4*units.Kelvin/temperature +
                                  0.25844*units.Kelvin**2/temperature**2.)
    densityFactor = (1. + eqn)*(dryPressure/cds.mbar)/(temperature/units.Kelvin)
    return densityFactor


def densityFactorWater(weather):
    """Calculate water vapor pressure term to refractive index calculation.

    This replicates equation 16 of Stone 1996

    Parameters
    ----------
    weather : `lsst.afw.coord.Weather`
        Class containing the measured temperature, pressure, and humidity
        at the observatory during an observation

    Returns
    -------
    `float`
        Returns the relative density of water vapor
        at the given pressure and temperature.
    """
    temperature = extractTemperature(weather, useKelvin=True)
    waterVaporPressure = humidityToPressure(weather)
    densityEqn1 = (-2.37321E-3 + 2.23366*units.Kelvin/temperature -
                   710.792*units.Kelvin**2/temperature**2. +
                   7.75141E-4*units.Kelvin**3/temperature**3.)
    densityEqn2 = (waterVaporPressure/cds.mbar)*(1. + 3.7E-4*waterVaporPressure/cds.mbar)
    relativeDensity = waterVaporPressure*units.Kelvin/(temperature*cds.mbar)
    densityFactor = (1 + densityEqn2*densityEqn1)*relativeDensity

    return densityFactor


def humidityToPressure(weather):
    """Convert humidity and temperature to water vapor pressure.

    This replicates equations 18 & 20 of Stone 1996

    Parameters
    ----------
    weather : `lsst.afw.coord.Weather`
        Class containing the measured temperature, pressure, and humidity
        at the observatory during an observation

    Returns
    -------
    `float`
        The water vapor pressure in millibar
        calculated from the given humidity and temperature.
    """
    humidity = weather.getHumidity()
    x = np.log(humidity/100.0)
    temperature = extractTemperature(weather)
    temperatureEqn1 = (temperature + 238.3*units.Celsius)*x + 17.2694*temperature
    temperatureEqn2 = (temperature + 238.3*units.Celsius)*(17.2694 - x) - 17.2694*temperature
    dewPoint = 238.3*temperatureEqn1/temperatureEqn2
    waterVaporPressure = (4.50874 + 0.341724*dewPoint + 0.0106778*dewPoint**2 + 0.184889E-3*dewPoint**3 +
                          0.238294E-5*dewPoint**4 + 0.203447E-7*dewPoint**5)*133.32239*units.pascal

    return waterVaporPressure


def extractTemperature(weather, useKelvin=False):
    """Thin wrapper to return the measured temperature from an observation.

    Parameters
    ----------
    weather : `lsst.afw.coord.Weather`
        Class containing the measured temperature, pressure, and humidity
        at the observatory during an observation
    useKelvin : bool, optional
        Set to True to return the temperature in Kelvin instead of Celsius
        This is needed because Astropy can't easily convert
        between Kelvin and Celsius.

    Returns
    -------
    `astropy.units.quantity.Quantity`
        The temperature in Celsius, unless `useKelvin` is set.
    """
    temperature = weather.getAirTemperature()*units.Celsius
    if useKelvin:
        temperature = temperature.to(units.Kelvin, equivalencies=units.temperature())
    return temperature


def defaultWeather(altitude):
    """Set default local weather conditions if they are missing.

    Parameters
    ----------
    weather : `lsst.afw.coord.Weather`
        Class containing the measured temperature, pressure, and humidity
        at the observatory during an observation
    altitude : `astropy.units.quantity.Quantity`
        The altitude of the observatory, in meters.

    Returns
    -------
    `lsst.afw.coord.Weather`
        Updated Weather class with any `nan` values replaced by defaults.
    """
    if isinstance(altitude, units.quantity.Quantity):
        altitude2 = altitude
    else:
        altitude2 = altitude*units.meter
    p0 = 101325.*units.pascal  # sea level air pressure
    g = 9.80665*units.meter/units.second**2  # typical gravitational acceleration at sea level
    R0 = 8.31447*units.Joule/(units.mol*units.Kelvin)  # gas constant
    T0 = 19.*units.Celsius  # Typical sea-level temperature
    lapseRate = -6.5*units.Celsius/units.km  # Typical rate of change of temperature with altitude
    M = 0.0289644*units.kg/units.mol  # molar mass of dry air

    temperature = T0 + lapseRate*altitude2
    temperatureK = temperature.to(units.Kelvin, equivalencies=units.temperature())
    pressure = p0*np.exp(-(g*M*altitude2)/(R0*temperatureK))
    humidity = 40.  # Typical humidity at many observatory sites.
    weather = Weather((temperature/units.Celsius).value, (pressure/units.pascal).value, humidity)
    return weather
