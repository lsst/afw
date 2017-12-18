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

from astropy import units as u
from astropy.units import cds
import numpy as np

from lsst.afw.geom import Angle
from lsst.afw.coord import Observatory, Weather
from lsst.afw.geom import degrees

__all__ = ["refraction", "differentialRefraction"]

lsstLat = -30.244639*degrees
lsstLon = -70.749417*degrees
lsstAlt = 2663.
lsstTemperature = 20.*u.Celsius  # in degrees Celcius
lsstHumidity = 10.  # in percent
lsstPressure = 101325.*u.pascal  # 1 atmosphere.

lsstWeather = Weather(lsstTemperature.value, lsstPressure.value, lsstHumidity)
lsstObservatory = Observatory(lsstLon, lsstLat, lsstAlt)


def refraction(wavelength, elevation, weather=lsstWeather, observatory=lsstObservatory):
    """Calculate overall refraction under atmospheric and observing conditions.

    Parameters
    ----------
    wavelength : float
        wavelength is in nm (valid for 230.2 < wavelength < 2058.6)
    elevation : lsst.afw.geom Angle
        Elevation of the observation, as an Angle.
    weather : lsst.afw.coord Weather, optional
        Class containing the measured temperature, pressure, and humidity
        at the observatory during an observation
    observatory : lsst.afw.coord Observatory, optional
        Class containing the longitude, latitude, and altitude of the observatory.

    Returns
    -------
    lsst.afw.geom Angle
        The angular refraction for light of the given wavelength, under the given observing conditions.
    """
    latitude = observatory.getLatitude()
    altitude = observatory.getElevation()

    reducedN = deltaN(wavelength, weather)*1E-8

    temperature = _extractTemperature(weather, useKelvin=True)
    atmosScaleheightRatio = float(4.5908E-6*temperature/u.Kelvin)

    # Account for oblate Earth
    relativeGravity = (1. + 0.005302*np.sin(latitude.asRadians())**2. -
                       0.00000583*np.sin(2.*latitude.asRadians())**2. - 0.000000315*altitude)

    # Calculate the tangent of the zenith angle.
    tanZ = np.tan(np.pi/2. - elevation.asRadians())

    atmosTerm1 = reducedN*relativeGravity*(1. - atmosScaleheightRatio)
    atmosTerm2 = reducedN*relativeGravity*(atmosScaleheightRatio - reducedN/2.)
    result = Angle(float(atmosTerm1*tanZ + atmosTerm2*tanZ**3.))
    return result


def differentialRefraction(wavelength, wavelengthRef, elevation,
                           weather=lsstWeather, observatory=lsstObservatory):
    """Calculate the differential refraction between two wavelengths.

    Parameters
    ----------
    wavelength : float
        wavelength is in nm (valid for 230.2 < wavelength < 2058.6)
    wavelengthRef : float
        Reference wavelength, typically the effective wavelength of a filter.
    elevation : lsst.afw.geom Angle
        Elevation of the observation, as an Angle.
    weather : lsst.afw.coord Weather, optional
        Class containing the measured temperature, pressure, and humidity
        at the observatory during an observation
    observatory : lsst.afw.coord Observatory, optional
        Class containing the longitude, latitude, and altitude of the observatory.

    Returns
    -------
    lsst.afw.geom Angle
        The refraction at `wavelength` - the refraction at `wavelengthRef`.
    """
    refractionStart = refraction(wavelength, elevation, weather=weather, observatory=observatory)
    refractionEnd = refraction(wavelengthRef, elevation, weather=weather, observatory=observatory)
    return refractionStart - refractionEnd


def deltaN(wavelength, weather):
    """Calculate the differential refractive index of air.

    The differential refractive index is the difference of the refractive index from 1.,
    multiplied by 1E8 to simplify the notation and equations.
    Calculated as (n_air - 1)*10^8

    This replicates equation 14 of Stone 1996 "An Accurate Method for Computing Atmospheric Refraction"

    Parameters
    ----------
    wavelength : float
        wavelength is in nanometers
    weather : lsst.afw.coord Weather
        Class containing the measured temperature, pressure, and humidity
        at the observatory during an observation

    Returns
    -------
    float
        The difference of the refractive index of air from 1., calculated as (n_air - 1)*10^8
    """
    waveNum = 1E3/wavelength  # want wave number in units 1/micron

    dryAirTerm = 2371.34 + (683939.7/(130. - waveNum**2.)) + (4547.3/(38.9 - waveNum**2.))

    wetAirTerm = 6487.31 + 58.058*waveNum**2. - 0.71150*waveNum**4. + 0.08851*waveNum**6.

    return (dryAirTerm*densityFactorDry(weather) +
            wetAirTerm*densityFactorWater(weather))


def densityFactorDry(weather):
    """Calculate dry air pressure term to refractive index calculation.

    This replicates equation 15 of Stone 1996 "An Accurate Method for Computing Atmospheric Refraction"

    Parameters
    ----------
    weather : lsst.afw.coord Weather
        Class containing the measured temperature, pressure, and humidity
        at the observatory during an observation

    Returns
    -------
    float
        Returns the relative density of dry air at the given pressure and temperature.
    """
    temperature = _extractTemperature(weather, useKelvin=True)
    waterVaporPressure = humidityToPressure(weather)
    airPressure = _extractPressure(weather)
    dryPressure = airPressure - waterVaporPressure

    eqn = (dryPressure/cds.mbar)*(57.90E-8 - 9.3250E-4*u.Kelvin/temperature +
                                  0.25844*u.Kelvin**2/temperature**2.)

    densityFactor = float((1. + eqn)*(dryPressure/cds.mbar)/(temperature/u.Kelvin))

    return densityFactor


def densityFactorWater(weather):
    """Calculate water vapor pressure term to refractive index calculation.

    This replicates equation 16 of Stone 1996 "An Accurate Method for Computing Atmospheric Refraction"

    Parameters
    ----------
    weather : lsst.afw.coord Weather
        Class containing the measured temperature, pressure, and humidity
        at the observatory during an observation

    Returns
    -------
    float
        Returns the relative density of water vapor at the given pressure and temperature.
    """
    temperature = _extractTemperature(weather, useKelvin=True)
    waterVaporPressure = humidityToPressure(weather)

    densityEqn1 = float(-2.37321E-3 + 2.23366*u.Kelvin/temperature -
                        710.792*u.Kelvin**2/temperature**2. +
                        7.75141E-4*u.Kelvin**3/temperature**3.)

    densityEqn2 = float(waterVaporPressure/cds.mbar)*(1. + 3.7E-4*waterVaporPressure/cds.mbar)

    relativeDensity = float(waterVaporPressure*u.Kelvin/(temperature*cds.mbar))
    densityFactor = (1 + densityEqn2*densityEqn1)*relativeDensity

    return densityFactor


def humidityToPressure(weather):
    """Simple function that converts humidity and temperature to water vapor pressure.

    This replicates equations 18 & 20 of Stone 1996 "An Accurate Method for Computing Atmospheric Refraction"

    Parameters
    ----------
    weather : lsst.afw.coord Weather
        Class containing the measured temperature, pressure, and humidity
        at the observatory during an observation

    Returns
    -------
    float
        The water vapor pressure in millibar, calculated from the given humidity and temperature.
    """
    if np.isnan(weather.getHumidity()):
        humidity = lsstWeather.getHumidity()
    else:
        humidity = weather.getHumidity()
    x = np.log(humidity/100.0)
    temperature = _extractTemperature(weather)
    temperatureEqn1 = (temperature + 238.3*u.Celsius)*x + 17.2694*temperature
    temperatureEqn2 = (temperature + 238.3*u.Celsius)*(17.2694 - x) - 17.2694*temperature
    dewPoint = 238.3*float(temperatureEqn1/temperatureEqn2)

    waterVaporPressure = (4.50874 + 0.341724*dewPoint + 0.0106778*dewPoint**2 + 0.184889E-3*dewPoint**3 +
                          0.238294E-5*dewPoint**4 + 0.203447E-7*dewPoint**5)*133.32239*u.pascal

    return waterVaporPressure


def _extractTemperature(weather, useKelvin=False):
    """Thin wrapper to return the measured temperature from an observation with astropy units attached.

    Parameters
    ----------
    weather : lsst.afw.coord Weather
        Class containing the measured temperature, pressure, and humidity
        at the observatory during an observation
    useKelvin : bool, optional
        Set to True to return the temperature in Kelvin instead of Celsius
        This is needed because Astropy can't easily convert between Kelvin and Celsius.

    Returns
    -------
    astropy.units
        The temperature in Celsius, unless `useKelvin` is set.
    """
    temperature = weather.getAirTemperature()
    if np.isnan(temperature):
        temperature = lsstWeather.getAirTemperature()*u.Celsius
    else:
        temperature *= u.Celsius
    if useKelvin:
        temperature = temperature.to(u.Kelvin, equivalencies=u.temperature())
    return temperature


def _extractPressure(weather):
    """Thin wrapper to return the measured pressure from an observation with astropy units attached.

    Parameters
    ----------
    weather : lsst.afw.coord Weather
        Class containing the measured temperature, pressure, and humidity
        at the observatory during an observation

    Returns
    -------
    astropy.units
        The air pressure in pascals.
    """
    pressure = weather.getAirPressure()
    if np.isnan(pressure):
        pressure = lsstWeather.getAirPressure()*u.pascal
    else:
        pressure *= u.pascal
    return pressure
