#
# LSST Data Management System
# See COPYRIGHT file at the top of the source tree.
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
import numpy as np
import unittest

from lsst.geom import Angle, degrees
from lsst.afw.coord import Observatory, Weather
from lsst.afw.coord.refraction import refraction, differentialRefraction
import lsst.utils.tests


class RefractionTestSuite(lsst.utils.tests.TestCase):
    """Test the refraction calculations."""

    def setUp(self):
        """Define parameters used by every test."""
        lsstLat = -30.244639*degrees
        lsstLon = -70.749417*degrees
        lsstAlt = 2663.
        lsstTemperature = 20.*units.Celsius  # in degrees Celsius
        lsstHumidity = 10.  # in percent
        lsstPressure = 73892.*units.pascal  # 1 atmosphere.
        np.random.seed(5)

        self.weather = Weather(lsstTemperature/units.Celsius, lsstPressure/units.pascal, lsstHumidity)
        self.observatory = Observatory(lsstLon, lsstLat, lsstAlt)

    def testWavelengthRangeError(self):
        """Refraction should raise an error if the wavelength is out of range."""
        elevation = Angle(np.random.random()*np.pi/2.)
        wl_low = 230.
        wl_high = 2059.
        self.assertRaises(ValueError, refraction, wl_low, elevation, self.observatory, weather=self.weather)
        self.assertRaises(ValueError, refraction, wl_high, elevation, self.observatory, weather=self.weather)

    def testZenithZero(self):
        """There should be no refraction exactly at zenith."""
        elevation = Angle(np.pi/2.)
        wl = 505.  # in nm
        refractZen = refraction(wl, elevation, self.observatory, weather=self.weather)
        self.assertAlmostEqual(refractZen.asDegrees(), 0.)

    def testNoDifferential(self):
        """There should be no differential refraction if the wavelength is the same as the reference."""
        wl = 470.  # in nm
        wl_ref = wl
        elevation = Angle(np.random.random()*np.pi/2.)
        diffRefraction = differentialRefraction(wl, wl_ref, elevation, self.observatory, weather=self.weather)
        self.assertFloatsAlmostEqual(diffRefraction.asDegrees(), 0.)

    def testRefractHighAirmass(self):
        """Compare the refraction calculation to precomputed values."""
        elevation = Angle(np.pi/6.)  # Airmass 2.0
        wls = [370., 480., 620., 860., 960., 1025.]  # in nm
        refVals = [73.04868430514726,
                   71.74884360909664,
                   71.02058121935002,
                   70.51172189207065,
                   70.40446894800584,
                   70.35113687114644,
                   ]
        for wl, refVal in zip(wls, refVals):
            refract = refraction(wl, elevation, self.observatory, weather=self.weather)
            self.assertFloatsAlmostEqual(refract.asArcseconds(), refVal, rtol=1e-3)

    def testRefractWeatherNan(self):
        """Test the values of refraction when no weather is supplied."""
        elevation = Angle(np.random.random()*np.pi/2.)
        elevation = Angle(np.pi/6.)  # Airmass 2.0
        wls = [370., 480., 620., 860., 960., 1025.]  # in nm
        refVals = [76.7339313496466,
                   75.36869048516252,
                   74.60378630142982,
                   74.06932963258161,
                   73.95668242959853,
                   73.9006681751504,
                   ]
        for wl, refVal in zip(wls, refVals):
            refract = refraction(wl, elevation, self.observatory)
            self.assertFloatsAlmostEqual(refract.asArcseconds(), refVal, rtol=1e-3)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
