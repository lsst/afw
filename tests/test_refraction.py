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

from __future__ import absolute_import, division, print_function

from astropy import units as u
import numpy as np
import unittest

from lsst.afw.geom import Angle
from lsst.afw.coord import Observatory, Weather
from lsst.afw.coord.refraction import refraction, differentialRefraction
from lsst.afw.geom import degrees
import lsst.utils.tests


class RefractionTestSuite(lsst.utils.tests.TestCase):
    """Test the refraction calculations."""

    def setUp(self):
        """Define parameters used by every test."""
        lsstLat = -30.244639*degrees
        lsstLon = -70.749417*degrees
        lsstAlt = 2663.
        lsstTemperature = 20.*u.Celsius  # in degrees Celcius
        lsstHumidity = 10.  # in percent
        lsstPressure = 101325.*u.pascal  # 1 atmosphere.
        self.randGen = np.random
        self.randGen.seed = 5

        self.weather = Weather(lsstTemperature.value, lsstPressure.value, lsstHumidity)
        self.observatory = Observatory(lsstLon, lsstLat, lsstAlt)

    def testZenithZero(self):
        """There should be no refraction exactly at zenith."""
        elevation = Angle(np.pi/2.)
        wl = 505.  # in nm
        refractZen = refraction(wl, elevation)
        self.assertAlmostEqual(refractZen.asDegrees(), 0.)

    def testNoDifferential(self):
        """There should be no differential refraction if the wavelength is the same as the reference."""
        wl = 470.  # in nm
        wl_ref = wl
        elevation = Angle(self.randGen.random()*np.pi/2.)
        diffRefraction = differentialRefraction(wl, wl_ref, elevation)
        self.assertFloatsAlmostEqual(diffRefraction.asDegrees(), 0.)

    def testRefractHighAirmass(self):
        """Compare the refraction calculation to precomputed values."""
        elevation = Angle(np.pi/6.)  # Airmass 2.0
        wls = [370., 480., 620., 860., 960., 1025.]  # in nm
        refVals = [100.18009112043688,
                   98.39816181171116,
                   97.39979727127752,
                   96.70222559729264,
                   96.5552003626887,
                   96.482091761146,
                   ]
        for wl, refVal in zip(wls, refVals):
            refract = refraction(wl, elevation)
            self.assertFloatsAlmostEqual(refract.asArcseconds(), refVal, rtol=1e-3)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
