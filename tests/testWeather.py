#
# LSST Data Management System
# Copyright 2016 LSST Corporation.
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
"""
Tests for lsst.afw.image.Weather
"""
from __future__ import absolute_import, division, print_function
import unittest

import lsst.utils.tests
import lsst.pex.exceptions
from lsst.afw.coord import Weather


class WeatherTestCase(unittest.TestCase):
    """Test lsst.afw.coord.Weather, a simple struct-like class"""
    def testBasics(self):
        prevWeather = None
        for temp, pressure in ((1.1, 2.2), (100.1, 200.2)):  # arbitrary values
            for humidity in (0.0, 10.1, 100.0, 120.5):  # 0 and greater, including supersaturation
                weather = Weather(temp, pressure, humidity)
                self.assertEqual(weather.getAirTemperature(), temp)
                self.assertEqual(weather.getAirPressure(), pressure)
                self.assertEqual(weather.getHumidity(), humidity)

                # test copy constructor
                weatherCopy = Weather(weather)
                self.assertEqual(weatherCopy.getAirTemperature(), temp)
                self.assertEqual(weatherCopy.getAirPressure(), pressure)
                self.assertEqual(weatherCopy.getHumidity(), humidity)

                # test == (using a copy, to make sure the test is not based on identity) and !=
                self.assertEqual(weather, weatherCopy)
                if prevWeather is not None:
                    self.assertNotEqual(weather, prevWeather)
                prevWeather = weather

    def testBadHumidity(self):
        """Check bad humidity handling (humidity is the only value whose range is checked)"""
        for humidity in (-1, -0.0001):
            with self.assertRaises(lsst.pex.exceptions.InvalidParameterError):
                Weather(1.1, 2.2, humidity)


def setup_module(module):
    lsst.utils.tests.init()


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
