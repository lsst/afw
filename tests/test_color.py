# This file is part of afw.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
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
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import unittest

import lsst.utils.tests
import lsst.afw.image as afwImage


class ColorTestCase(lsst.utils.tests.TestCase):
    def testCtor(self):
        afwImage.Color()
        afwImage.Color(colorValue=1.2, colorType="g-r")

    def testIsIndeterminate(self):
        """Test that a default-constructed Color tests True, but ones with a g-r value test False"""
        self.assertTrue(afwImage.Color().isIndeterminate())
        self.assertFalse(afwImage.Color(colorValue=1.2, colorType="g-r").isIndeterminate())

    def testGetColor(self):
        color = afwImage.Color(colorValue=0.42, colorType="g-r")
        self.assertEqual(color.getColorValue(), 0.42)
        self.assertEqual(color.getColorType(), "g-r")


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
