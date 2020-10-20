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

from lsst.afw.image import FilterLabel


class FilterLabelTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.className = "FilterLabel"
        self.physicalName = "k-0324"
        self.band = "k"

    def _labelVariants(self):
        return iter({FilterLabel(physical=self.physicalName, band=self.band),
                     FilterLabel.fromBand(self.band),
                     FilterLabel(physical=self.physicalName),
                     })

    def _checkProperty(self, label, has, property, value):
        # For consistency with C++ API, getting a missing label raises instead of returning None
        if value:
            self.assertTrue(has(label))
            self.assertEqual(property.__get__(label), value)
        else:
            self.assertFalse(has(label))
            with self.assertRaises(RuntimeError):
                property.__get__(label)

    def _checkFactory(self, label, band, physical):
        self._checkProperty(label, FilterLabel.hasBandLabel, FilterLabel.bandLabel, band)
        self._checkProperty(label, FilterLabel.hasPhysicalLabel, FilterLabel.physicalLabel, physical)

    def testFactories(self):
        """This method tests the getters as well as the factories, since their behaviors are linked.
        """
        self._checkFactory(FilterLabel.fromBandPhysical(self.band, self.physicalName),
                           self.band, self.physicalName)
        self._checkFactory(FilterLabel.fromBand(self.band), self.band, None)
        self._checkFactory(FilterLabel.fromPhysical(self.physicalName), None, self.physicalName)

    def _checkCopy(self, label1, label2):
        self.assertEqual(label1, label2)
        self.assertIsNot(label1, label2)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
