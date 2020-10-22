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

import copy
import unittest

import lsst.utils.tests

from lsst.afw.image import FilterLabel


class FilterLabelTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.className = "FilterLabel"
        self.physicalName = "k-0324"
        self.band = "k"

    def _labelVariants(self):
        # Contains some redundant entries to check that equality behaves
        # consistently across both construction methods.
        return iter({FilterLabel(physical=self.physicalName, band=self.band),
                     FilterLabel.fromBand(self.band),
                     FilterLabel(band=self.band),
                     FilterLabel(physical=self.physicalName),
                     })

    def testInit(self):
        with self.assertRaises(ValueError):
            FilterLabel()
        self.assertEqual(FilterLabel(physical=self.physicalName),
                         FilterLabel.fromPhysical(self.physicalName))
        self.assertEqual(FilterLabel(band=self.band),
                         FilterLabel.fromBand(self.band))
        self.assertEqual(FilterLabel(physical=self.physicalName, band=self.band),
                         FilterLabel.fromBandPhysical(self.band, self.physicalName))

        with self.assertRaises(TypeError):
            FilterLabel(physical=42)
        with self.assertRaises(TypeError):
            FilterLabel(band=("g", "r"))

    def testEqualsBasic(self):
        # Reflexivity
        for label in self._labelVariants():
            self.assertEqual(label, label)
            self.assertFalse(label != label)

        # Symmetry
        for labelA in self._labelVariants():
            for labelB in self._labelVariants():
                self.assertEqual(labelA == labelB, labelB == labelA)
                self.assertEqual(labelA != labelB, labelB != labelA)

        # Transitivity
        for labelA in self._labelVariants():
            for labelB in self._labelVariants():
                for labelC in self._labelVariants():
                    if labelA == labelB and labelB == labelC:
                        self.assertEqual(labelA, labelC)
                    # The logical implications if A != B or B != C are handled
                    # on a different iteration/permutation of (A, B, C).

    def testEqualsIdentical(self):
        self.assertEqual(FilterLabel(physical=self.physicalName), FilterLabel(physical=self.physicalName))

    def testEqualsSameText(self):
        # Ensure different kinds of labels are distinguishable, even if they have the same string
        self.assertNotEqual(FilterLabel(band=self.band), FilterLabel(physical=self.band))

    def testEqualsMissingField(self):
        self.assertNotEqual(FilterLabel(band=self.band),
                            FilterLabel(band=self.band, physical=self.physicalName))

    def testRepr(self):
        for label in self._labelVariants():
            try:
                self.assertEqual(eval(repr(label)), label)
            except (SyntaxError, ValueError):
                print(f"repr(label) = '{label!r}'")
                raise

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

    def testCopy(self):
        for label in self._labelVariants():
            copy1 = copy.copy(label)
            self._checkCopy(copy1, label)

            copy2 = copy.deepcopy(label)
            self._checkCopy(copy2, label)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
