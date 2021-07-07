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
"""Tests for statsCtrl mask propagation thresholds."""
import unittest

import lsst.utils.tests
import lsst.afw.math as afwMath


class PropagationThresholdTestCase(lsst.utils.tests.TestCase):
    """Testing for propagation thresholds."""
    def testEmpty(self):
        """Test retrieving an empty threshold."""
        statsCtrl = afwMath.StatisticsControl()
        self.assertEqual(statsCtrl.getMaskPropagationThreshold(0), 1.0)

    def testSetSingle(self):
        """Test setting and retrieving a single threshold."""
        statsCtrl = afwMath.StatisticsControl()
        statsCtrl.setMaskPropagationThreshold(1, 0.1)

        for bit in range(32):
            if bit == 1:
                self.assertEqual(statsCtrl.getMaskPropagationThreshold(bit), 0.1)
            else:
                self.assertEqual(statsCtrl.getMaskPropagationThreshold(bit), 1.0)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
