#
# LSST Data Management System
# Copyright 2008, 2009, 2010 LSST Corporation.
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
import math
import os
import unittest

from builtins import range
import numpy as np

import lsst.utils.tests
import lsst.pex.exceptions
import lsst.afw.math as afwMath
import lsst.afw.image as afwImage
import lsst.afw.geom as afwGeom
import lsst.pex.exceptions as pexExcept

try:
    afwdataDir = lsst.utils.getPackageDir("afwdata")
except pexExcept.NotFoundError:
    afwdataDir = None

try:
    type(display)
except NameError:
    display = False


class StatisticsTestCase(lsst.utils.tests.TestCase):
    """A test case for Statistics"""

    def setUp(self):
        self.nx = 4096
        self.ny = 4096

        self.mi = afwImage.MaskedImageD(afwGeom.Extent2I(self.nx, self.ny))
        self.mi.getImage().set(1.0)
        self.mi.getVariance().set(0.1)

        self.image = self.mi.getImage().getArray()
        self.mask = self.mi.getMask().getArray()
        self.variance = self.mi.getVariance().getArray()
        self.weight = self.mi.getImage().getArray()

    def tearDown(self):
        del self.mi

    def testStandard(self):
        sctrl = afwMath.NewStatisticsControl()
        sctrl.baseCaseSize = 100

        result = afwMath.standardStatistics(self.image.ravel(), self.mask.ravel(), self.weight.ravel(), self.variance.ravel(), True, True, False, sctrl)
        print("count", result.count)
        print("npcount", self.nx*self.ny)
        print("mean", result.mean)
        print("npmean", np.average(self.image.ravel(), weights=self.weight.ravel()))
        print("min", result.min)
        print("npmin", np.min(self.image.ravel()))
        print("max", result.max)
        print("npmax", np.max(self.image.ravel()))
        print("variance", result.variance)
        print("npvariance", np.var(self.image.ravel(), ddof=1))
        print("biasedVariance", result.biasedVariance)
        print("npbiasedvariance", np.var(self.image.ravel(), ddof=0))
        print("median", result.median)
        print("npmedian", np.median(self.image.ravel()))

class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()

if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
