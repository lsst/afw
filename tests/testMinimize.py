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
import unittest

from builtins import zip
import numpy as np

import lsst.utils.tests
import lsst.afw.math as afwMath


class MinimizeTestCase(lsst.utils.tests.TestCase):

    def testMinimize2(self):

        variances = np.array([0.01, 0.01, 0.01, 0.01])
        xPositions = np.array([0.0, 1.0, 0.0, 1.0])
        yPositions = np.array([0.0, 0.0, 1.0, 1.0])

        polyOrder = 1
        polyFunc = afwMath.PolynomialFunction2D(polyOrder)

        modelParams = [0.1, 0.2, 0.3]
        polyFunc.setParameters(modelParams)
        measurements = []
        for x, y in zip(xPositions, yPositions):
            measurements.append(polyFunc(x, y))
        print("measurements=", measurements)

        # Set up initial guesses
        nParameters = polyFunc.getNParameters()
        initialParameters = np.zeros(nParameters, float)
        stepsize = np.ones(nParameters, float)
        stepsize *= 0.1

        # Minimize!
        fitResults = afwMath.minimize(
            polyFunc,
            initialParameters,
            stepsize,
            measurements,
            variances,
            xPositions,
            yPositions,
            0.1,
        )

        print("modelParams=", modelParams)
        print("fitParams  =", fitResults.parameterList)
        self.assertTrue(fitResults.isValid, "fit failed")
        self.assertFloatsAlmostEqual(np.array(modelParams), np.array(fitResults.parameterList), 1e-11)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()

if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
