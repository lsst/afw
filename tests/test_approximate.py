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
import unittest

import numpy as np

import lsst.utils.tests
import lsst.afw.display.ds9 as ds9
import lsst.geom
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.pex.exceptions as pexExcept

# Set to True to display things in ds9.
display = False


class ApproximateTestCase(lsst.utils.tests.TestCase):
    def makeRamp(self, binsize=1):
        """Make a linear ramp"""
        ramp = afwImage.MaskedImageF(20, 40)

        x = []
        for i in range(ramp.getWidth()):
            x.append((i + 0.5)*binsize - 0.5)

        y = []
        for j in range(ramp.getHeight()):
            y.append((j + 0.5)*binsize - 0.5)

        var = 1
        rampCoeffs = (1000, 1, 1)
        for i in range(ramp.getHeight()):
            for j in range(ramp.getWidth()):
                ramp[j, i, afwImage.LOCAL] = (rampCoeffs[0] + rampCoeffs[1]*x[j] + rampCoeffs[2]*y[i],
                                              0x0, var)

        return ramp, rampCoeffs, x, y

    def testLinearRamp(self):
        """Fit a ramp"""

        binsize = 1
        ramp, rampCoeffs, xVec, yVec = self.makeRamp(binsize)
        # Add a (labelled) bad value
        ramp[ramp.getWidth()//2, ramp.getHeight()//2, afwImage.LOCAL] = (0, 0x1, np.nan)

        if display:
            ds9.mtv(ramp, title="Input", frame=0)
        # Here's the range that the approximation should be valid (and also the
        # bbox of the image returned by getImage)
        bbox = lsst.geom.BoxI(lsst.geom.PointI(0, 0),
                              lsst.geom.PointI(binsize*ramp.getWidth() - 1,
                                               binsize*ramp.getHeight() - 1))

        order = 3                       # 1 would be enough to fit the ramp
        actrl = afwMath.ApproximateControl(
            afwMath.ApproximateControl.CHEBYSHEV, order)
        approx = afwMath.makeApproximate(xVec, yVec, ramp, bbox, actrl)

        for i, aim in enumerate([approx.getImage(),
                                 approx.getMaskedImage().getImage(),
                                 ]):
            if i == 0 and display:
                ds9.mtv(aim, title="interpolated", frame=1)
                with ds9.Buffering():
                    for x in xVec:
                        for y in yVec:
                            ds9.dot('+', x, y, size=0.4, frame=1)

            for x, y in aim.getBBox().getCorners():
                self.assertEqual(
                    aim[x, y, afwImage.LOCAL], rampCoeffs[0] + rampCoeffs[1]*x + rampCoeffs[1]*y)

    def testChebyshevEqualOrder(self):
        """Check that we enforce the condition orderX == orderY"""

        self.assertRaises(pexExcept.InvalidParameterError,
                          lambda:
                          afwMath.ApproximateControl(afwMath.ApproximateControl.CHEBYSHEV, 1, 2))

    def testNoFinitePoints(self):
        """Check that makeApproximate throws a RuntimeError if grid has no finite points and weights to fit
        """
        binsize = 1
        for badValue in [(3, 0x1, 0), (np.nan, 0x1, 1)]:
            ramp, rampCoeffs, xVec, yVec = self.makeRamp(binsize)
            ramp.set(badValue)
            bbox = lsst.geom.BoxI(lsst.geom.PointI(0, 0), lsst.geom.PointI(binsize*ramp.getWidth() - 1,
                                                                           binsize*ramp.getHeight() - 1))
            order = 2
            actrl = afwMath.ApproximateControl(
                afwMath.ApproximateControl.CHEBYSHEV, order)
            self.assertRaises(pexExcept.RuntimeError,
                              lambda: afwMath.makeApproximate(xVec, yVec, ramp, bbox, actrl))

    def testLinearRampAsBackground(self):
        """Fit a ramp"""

        ramp, rampCoeffs = self.makeRamp()[0:2]

        if display:
            ds9.mtv(ramp, title="Input", frame=0)
        # Here's the range that the approximation should be valid (and also the
        # bbox of the image returned by getImage)
        bkgd = afwMath.makeBackground(ramp, afwMath.BackgroundControl(10, 10))

        orderMax = 3                    # 1 would be enough to fit the ramp
        for order in range(orderMax + 1):
            actrl = afwMath.ApproximateControl(
                afwMath.ApproximateControl.CHEBYSHEV, order)

            approx = bkgd.getApproximate(actrl)
            # Get the Image, the MaskedImage, and the Image with a truncated
            # expansion
            for i, aim in enumerate([approx.getImage(),
                                     approx.getMaskedImage().getImage(),
                                     approx.getImage(
                                         order - 1 if order > 1 else -1),
                                     ]):
                if display and (i == 0 and order == 1):
                    ds9.mtv(aim, title="Interpolated", frame=1)

                for x, y in aim.getBBox().getCorners():
                    val = np.mean(aim.getArray()) if order == 0 else \
                        rampCoeffs[0] + rampCoeffs[1]*x + rampCoeffs[1]*y

                    self.assertEqual(aim[x, y, afwImage.LOCAL], val)
        # Check that we can't "truncate" the expansion to a higher order than
        # we requested
        self.assertRaises(pexExcept.InvalidParameterError,
                          lambda: approx.getImage(orderMax + 1, orderMax + 1))


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
