#pybind11##!/usr/bin/env python
#pybind11#from __future__ import absolute_import, division
#pybind11#from builtins import range
#pybind11#
#pybind11##
#pybind11## LSST Data Management System
#pybind11## Copyright 2008, 2009, 2010 LSST Corporation.
#pybind11##
#pybind11## This product includes software developed by the
#pybind11## LSST Project (http://www.lsst.org/).
#pybind11##
#pybind11## This program is free software: you can redistribute it and/or modify
#pybind11## it under the terms of the GNU General Public License as published by
#pybind11## the Free Software Foundation, either version 3 of the License, or
#pybind11## (at your option) any later version.
#pybind11##
#pybind11## This program is distributed in the hope that it will be useful,
#pybind11## but WITHOUT ANY WARRANTY; without even the implied warranty of
#pybind11## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#pybind11## GNU General Public License for more details.
#pybind11##
#pybind11## You should have received a copy of the LSST License Statement and
#pybind11## the GNU General Public License along with this program.  If not,
#pybind11## see <http://www.lsstcorp.org/LegalNotices/>.
#pybind11##
#pybind11#import unittest
#pybind11#import numpy as np
#pybind11#import lsst.utils.tests
#pybind11#import lsst.afw.display.ds9 as ds9
#pybind11#import lsst.afw.geom as afwGeom
#pybind11#import lsst.afw.image as afwImage
#pybind11#import lsst.afw.math as afwMath
#pybind11#import lsst.pex.exceptions as pexExcept
#pybind11#
#pybind11## Set to True to display things in ds9.
#pybind11#display = False
#pybind11#
#pybind11#
#pybind11#class ApproximateTestCase(lsst.utils.tests.TestCase):
#pybind11#    def makeRamp(self, binsize=1):
#pybind11#        """Make a linear ramp"""
#pybind11#        ramp = afwImage.MaskedImageF(20, 40)
#pybind11#
#pybind11#        x = []
#pybind11#        for i in range(ramp.getWidth()):
#pybind11#            x.append((i + 0.5)*binsize - 0.5)
#pybind11#
#pybind11#        y = []
#pybind11#        for j in range(ramp.getHeight()):
#pybind11#            y.append((j + 0.5)*binsize - 0.5)
#pybind11#
#pybind11#        var = 1
#pybind11#        rampCoeffs = (1000, 1, 1)
#pybind11#        for i in range(ramp.getHeight()):
#pybind11#            for j in range(ramp.getWidth()):
#pybind11#                ramp.set(j, i, (rampCoeffs[0] + rampCoeffs[1]*x[j] + rampCoeffs[2]*y[i], 0x0, var))
#pybind11#
#pybind11#        return ramp, rampCoeffs, x, y
#pybind11#
#pybind11#    def testLinearRamp(self):
#pybind11#        """Fit a ramp"""
#pybind11#
#pybind11#        binsize = 1
#pybind11#        ramp, rampCoeffs, xVec, yVec = self.makeRamp(binsize)
#pybind11#        # Add a (labelled) bad value
#pybind11#        ramp.set(ramp.getWidth()//2, ramp.getHeight()//2, (0, 0x1, np.nan))
#pybind11#
#pybind11#        if display:
#pybind11#            ds9.mtv(ramp, title="Input", frame=0)
#pybind11#        # Here's the range that the approximation should be valid (and also the
#pybind11#        # bbox of the image returned by getImage)
#pybind11#        bbox = afwGeom.BoxI(afwGeom.PointI(0, 0), afwGeom.PointI(binsize*ramp.getWidth() - 1,
#pybind11#                                                                 binsize*ramp.getHeight() - 1))
#pybind11#
#pybind11#        order = 3                       # 1 would be enough to fit the ramp
#pybind11#        actrl = afwMath.ApproximateControl(afwMath.ApproximateControl.CHEBYSHEV, order)
#pybind11#        approx = afwMath.makeApproximate(xVec, yVec, ramp, bbox, actrl)
#pybind11#
#pybind11#        for i, aim in enumerate([approx.getImage(),
#pybind11#                                 approx.getMaskedImage().getImage(),
#pybind11#                                 ]):
#pybind11#            if i == 0 and display:
#pybind11#                ds9.mtv(aim, title="interpolated", frame=1)
#pybind11#                with ds9.Buffering():
#pybind11#                    for x in xVec:
#pybind11#                        for y in yVec:
#pybind11#                            ds9.dot('+', x, y, size=0.4, frame=1)
#pybind11#
#pybind11#            for x, y in aim.getBBox().getCorners():
#pybind11#                self.assertEqual(aim.get(x, y), rampCoeffs[0] + rampCoeffs[1]*x + rampCoeffs[1]*y)
#pybind11#
#pybind11#    def testChebyshevEqualOrder(self):
#pybind11#        """Check that we enforce the condition orderX == orderY"""
#pybind11#
#pybind11#        self.assertRaises(pexExcept.InvalidParameterError,
#pybind11#                          lambda:
#pybind11#                          afwMath.ApproximateControl(afwMath.ApproximateControl.CHEBYSHEV, 1, 2))
#pybind11#
#pybind11#    def testNoFinitePoints(self):
#pybind11#        """Check that makeApproximate throws a RuntimeError if grid has no finite points and weights to fit
#pybind11#        """
#pybind11#        binsize = 1
#pybind11#        for badValue in [(3, 0x1, 0), (np.nan, 0x1, 1)]:
#pybind11#            ramp, rampCoeffs, xVec, yVec = self.makeRamp(binsize)
#pybind11#            ramp.set(badValue)
#pybind11#            bbox = afwGeom.BoxI(afwGeom.PointI(0, 0), afwGeom.PointI(binsize*ramp.getWidth() - 1,
#pybind11#                                                                     binsize*ramp.getHeight() - 1))
#pybind11#            order = 2
#pybind11#            actrl = afwMath.ApproximateControl(afwMath.ApproximateControl.CHEBYSHEV, order)
#pybind11#            self.assertRaises(pexExcept.RuntimeError,
#pybind11#                              lambda: afwMath.makeApproximate(xVec, yVec, ramp, bbox, actrl))
#pybind11#
#pybind11#    def testLinearRampAsBackground(self):
#pybind11#        """Fit a ramp"""
#pybind11#
#pybind11#        ramp, rampCoeffs = self.makeRamp()[0:2]
#pybind11#
#pybind11#        if display:
#pybind11#            ds9.mtv(ramp, title="Input", frame=0)
#pybind11#        # Here's the range that the approximation should be valid (and also the
#pybind11#        # bbox of the image returned by getImage)
#pybind11#        bkgd = afwMath.makeBackground(ramp, afwMath.BackgroundControl(10, 10))
#pybind11#
#pybind11#        orderMax = 3                    # 1 would be enough to fit the ramp
#pybind11#        for order in range(orderMax + 1):
#pybind11#            actrl = afwMath.ApproximateControl(afwMath.ApproximateControl.CHEBYSHEV, order)
#pybind11#
#pybind11#            approx = bkgd.getApproximate(actrl)
#pybind11#            # Get the Image, the MaskedImage, and the Image with a truncated expansion
#pybind11#            for i, aim in enumerate([approx.getImage(),
#pybind11#                                     approx.getMaskedImage().getImage(),
#pybind11#                                     approx.getImage(order - 1 if order > 1 else -1),
#pybind11#                                     ]):
#pybind11#                if display and (i == 0 and order == 1):
#pybind11#                    ds9.mtv(aim, title="Interpolated", frame=1)
#pybind11#
#pybind11#                for x, y in aim.getBBox().getCorners():
#pybind11#                    val = np.mean(aim.getArray()) if order == 0 else \
#pybind11#                        rampCoeffs[0] + rampCoeffs[1]*x + rampCoeffs[1]*y
#pybind11#
#pybind11#                    self.assertEqual(aim.get(x, y), val)
#pybind11#        # Check that we can't "truncate" the expansion to a higher order than we requested
#pybind11#        self.assertRaises(pexExcept.InvalidParameterError,
#pybind11#                          lambda: approx.getImage(orderMax + 1, orderMax + 1))
#pybind11#
#pybind11#
#pybind11#class MemoryTester(lsst.utils.tests.MemoryTestCase):
#pybind11#    pass
#pybind11#
#pybind11#
#pybind11#def setup_module(module):
#pybind11#    lsst.utils.tests.init()
#pybind11#
#pybind11#
#pybind11#if __name__ == "__main__":
#pybind11#    lsst.utils.tests.init()
#pybind11#    unittest.main()
