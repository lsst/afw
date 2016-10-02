#pybind11##!/usr/bin/env python
#pybind11#from __future__ import absolute_import, division
#pybind11#from __future__ import print_function
#pybind11#from builtins import zip
#pybind11#from builtins import str
#pybind11#from builtins import range
#pybind11#from builtins import object
#pybind11#
#pybind11##
#pybind11## LSST Data Management System
#pybind11## Copyright 2008-2015 LSST Corporation.
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
#pybind11#
#pybind11#"""
#pybind11#Tests for Footprints, and FootprintSets
#pybind11#
#pybind11#Run with:
#pybind11#   footprint1.py
#pybind11#or
#pybind11#   python
#pybind11#   >>> import footprint1; footprint1.run()
#pybind11#"""
#pybind11#
#pybind11#import math
#pybind11#import sys
#pybind11#import unittest
#pybind11#import os
#pybind11#import numpy
#pybind11#import lsst.utils.tests
#pybind11#import lsst.afw.geom as afwGeom
#pybind11#import lsst.afw.geom.ellipses as afwGeomEllipses
#pybind11#import lsst.afw.coord as afwCoord
#pybind11#import lsst.afw.image as afwImage
#pybind11#import lsst.afw.math as afwMath
#pybind11#import lsst.afw.detection as afwDetect
#pybind11#import lsst.afw.detection.utils as afwDetectUtils
#pybind11#import lsst.afw.display.ds9 as ds9
#pybind11#import lsst.afw.display.utils as displayUtils
#pybind11#
#pybind11#try:
#pybind11#    type(display)
#pybind11#except NameError:
#pybind11#    display = False
#pybind11#
#pybind11#testPath = os.path.abspath(os.path.dirname(__file__))
#pybind11#
#pybind11#def toString(*args):
#pybind11#    """toString written in python"""
#pybind11#    if len(args) == 1:
#pybind11#        args = args[0]
#pybind11#
#pybind11#    y, x0, x1 = args
#pybind11#    return "%d: %d..%d" % (y, x0, x1)
#pybind11#
#pybind11#
#pybind11#class Object(object):
#pybind11#
#pybind11#    def __init__(self, val, spans):
#pybind11#        self.val = val
#pybind11#        self.spans = spans
#pybind11#
#pybind11#    def __str__(self):
#pybind11#        return ", ".join([str(s) for s in self.spans])
#pybind11#
#pybind11#    def insert(self, im):
#pybind11#        """Insert self into an image"""
#pybind11#        for sp in self.spans:
#pybind11#            y, x0, x1 = sp
#pybind11#            for x in range(x0, x1+1):
#pybind11#                im.set(x, y, self.val)
#pybind11#
#pybind11#    def __eq__(self, other):
#pybind11#        for osp, sp in zip(other.getSpans(), self.spans):
#pybind11#            if osp.toString() != toString(sp):
#pybind11#                return False
#pybind11#
#pybind11#        return True
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#
#pybind11#
#pybind11#class SpanTestCase(unittest.TestCase):
#pybind11#
#pybind11#    def testLessThan(self):
#pybind11#        span1 = afwDetect.Span(42, 0, 100)
#pybind11#        span2 = afwDetect.Span(41, 0, 100)
#pybind11#        span3 = afwDetect.Span(43, 0, 100)
#pybind11#        span4 = afwDetect.Span(42, -100, 100)
#pybind11#        span5 = afwDetect.Span(42, 100, 200)
#pybind11#        span6 = afwDetect.Span(42, 0, 10)
#pybind11#        span7 = afwDetect.Span(42, 0, 200)
#pybind11#        span8 = afwDetect.Span(42, 0, 100)
#pybind11#
#pybind11#        # Cannot use assertLess and friends here
#pybind11#        # because Span only has operator <
#pybind11#        def assertOrder(x1, x2):
#pybind11#            self.assertTrue(x1 < x2)
#pybind11#            self.assertFalse(x2 < x1)
#pybind11#
#pybind11#        assertOrder(span2, span1)
#pybind11#        assertOrder(span1, span3)
#pybind11#        assertOrder(span4, span1)
#pybind11#        assertOrder(span1, span5)
#pybind11#        assertOrder(span6, span1)
#pybind11#        assertOrder(span1, span7)
#pybind11#        self.assertFalse(span1 < span8)
#pybind11#        self.assertFalse(span8 < span1)
#pybind11#
#pybind11#
#pybind11#class ThresholdTestCase(unittest.TestCase):
#pybind11#
#pybind11#    def testThresholdFactory(self):
#pybind11#        """
#pybind11#        Test the creation of a Threshold object
#pybind11#
#pybind11#        This is a white-box test.
#pybind11#        -tests missing parameters
#pybind11#        -tests mal-formed parameters
#pybind11#        """
#pybind11#        try:
#pybind11#            afwDetect.createThreshold(3.4)
#pybind11#        except:
#pybind11#            self.fail("Failed to build Threshold with proper parameters")
#pybind11#
#pybind11#        try:
#pybind11#            afwDetect.createThreshold(3.4, "foo bar")
#pybind11#        except:
#pybind11#            pass
#pybind11#        else:
#pybind11#            self.fail("Threhold parameters not properly validated")
#pybind11#
#pybind11#        try:
#pybind11#            afwDetect.createThreshold(3.4, "variance")
#pybind11#        except:
#pybind11#            self.fail("Failed to build Threshold with proper parameters")
#pybind11#
#pybind11#        try:
#pybind11#            afwDetect.createThreshold(3.4, "stdev")
#pybind11#        except:
#pybind11#            self.fail("Failed to build Threshold with proper parameters")
#pybind11#
#pybind11#        try:
#pybind11#            afwDetect.createThreshold(3.4, "value")
#pybind11#        except:
#pybind11#            self.fail("Failed to build Threshold with proper parameters")
#pybind11#
#pybind11#        try:
#pybind11#            afwDetect.createThreshold(3.4, "value", False)
#pybind11#        except:
#pybind11#            self.fail("Failed to build Threshold with VALUE, False parameters")
#pybind11#
#pybind11#        try:
#pybind11#            afwDetect.createThreshold(0x4, "bitmask")
#pybind11#        except:
#pybind11#            self.fail("Failed to build Threshold with BITMASK parameters")
#pybind11#
#pybind11#        try:
#pybind11#            afwDetect.createThreshold(5, "pixel_stdev")
#pybind11#        except:
#pybind11#            self.fail("Failed to build Threshold with PIXEL_STDEV parameters")
#pybind11#
#pybind11#
#pybind11#class FootprintTestCase(lsst.utils.tests.TestCase):
#pybind11#    """A test case for Footprint"""
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        self.foot = afwDetect.Footprint()
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        del self.foot
#pybind11#
#pybind11#    def testToString(self):
#pybind11#        y, x0, x1 = 10, 100, 101
#pybind11#        s = afwDetect.Span(y, x0, x1)
#pybind11#        self.assertEqual(s.toString(), toString(y, x0, x1))
#pybind11#
#pybind11#    def testGC(self):
#pybind11#        """Check that Footprints are automatically garbage collected (when MemoryTestCase runs)"""
#pybind11#
#pybind11#        afwDetect.Footprint()
#pybind11#
#pybind11#    def testId(self):
#pybind11#        """Test uniqueness of IDs"""
#pybind11#        self.assertNotEqual(self.foot.getId(), afwDetect.Footprint().getId())
#pybind11#
#pybind11#    def testIntersectMask(self):
#pybind11#        bbox = afwGeom.BoxI(afwGeom.PointI(0, 0), afwGeom.ExtentI(10))
#pybind11#        fp = afwDetect.Footprint(bbox)
#pybind11#        maskBBox = afwGeom.BoxI(bbox)
#pybind11#        maskBBox.grow(-2)
#pybind11#        mask = afwImage.MaskU(maskBBox)
#pybind11#        innerBBox = afwGeom.BoxI(maskBBox)
#pybind11#        innerBBox.grow(-2)
#pybind11#        subMask = mask.Factory(mask, innerBBox)
#pybind11#        subMask.set(1)
#pybind11#
#pybind11#        fp.intersectMask(mask)
#pybind11#        fpBBox = fp.getBBox()
#pybind11#        self.assertEqual(fpBBox.getMinX(), maskBBox.getMinX())
#pybind11#        self.assertEqual(fpBBox.getMinY(), maskBBox.getMinY())
#pybind11#        self.assertEqual(fpBBox.getMaxX(), maskBBox.getMaxX())
#pybind11#        self.assertEqual(fpBBox.getMaxY(), maskBBox.getMaxY())
#pybind11#
#pybind11#        self.assertEqual(fp.getArea(), maskBBox.getArea() - innerBBox.getArea())
#pybind11#
#pybind11#    def testTablePersistence(self):
#pybind11#        ellipse = afwGeomEllipses.Ellipse(afwGeomEllipses.Axes(8, 6, 0.25), afwGeom.Point2D(9, 15))
#pybind11#        fp1 = afwDetect.Footprint(ellipse)
#pybind11#        fp1.addPeak(6, 7, 2)
#pybind11#        fp1.addPeak(8, 9, 3)
#pybind11#        with lsst.utils.tests.getTempFilePath(".fits") as tmpFile:
#pybind11#            fp1.writeFits(tmpFile)
#pybind11#            fp2 = afwDetect.Footprint.readFits(tmpFile)
#pybind11#            self.assertEqual(fp1.getArea(), fp2.getArea())
#pybind11#            self.assertEqual(list(fp1.getSpans()), list(fp2.getSpans()))
#pybind11#            # can't use Peak operator== for comparison because it compares IDs, not positions/values
#pybind11#            self.assertEqual(len(fp1.getPeaks()), len(fp2.getPeaks()))
#pybind11#            for peak1, peak2 in zip(fp1.getPeaks(), fp2.getPeaks()):
#pybind11#                self.assertEqual(peak1.getIx(), peak2.getIx())
#pybind11#                self.assertEqual(peak1.getIy(), peak2.getIy())
#pybind11#                self.assertEqual(peak1.getFx(), peak2.getFx())
#pybind11#                self.assertEqual(peak1.getFy(), peak2.getFy())
#pybind11#                self.assertEqual(peak1.getPeakValue(), peak2.getPeakValue())
#pybind11#
#pybind11#    def testAddSpans(self):
#pybind11#        """Add spans to a Footprint"""
#pybind11#        for y, x0, x1 in [(10, 100, 105), (11, 99, 104)]:
#pybind11#            self.foot.addSpan(y, x0, x1)
#pybind11#
#pybind11#        sp = self.foot.getSpans()
#pybind11#
#pybind11#        self.assertEqual(sp[-1].toString(), toString(y, x0, x1))
#pybind11#
#pybind11#    def testBbox(self):
#pybind11#        """Add Spans and check bounding box"""
#pybind11#        foot = afwDetect.Footprint()
#pybind11#        for y, x0, x1 in [(10, 100, 105),
#pybind11#                          (11, 99, 104)]:
#pybind11#            foot.addSpan(y, x0, x1)
#pybind11#
#pybind11#        bbox = foot.getBBox()
#pybind11#        self.assertEqual(bbox.getWidth(), 7)
#pybind11#        self.assertEqual(bbox.getHeight(), 2)
#pybind11#        self.assertEqual(bbox.getMinX(), 99)
#pybind11#        self.assertEqual(bbox.getMinY(), 10)
#pybind11#        self.assertEqual(bbox.getMaxX(), 105)
#pybind11#        self.assertEqual(bbox.getMaxY(), 11)
#pybind11#        # clip with a bbox that doesn't overlap at all
#pybind11#        bbox2 = afwGeom.Box2I(afwGeom.Point2I(5, 90), afwGeom.Extent2I(1, 2))
#pybind11#        foot.clipTo(bbox2)
#pybind11#        self.assertTrue(foot.getBBox().isEmpty())
#pybind11#        self.assertEqual(foot.getArea(), 0)
#pybind11#
#pybind11#    def testSpanShift(self):
#pybind11#        """Test our ability to shift spans"""
#pybind11#        span = afwDetect.Span(10, 100, 105)
#pybind11#        foot = afwDetect.Footprint()
#pybind11#
#pybind11#        foot.addSpan(span, 1, 2)
#pybind11#
#pybind11#        bbox = foot.getBBox()
#pybind11#        self.assertEqual(bbox.getWidth(), 6)
#pybind11#        self.assertEqual(bbox.getHeight(), 1)
#pybind11#        self.assertEqual(bbox.getMinX(), 101)
#pybind11#        self.assertEqual(bbox.getMinY(), 12)
#pybind11#        #
#pybind11#        # Shift that span using Span.shift
#pybind11#        #
#pybind11#        foot = afwDetect.Footprint()
#pybind11#        span.shift(-1, -2)
#pybind11#        foot.addSpan(span)
#pybind11#
#pybind11#        bbox = foot.getBBox()
#pybind11#        self.assertEqual(bbox.getWidth(), 6)
#pybind11#        self.assertEqual(bbox.getHeight(), 1)
#pybind11#        self.assertEqual(bbox.getMinX(), 99)
#pybind11#        self.assertEqual(bbox.getMinY(), 8)
#pybind11#
#pybind11#    def testFootprintFromBBox1(self):
#pybind11#        """Create a rectangular Footprint"""
#pybind11#        x0, y0, w, h = 9, 10, 7, 4
#pybind11#        foot = afwDetect.Footprint(afwGeom.Box2I(afwGeom.Point2I(x0, y0), afwGeom.Extent2I(w, h)))
#pybind11#
#pybind11#        bbox = foot.getBBox()
#pybind11#
#pybind11#        self.assertEqual(bbox.getWidth(), w)
#pybind11#        self.assertEqual(bbox.getHeight(), h)
#pybind11#        self.assertEqual(bbox.getMinX(), x0)
#pybind11#        self.assertEqual(bbox.getMinY(), y0)
#pybind11#        self.assertEqual(bbox.getMaxX(), x0 + w - 1)
#pybind11#        self.assertEqual(bbox.getMaxY(), y0 + h - 1)
#pybind11#
#pybind11#        if False:
#pybind11#            idImage = afwImage.ImageU(w, h)
#pybind11#            idImage.set(0)
#pybind11#            foot.insertIntoImage(idImage, foot.getId(), bbox)
#pybind11#            ds9.mtv(idImage, frame=2)
#pybind11#
#pybind11#    def testGetBBox(self):
#pybind11#        """Check that Footprint.getBBox() returns a copy"""
#pybind11#        x0, y0, w, h = 9, 10, 7, 4
#pybind11#        foot = afwDetect.Footprint(afwGeom.Box2I(afwGeom.Point2I(x0, y0), afwGeom.Extent2I(w, h)))
#pybind11#        bbox = foot.getBBox()
#pybind11#
#pybind11#        dx, dy = 10, 20
#pybind11#        bbox.shift(afwGeom.Extent2I(dx, dy))
#pybind11#
#pybind11#        self.assertEqual(bbox.getMinX(), x0 + dx)
#pybind11#        self.assertEqual(foot.getBBox().getMinX(), x0)
#pybind11#
#pybind11#    def testFootprintFromCircle(self):
#pybind11#        """Create an elliptical Footprint"""
#pybind11#        ellipse = afwGeomEllipses.Ellipse(afwGeomEllipses.Axes(6, 6, 0), afwGeom.Point2D(9, 15))
#pybind11#        foot = afwDetect.Footprint(ellipse, afwGeom.Box2I(afwGeom.Point2I(0, 0), afwGeom.Extent2I(20, 30)))
#pybind11#
#pybind11#        idImage = afwImage.ImageU(afwGeom.Extent2I(foot.getRegion().getWidth(), foot.getRegion().getHeight()))
#pybind11#        idImage.set(0)
#pybind11#
#pybind11#        foot.insertIntoImage(idImage, foot.getId())
#pybind11#
#pybind11#        if False:
#pybind11#            ds9.mtv(idImage, frame=2)
#pybind11#
#pybind11#    def testFootprintFromEllipse(self):
#pybind11#        """Create an elliptical Footprint"""
#pybind11#        cen = afwGeom.Point2D(23, 25)
#pybind11#        a, b, theta = 25, 15, 30
#pybind11#        ellipse = afwGeomEllipses.Ellipse(afwGeomEllipses.Axes(a, b, math.radians(theta)), cen)
#pybind11#        foot = afwDetect.Footprint(ellipse, afwGeom.Box2I(afwGeom.Point2I(0, 0), afwGeom.Extent2I(50, 60)))
#pybind11#
#pybind11#        idImage = afwImage.ImageU(afwGeom.Extent2I(foot.getRegion().getWidth(), foot.getRegion().getHeight()))
#pybind11#        idImage.set(0)
#pybind11#
#pybind11#        foot.insertIntoImage(idImage, foot.getId())
#pybind11#
#pybind11#        if display:
#pybind11#            ds9.mtv(idImage, frame=2)
#pybind11#            displayUtils.drawFootprint(foot, frame=2)
#pybind11#            shape = foot.getShape()
#pybind11#            shape.scale(2)              # <r^2> = 1/2 for a disk
#pybind11#            ds9.dot(shape, *cen, frame=2, ctype=ds9.RED)
#pybind11#
#pybind11#            shape = foot.getShape()
#pybind11#            shape.scale(2)              # <r^2> = 1/2 for a disk
#pybind11#            ds9.dot(shape, *cen, frame=2, ctype=ds9.MAGENTA)
#pybind11#
#pybind11#        axes = afwGeom.ellipses.Axes(foot.getShape())
#pybind11#        axes.scale(2)                   # <r^2> = 1/2 for a disk
#pybind11#
#pybind11#        self.assertEqual(foot.getCentroid(), cen)
#pybind11#        self.assertLess(abs(a - axes.getA()), 0.15, "a: %g v. %g" % (a, axes.getA()))
#pybind11#        self.assertLess(abs(b - axes.getB()), 0.02, "b: %g v. %g" % (b, axes.getB()))
#pybind11#        self.assertLess(abs(theta - math.degrees(axes.getTheta())), 0.2,
#pybind11#                        "theta: %g v. %g" % (theta, math.degrees(axes.getTheta())))
#pybind11#
#pybind11#    def testCopy(self):
#pybind11#        bbox = afwGeom.BoxI(afwGeom.PointI(0, 2), afwGeom.PointI(5, 6))
#pybind11#
#pybind11#        fp = afwDetect.Footprint(bbox, bbox)
#pybind11#
#pybind11#        # test copy construct
#pybind11#        fp2 = afwDetect.Footprint(fp)
#pybind11#
#pybind11#        self.assertEqual(fp2.getBBox(), bbox)
#pybind11#        self.assertEqual(fp2.getRegion(), bbox)
#pybind11#        self.assertEqual(fp2.getArea(), bbox.getArea())
#pybind11#        self.assertEqual(fp2.isNormalized(), True)
#pybind11#
#pybind11#        y = bbox.getMinY()
#pybind11#        for s in fp2.getSpans():
#pybind11#            self.assertEqual(s.getY(), y)
#pybind11#            self.assertEqual(s.getX0(), bbox.getMinX())
#pybind11#            self.assertEqual(s.getX1(), bbox.getMaxX())
#pybind11#            y += 1
#pybind11#
#pybind11#        # test assignment
#pybind11#        fp3 = afwDetect.Footprint()
#pybind11#        fp3.assign(fp)
#pybind11#        self.assertEqual(fp3.getBBox(), bbox)
#pybind11#        self.assertEqual(fp3.getRegion(), bbox)
#pybind11#        self.assertEqual(fp3.getArea(), bbox.getArea())
#pybind11#        self.assertEqual(fp3.isNormalized(), True)
#pybind11#
#pybind11#        y = bbox.getMinY()
#pybind11#        for s in fp3.getSpans():
#pybind11#            self.assertEqual(s.getY(), y)
#pybind11#            self.assertEqual(s.getX0(), bbox.getMinX())
#pybind11#            self.assertEqual(s.getX1(), bbox.getMaxX())
#pybind11#            y += 1
#pybind11#
#pybind11#    def testShrink(self):
#pybind11#        width, height = 5, 10  # Size of footprint
#pybind11#        x0, y0 = 50, 50  # Position of footprint
#pybind11#        imwidth, imheight = 100, 100  # Size of image
#pybind11#
#pybind11#        foot = afwDetect.Footprint(afwGeom.Box2I(afwGeom.Point2I(x0, y0), afwGeom.Extent2I(width, height)),
#pybind11#                                   afwGeom.Box2I(afwGeom.Point2I(0, 0), afwGeom.Extent2I(imwidth, imheight)))
#pybind11#        self.assertEqual(foot.getNpix(), width*height)
#pybind11#
#pybind11#        # Add some peaks to the original footprint and check that those lying outside
#pybind11#        # the shrunken footprint are omitted from the returned shrunken footprint.
#pybind11#        foot.addPeak(50, 50, 1)  # should be omitted in shrunken footprint
#pybind11#        foot.addPeak(52, 52, 2)  # should be kept in shrunken footprint
#pybind11#        foot.addPeak(50, 59, 3)  # should be omitted in shrunken footprint
#pybind11#        self.assertEqual(len(foot.getPeaks()), 3)  # check that all three peaks were added
#pybind11#
#pybind11#        # Shrinking by one pixel makes each dimension *two* pixels shorter.
#pybind11#        shrunk = afwDetect.shrinkFootprint(foot, 1, True)
#pybind11#        self.assertEqual(3*8, shrunk.getNpix())
#pybind11#
#pybind11#        # Shrunken footprint should now only contain one peak at (52, 52)
#pybind11#        self.assertEqual(len(shrunk.getPeaks()), 1)
#pybind11#        peak = shrunk.getPeaks()[0]
#pybind11#        self.assertEqual((peak.getIx(), peak.getIy()), (52, 52))
#pybind11#
#pybind11#        # Without shifting the centroid
#pybind11#        self.assertEqual(shrunk.getCentroid(), foot.getCentroid())
#pybind11#
#pybind11#        # Get the same result from a Manhattan shrink
#pybind11#        shrunk = afwDetect.shrinkFootprint(foot, 1, False)
#pybind11#        self.assertEqual(3*8, shrunk.getNpix())
#pybind11#        self.assertEqual(shrunk.getCentroid(), foot.getCentroid())
#pybind11#
#pybind11#        # Shrinking by a large amount leaves nothing.
#pybind11#        self.assertEqual(afwDetect.shrinkFootprint(foot, 100, True).getNpix(), 0)
#pybind11#
#pybind11#    def testShrinkIsoVsManhattan(self):
#pybind11#        # Demonstrate that isotropic and Manhattan shrinks are different.
#pybind11#        radius = 8
#pybind11#        imwidth, imheight = 100, 100
#pybind11#        x0, y0 = imwidth//2, imheight//2
#pybind11#        nshrink = 4
#pybind11#
#pybind11#        ellipse = afwGeomEllipses.Ellipse(afwGeomEllipses.Axes(1.5*radius, 2*radius, 0),
#pybind11#                                          afwGeom.Point2D(x0, y0))
#pybind11#        foot = afwDetect.Footprint(ellipse, afwGeom.Box2I(afwGeom.Point2I(0, 0),
#pybind11#                                                          afwGeom.Extent2I(imwidth, imheight)))
#pybind11#        self.assertNotEqual(afwDetect.shrinkFootprint(foot, nshrink, False),
#pybind11#                            afwDetect.shrinkFootprint(foot, nshrink, True))
#pybind11#
#pybind11#    def _fig8Test(self, x1, y1, x2, y2):
#pybind11#        # Construct a "figure of 8" consisting of two circles touching at the
#pybind11#        # centre of an image, then demonstrate that it shrinks correctly.
#pybind11#        # (Helper method for tests below.)
#pybind11#        radius = 3
#pybind11#        imwidth, imheight = 100, 100
#pybind11#        nshrink = 1
#pybind11#
#pybind11#        # These are the correct values for footprint sizes given the paramters
#pybind11#        # above.
#pybind11#        circle_npix = 29
#pybind11#        initial_npix = circle_npix * 2 - 1  # touch at one pixel
#pybind11#        shrunk_npix = 26
#pybind11#
#pybind11#        box = afwGeom.Box2I(afwGeom.Point2I(0, 0), afwGeom.Extent2I(imwidth, imheight))
#pybind11#
#pybind11#        e1 = afwGeomEllipses.Ellipse(afwGeomEllipses.Axes(radius, radius, 0),
#pybind11#                                     afwGeom.Point2D(x1, y1))
#pybind11#        f1 = afwDetect.Footprint(e1, box)
#pybind11#        self.assertEqual(f1.getNpix(), circle_npix)
#pybind11#
#pybind11#        e2 = afwGeomEllipses.Ellipse(afwGeomEllipses.Axes(radius, radius, 0),
#pybind11#                                     afwGeom.Point2D(x2, y2))
#pybind11#        f2 = afwDetect.Footprint(e2, box)
#pybind11#        self.assertEqual(f2.getNpix(), circle_npix)
#pybind11#
#pybind11#        initial = afwDetect.mergeFootprints(f1, f2)
#pybind11#        initial.setRegion(f2.getRegion())  # merge does not propagate the region
#pybind11#        self.assertEqual(initial_npix, initial.getNpix())
#pybind11#
#pybind11#        shrunk = afwDetect.shrinkFootprint(initial, nshrink, True)
#pybind11#        self.assertEqual(shrunk_npix, shrunk.getNpix())
#pybind11#
#pybind11#        if display:
#pybind11#            idImage = afwImage.ImageU(imwidth, imheight)
#pybind11#            for i, foot in enumerate([initial, shrunk]):
#pybind11#                print(foot.getNpix())
#pybind11#                foot.insertIntoImage(idImage, i+1)
#pybind11#            ds9.mtv(idImage)
#pybind11#
#pybind11#    def testShrinkEightVertical(self):
#pybind11#        # Test a "vertical" figure of 8.
#pybind11#        radius = 3
#pybind11#        imwidth, imheight = 100, 100
#pybind11#        self._fig8Test(imwidth//2, imheight//2-radius, imwidth//2, imheight//2+radius)
#pybind11#
#pybind11#    def testShrinkEightHorizontal(self):
#pybind11#        # Test a "horizontal" figure of 8.
#pybind11#        radius = 3
#pybind11#        imwidth, imheight = 100, 100
#pybind11#        self._fig8Test(imwidth//2-radius, imheight//2, imwidth//2+radius, imheight//2)
#pybind11#
#pybind11#    def testGrow(self):
#pybind11#        """Test growing a footprint"""
#pybind11#        x0, y0 = 20, 20
#pybind11#        width, height = 20, 30
#pybind11#        foot1 = afwDetect.Footprint(afwGeom.Box2I(afwGeom.Point2I(x0, y0), afwGeom.Extent2I(width, height)),
#pybind11#                                    afwGeom.Box2I(afwGeom.Point2I(0, 0), afwGeom.Extent2I(100, 100)))
#pybind11#
#pybind11#        # Add some peaks and check that they get copied into the new grown footprint
#pybind11#        foot1.addPeak(20, 20, 1)
#pybind11#        foot1.addPeak(30, 35, 2)
#pybind11#        foot1.addPeak(25, 45, 3)
#pybind11#        self.assertEqual(len(foot1.getPeaks()), 3)
#pybind11#
#pybind11#        bbox1 = foot1.getBBox()
#pybind11#
#pybind11#        self.assertEqual(bbox1.getMinX(), x0)
#pybind11#        self.assertEqual(bbox1.getMaxX(), x0 + width - 1)
#pybind11#        self.assertEqual(bbox1.getWidth(), width)
#pybind11#
#pybind11#        self.assertEqual(bbox1.getMinY(), y0)
#pybind11#        self.assertEqual(bbox1.getMaxY(), y0 + height - 1)
#pybind11#        self.assertEqual(bbox1.getHeight(), height)
#pybind11#
#pybind11#        ngrow = 5
#pybind11#        for isotropic in (True, False):
#pybind11#            foot2 = afwDetect.growFootprint(foot1, ngrow, isotropic)
#pybind11#
#pybind11#            # Check that the grown footprint is normalized
#pybind11#            self.assertTrue(foot2.isNormalized())
#pybind11#
#pybind11#            # Check that the grown footprint is bigger than the original
#pybind11#            self.assertGreater(foot2.getArea(), foot1.getArea())
#pybind11#
#pybind11#            # Check that peaks got copied into grown footprint
#pybind11#            self.assertEqual(len(foot2.getPeaks()), 3)
#pybind11#            for peak in foot2.getPeaks():
#pybind11#                self.assertIn((peak.getIx(), peak.getIy()), [(20, 20), (30, 35), (25, 45)])
#pybind11#
#pybind11#            bbox2 = foot2.getBBox()
#pybind11#
#pybind11#            if False and display:
#pybind11#                idImage = afwImage.ImageU(width, height)
#pybind11#                idImage.set(0)
#pybind11#
#pybind11#                i = 1
#pybind11#                for foot in [foot1, foot2]:
#pybind11#                    foot.insertIntoImage(idImage, i)
#pybind11#                    i += 1
#pybind11#
#pybind11#                metricImage = afwImage.ImageF("foo.fits")
#pybind11#                ds9.mtv(metricImage, frame=1)
#pybind11#                ds9.mtv(idImage)
#pybind11#
#pybind11#            # check bbox2
#pybind11#            self.assertEqual(bbox2.getMinX(), x0 - ngrow)
#pybind11#            self.assertEqual(bbox2.getWidth(), width + 2*ngrow)
#pybind11#
#pybind11#            self.assertEqual(bbox2.getMinY(), y0 - ngrow)
#pybind11#            self.assertEqual(bbox2.getHeight(), height + 2*ngrow)
#pybind11#            # Check that region was preserved
#pybind11#            self.assertEqual(foot1.getRegion(), foot2.getRegion())
#pybind11#
#pybind11#    def testFootprintToBBoxList(self):
#pybind11#        """Test footprintToBBoxList"""
#pybind11#        region = afwGeom.Box2I(afwGeom.Point2I(0, 0), afwGeom.Extent2I(12, 10))
#pybind11#        foot = afwDetect.Footprint(0, region)
#pybind11#        for y, x0, x1 in [(3, 3, 5), (3, 7, 7),
#pybind11#                          (4, 2, 3), (4, 5, 7),
#pybind11#                          (5, 2, 3), (5, 5, 8),
#pybind11#                          (6, 3, 5),
#pybind11#                          ]:
#pybind11#            foot.addSpan(y, x0, x1)
#pybind11#
#pybind11#        idImage = afwImage.ImageU(region.getDimensions())
#pybind11#        idImage.set(0)
#pybind11#
#pybind11#        foot.insertIntoImage(idImage, 1)
#pybind11#        if display:
#pybind11#            ds9.mtv(idImage)
#pybind11#
#pybind11#        idImageFromBBox = idImage.Factory(idImage, True)
#pybind11#        idImageFromBBox.set(0)
#pybind11#        bboxes = afwDetect.footprintToBBoxList(foot)
#pybind11#        for bbox in bboxes:
#pybind11#            x0, y0, x1, y1 = bbox.getMinX(), bbox.getMinY(), bbox.getMaxX(), bbox.getMaxY()
#pybind11#
#pybind11#            for y in range(y0, y1 + 1):
#pybind11#                for x in range(x0, x1 + 1):
#pybind11#                    idImageFromBBox.set(x, y, 1)
#pybind11#
#pybind11#            if display:
#pybind11#                x0 -= 0.5
#pybind11#                y0 -= 0.5
#pybind11#                x1 += 0.5
#pybind11#                y1 += 0.5
#pybind11#
#pybind11#                ds9.line([(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)], ctype=ds9.RED)
#pybind11#
#pybind11#        idImageFromBBox -= idImage      # should be blank
#pybind11#        stats = afwMath.makeStatistics(idImageFromBBox, afwMath.MAX)
#pybind11#
#pybind11#        self.assertEqual(stats.getValue(), 0)
#pybind11#
#pybind11#    def testWriteDefect(self):
#pybind11#        """Write a Footprint as a set of Defects"""
#pybind11#        region = afwGeom.Box2I(afwGeom.Point2I(0, 0), afwGeom.Extent2I(12, 10))
#pybind11#        foot = afwDetect.Footprint(0, region)
#pybind11#        for y, x0, x1 in [(3, 3, 5), (3, 7, 7),
#pybind11#                          (4, 2, 3), (4, 5, 7),
#pybind11#                          (5, 2, 3), (5, 5, 8),
#pybind11#                          (6, 3, 5),
#pybind11#                          ]:
#pybind11#            foot.addSpan(y, x0, x1)
#pybind11#
#pybind11#        if True:
#pybind11#            fd = open("/dev/null", "w")
#pybind11#        else:
#pybind11#            fd = sys.stdout
#pybind11#
#pybind11#        afwDetectUtils.writeFootprintAsDefects(fd, foot)
#pybind11#
#pybind11#    def testNormalize(self):
#pybind11#        """Test Footprint.normalize"""
#pybind11#        w, h = 12, 10
#pybind11#        region = afwGeom.Box2I(afwGeom.Point2I(0, 0), afwGeom.Extent2I(w, h))
#pybind11#        im = afwImage.ImageU(afwGeom.Extent2I(w, h))
#pybind11#        im.set(0)
#pybind11#        #
#pybind11#        # Create a footprint;  note that these Spans overlap
#pybind11#        #
#pybind11#        for spans, box in (([(3, 5, 6),
#pybind11#                             (4, 7, 7), ], afwGeom.Box2I(afwGeom.Point2I(5, 3), afwGeom.Point2I(7, 4))),
#pybind11#                           ([(3, 3, 5), (3, 6, 9),
#pybind11#                             (4, 2, 3), (4, 5, 7), (4, 8, 8),
#pybind11#                             (5, 2, 3), (5, 5, 8), (5, 6, 7),
#pybind11#                             (6, 3, 5),
#pybind11#                             ], afwGeom.Box2I(afwGeom.Point2I(2, 3), afwGeom.Point2I(9, 6)))
#pybind11#                           ):
#pybind11#
#pybind11#            foot = afwDetect.Footprint(0, region)
#pybind11#            for y, x0, x1 in spans:
#pybind11#                foot.addSpan(y, x0, x1)
#pybind11#
#pybind11#                for x in range(x0, x1 + 1):  # also insert into im
#pybind11#                    im.set(x, y, 1)
#pybind11#
#pybind11#            idImage = afwImage.ImageU(afwGeom.Extent2I(w, h))
#pybind11#            idImage.set(0)
#pybind11#
#pybind11#            foot.insertIntoImage(idImage, 1)
#pybind11#            if display:             # overlaping pixels will be > 1
#pybind11#                ds9.mtv(idImage)
#pybind11#            #
#pybind11#            # Normalise the Footprint, removing overlapping spans
#pybind11#            #
#pybind11#            foot.normalize()
#pybind11#
#pybind11#            idImage.set(0)
#pybind11#            foot.insertIntoImage(idImage, 1)
#pybind11#            if display:
#pybind11#                ds9.mtv(idImage, frame=1)
#pybind11#
#pybind11#            idImage -= im
#pybind11#
#pybind11#            self.assertEqual(box, foot.getBBox())
#pybind11#            self.assertEqual(afwMath.makeStatistics(idImage, afwMath.MAX).getValue(), 0)
#pybind11#
#pybind11#    def testSetFromFootprint(self):
#pybind11#        """Test setting mask/image pixels from a Footprint list"""
#pybind11#        mi = afwImage.MaskedImageF(afwGeom.Extent2I(12, 8))
#pybind11#        im = mi.getImage()
#pybind11#        #
#pybind11#        # Objects that we should detect
#pybind11#        #
#pybind11#        self.objects = []
#pybind11#        self.objects += [Object(10, [(1, 4, 4), (2, 3, 5), (3, 4, 4)])]
#pybind11#        self.objects += [Object(20, [(5, 7, 8), (5, 10, 10), (6, 8, 9)])]
#pybind11#        self.objects += [Object(20, [(6, 3, 3)])]
#pybind11#
#pybind11#        im.set(0)                       # clear image
#pybind11#        for obj in self.objects:
#pybind11#            obj.insert(im)
#pybind11#
#pybind11#        if False and display:
#pybind11#            ds9.mtv(mi, frame=0)
#pybind11#
#pybind11#        ds = afwDetect.FootprintSet(mi, afwDetect.Threshold(15))
#pybind11#
#pybind11#        objects = ds.getFootprints()
#pybind11#        afwDetect.setMaskFromFootprintList(mi.getMask(), objects, 0x1)
#pybind11#
#pybind11#        self.assertEqual(mi.getMask().get(4, 2), 0x0)
#pybind11#        self.assertEqual(mi.getMask().get(3, 6), 0x1)
#pybind11#
#pybind11#        self.assertEqual(mi.getImage().get(3, 6), 20)
#pybind11#        afwDetect.setImageFromFootprintList(mi.getImage(), objects, 5.0)
#pybind11#        self.assertEqual(mi.getImage().get(4, 2), 10)
#pybind11#        self.assertEqual(mi.getImage().get(3, 6), 5)
#pybind11#
#pybind11#        if display:
#pybind11#            ds9.mtv(mi, frame=1)
#pybind11#        #
#pybind11#        # Check Footprint.contains() while we are about it
#pybind11#        #
#pybind11#        self.assertTrue(objects[0].contains(afwGeom.Point2I(7, 5)))
#pybind11#        self.assertFalse(objects[0].contains(afwGeom.Point2I(10, 6)))
#pybind11#        self.assertFalse(objects[0].contains(afwGeom.Point2I(7, 6)))
#pybind11#        self.assertFalse(objects[0].contains(afwGeom.Point2I(4, 2)))
#pybind11#
#pybind11#        self.assertTrue(objects[1].contains(afwGeom.Point2I(3, 6)))
#pybind11#
#pybind11#    def testMakeFootprintSetXY0(self):
#pybind11#        """Test setting mask/image pixels from a Footprint list"""
#pybind11#        mi = afwImage.MaskedImageF(afwGeom.Extent2I(12, 8))
#pybind11#        im = mi.getImage()
#pybind11#        im.set(100)
#pybind11#
#pybind11#        mi.setXY0(afwGeom.PointI(2, 2))
#pybind11#        afwDetect.FootprintSet(mi, afwDetect.Threshold(1), "DETECTED")
#pybind11#
#pybind11#        bitmask = mi.getMask().getPlaneBitMask("DETECTED")
#pybind11#        for y in range(im.getHeight()):
#pybind11#            for x in range(im.getWidth()):
#pybind11#                self.assertEqual(mi.getMask().get(x, y), bitmask)
#pybind11#
#pybind11#    def testTransform(self):
#pybind11#        dims = afwGeom.Extent2I(512, 512)
#pybind11#        bbox = afwGeom.Box2I(afwGeom.Point2I(0, 0), dims)
#pybind11#        radius = 5
#pybind11#        offset = afwGeom.Extent2D(123, 456)
#pybind11#        crval = afwCoord.Coord(0*afwGeom.degrees, 0*afwGeom.degrees)
#pybind11#        crpix = afwGeom.Point2D(0, 0)
#pybind11#        cdMatrix = [1.0e-5, 0.0, 0.0, 1.0e-5]
#pybind11#        source = afwImage.makeWcs(crval, crpix, *cdMatrix)
#pybind11#        target = afwImage.makeWcs(crval, crpix + offset, *cdMatrix)
#pybind11#        fpSource = afwDetect.Footprint(afwGeom.Point2I(12, 34), radius, bbox)
#pybind11#
#pybind11#        fpTarget = fpSource.transform(source, target, bbox)
#pybind11#
#pybind11#        self.assertEqual(len(fpSource.getSpans()), len(fpTarget.getSpans()))
#pybind11#        self.assertEqual(fpSource.getNpix(), fpTarget.getNpix())
#pybind11#        self.assertEqual(fpSource.getArea(), fpTarget.getArea())
#pybind11#
#pybind11#        imSource = afwImage.ImageU(dims)
#pybind11#        fpSource.insertIntoImage(imSource, 1)
#pybind11#
#pybind11#        imTarget = afwImage.ImageU(dims)
#pybind11#        fpTarget.insertIntoImage(imTarget, 1)
#pybind11#
#pybind11#        subSource = imSource.Factory(imSource, fpSource.getBBox())
#pybind11#        subTarget = imTarget.Factory(imTarget, fpTarget.getBBox())
#pybind11#        self.assertTrue(numpy.all(subSource.getArray() == subTarget.getArray()))
#pybind11#
#pybind11#        # make a bbox smaller than the target footprint
#pybind11#        bbox2 = afwGeom.Box2I(fpTarget.getBBox())
#pybind11#        bbox2.grow(-1)
#pybind11#        fpTarget2 = fpSource.transform(source, target, bbox2)  # this one clips
#pybind11#        fpTarget3 = fpSource.transform(source, target, bbox2, False)  # this one doesn't
#pybind11#        self.assertTrue(bbox2.contains(fpTarget2.getBBox()))
#pybind11#        self.assertFalse(bbox2.contains(fpTarget3.getBBox()))
#pybind11#        self.assertNotEqual(fpTarget.getArea(), fpTarget2.getArea())
#pybind11#        self.assertEqual(fpTarget.getArea(), fpTarget3.getArea())
#pybind11#
#pybind11#    def testCopyWithinFootprintImage(self):
#pybind11#        W, H = 10, 10
#pybind11#        dims = afwGeom.Extent2I(W, H)
#pybind11#        source = afwImage.ImageF(dims)
#pybind11#        dest = afwImage.ImageF(dims)
#pybind11#        sa = source.getArray()
#pybind11#        for i in range(H):
#pybind11#            for j in range(W):
#pybind11#                sa[i, j] = 100 * i + j
#pybind11#
#pybind11#        self.foot.addSpan(4, 3, 6)
#pybind11#        self.foot.addSpan(5, 2, 4)
#pybind11#
#pybind11#        afwDetect.copyWithinFootprintImage(self.foot, source, dest)
#pybind11#
#pybind11#        da = dest.getArray()
#pybind11#        self.assertEqual(da[4, 2], 0)
#pybind11#        self.assertEqual(da[4, 3], 403)
#pybind11#        self.assertEqual(da[4, 4], 404)
#pybind11#        self.assertEqual(da[4, 5], 405)
#pybind11#        self.assertEqual(da[4, 6], 406)
#pybind11#        self.assertEqual(da[4, 7], 0)
#pybind11#        self.assertEqual(da[5, 1], 0)
#pybind11#        self.assertEqual(da[5, 2], 502)
#pybind11#        self.assertEqual(da[5, 3], 503)
#pybind11#        self.assertEqual(da[5, 4], 504)
#pybind11#        self.assertEqual(da[5, 5], 0)
#pybind11#        self.assertTrue(numpy.all(da[:4, :] == 0))
#pybind11#        self.assertTrue(numpy.all(da[6:, :] == 0))
#pybind11#
#pybind11#    def testCopyWithinFootprintOutside(self):
#pybind11#        """Copy a footprint that is larger than the image"""
#pybind11#        target = afwImage.ImageF(100, 100)
#pybind11#        target.set(0)
#pybind11#        subTarget = afwImage.ImageF(target, afwGeom.Box2I(afwGeom.Point2I(40, 40), afwGeom.Extent2I(20, 20)))
#pybind11#        source = afwImage.ImageF(10, 30)
#pybind11#        source.setXY0(45, 45)
#pybind11#        source.set(1.0)
#pybind11#
#pybind11#        foot = afwDetect.Footprint()
#pybind11#        foot.addSpan(50, 50, 60)  # Oversized on the source image, right; only some pixels overlap
#pybind11#        foot.addSpan(60, 0, 100)  # Oversized on the source, left and right; and on sub-target image, top
#pybind11#        foot.addSpan(99, 0, 1000)  # Oversized on the source image, top, left and right; aiming for segfault
#pybind11#
#pybind11#        afwDetect.copyWithinFootprintImage(foot, source, subTarget)
#pybind11#
#pybind11#        expected = numpy.zeros((100, 100))
#pybind11#        expected[50, 50:55] = 1.0
#pybind11#
#pybind11#        self.assertTrue(numpy.all(target.getArray() == expected))
#pybind11#
#pybind11#    def testCopyWithinFootprintMaskedImage(self):
#pybind11#        W, H = 10, 10
#pybind11#        dims = afwGeom.Extent2I(W, H)
#pybind11#        source = afwImage.MaskedImageF(dims)
#pybind11#        dest = afwImage.MaskedImageF(dims)
#pybind11#        sa = source.getImage().getArray()
#pybind11#        sv = source.getVariance().getArray()
#pybind11#        sm = source.getMask().getArray()
#pybind11#        for i in range(H):
#pybind11#            for j in range(W):
#pybind11#                sa[i, j] = 100 * i + j
#pybind11#                sv[i, j] = 100 * j + i
#pybind11#                sm[i, j] = 1
#pybind11#
#pybind11#        self.foot.addSpan(4, 3, 6)
#pybind11#        self.foot.addSpan(5, 2, 4)
#pybind11#
#pybind11#        afwDetect.copyWithinFootprintMaskedImage(self.foot, source, dest)
#pybind11#
#pybind11#        da = dest.getImage().getArray()
#pybind11#        dv = dest.getVariance().getArray()
#pybind11#        dm = dest.getMask().getArray()
#pybind11#
#pybind11#        self.assertEqual(da[4, 2], 0)
#pybind11#        self.assertEqual(da[4, 3], 403)
#pybind11#        self.assertEqual(da[4, 4], 404)
#pybind11#        self.assertEqual(da[4, 5], 405)
#pybind11#        self.assertEqual(da[4, 6], 406)
#pybind11#        self.assertEqual(da[4, 7], 0)
#pybind11#        self.assertEqual(da[5, 1], 0)
#pybind11#        self.assertEqual(da[5, 2], 502)
#pybind11#        self.assertEqual(da[5, 3], 503)
#pybind11#        self.assertEqual(da[5, 4], 504)
#pybind11#        self.assertEqual(da[5, 5], 0)
#pybind11#        self.assertTrue(numpy.all(da[:4, :] == 0))
#pybind11#        self.assertTrue(numpy.all(da[6:, :] == 0))
#pybind11#
#pybind11#        self.assertEqual(dv[4, 2], 0)
#pybind11#        self.assertEqual(dv[4, 3], 304)
#pybind11#        self.assertEqual(dv[4, 4], 404)
#pybind11#        self.assertEqual(dv[4, 5], 504)
#pybind11#        self.assertEqual(dv[4, 6], 604)
#pybind11#        self.assertEqual(dv[4, 7], 0)
#pybind11#        self.assertEqual(dv[5, 1], 0)
#pybind11#        self.assertEqual(dv[5, 2], 205)
#pybind11#        self.assertEqual(dv[5, 3], 305)
#pybind11#        self.assertEqual(dv[5, 4], 405)
#pybind11#        self.assertEqual(dv[5, 5], 0)
#pybind11#        self.assertTrue(numpy.all(dv[:4, :] == 0))
#pybind11#        self.assertTrue(numpy.all(dv[6:, :] == 0))
#pybind11#
#pybind11#        self.assertTrue(numpy.all(dm[4, 3:7] == 1))
#pybind11#        self.assertTrue(numpy.all(dm[5, 2:5] == 1))
#pybind11#        self.assertTrue(numpy.all(dm[:4, :] == 0))
#pybind11#        self.assertTrue(numpy.all(dm[6:, :] == 0))
#pybind11#        self.assertTrue(numpy.all(dm[4, :3] == 0))
#pybind11#        self.assertTrue(numpy.all(dm[4, 7:] == 0))
#pybind11#
#pybind11#    def testMergeFootprints(self):
#pybind11#        f1 = self.foot
#pybind11#        f2 = afwDetect.Footprint()
#pybind11#
#pybind11#        f1.addSpan(10, 10, 20)
#pybind11#        f1.addSpan(10, 30, 40)
#pybind11#        f1.addSpan(10, 50, 60)
#pybind11#
#pybind11#        f1.addSpan(11, 30, 50)
#pybind11#        f1.addSpan(12, 30, 50)
#pybind11#
#pybind11#        f1.addSpan(13, 10, 20)
#pybind11#        f1.addSpan(13, 30, 40)
#pybind11#        f1.addSpan(13, 50, 60)
#pybind11#
#pybind11#        f1.addSpan(15, 10, 20)
#pybind11#        f1.addSpan(15, 31, 40)
#pybind11#        f1.addSpan(15, 51, 60)
#pybind11#
#pybind11#        f2.addSpan(8, 10, 20)
#pybind11#        f2.addSpan(9, 20, 30)
#pybind11#        f2.addSpan(10, 0, 9)
#pybind11#        f2.addSpan(10, 35, 65)
#pybind11#        f2.addSpan(10, 70, 80)
#pybind11#
#pybind11#        f2.addSpan(13, 49, 54)
#pybind11#        f2.addSpan(14, 10, 30)
#pybind11#
#pybind11#        f2.addSpan(15, 21, 30)
#pybind11#        f2.addSpan(15, 41, 50)
#pybind11#        f2.addSpan(15, 61, 70)
#pybind11#
#pybind11#        f1.normalize()
#pybind11#        f2.normalize()
#pybind11#
#pybind11#        fA = afwDetect.mergeFootprints(f1, f2)
#pybind11#        fB = afwDetect.mergeFootprints(f2, f1)
#pybind11#
#pybind11#        ims = []
#pybind11#        for i, f in enumerate([f1, f2, fA, fB]):
#pybind11#            im1 = afwImage.ImageU(100, 100)
#pybind11#            im1.set(0)
#pybind11#            imbb = im1.getBBox()
#pybind11#            f.setRegion(imbb)
#pybind11#            f.insertIntoImage(im1, 1)
#pybind11#            ims.append(im1)
#pybind11#
#pybind11#        for i, merged in enumerate([ims[2], ims[3]]):
#pybind11#            m = merged.getArray()
#pybind11#            a1 = ims[0].getArray()
#pybind11#            a2 = ims[1].getArray()
#pybind11#            # Slightly looser tests to start...
#pybind11#            # Every pixel in f1 is in f[AB]
#pybind11#            self.assertTrue(numpy.all(m.flat[numpy.flatnonzero(a1)] == 1))
#pybind11#            # Every pixel in f2 is in f[AB]
#pybind11#            self.assertTrue(numpy.all(m.flat[numpy.flatnonzero(a2)] == 1))
#pybind11#            # merged == a1 | a2.
#pybind11#            self.assertTrue(numpy.all(m == numpy.maximum(a1, a2)))
#pybind11#
#pybind11#        if False:
#pybind11#            import matplotlib
#pybind11#            matplotlib.use('Agg')
#pybind11#            import pylab as plt
#pybind11#            plt.clf()
#pybind11#            for i, im1 in enumerate(ims):
#pybind11#                plt.subplot(4, 1, i+1)
#pybind11#                plt.imshow(im1.getArray(), interpolation='nearest', origin='lower')
#pybind11#                plt.axis([0, 100, 0, 20])
#pybind11#            plt.savefig('merge2.png')
#pybind11#
#pybind11#    def testPeakSort(self):
#pybind11#        footprint = afwDetect.Footprint(afwGeom.Box2I(afwGeom.Point2I(0, 0), afwGeom.Point2I(10, 10)))
#pybind11#        footprint.addPeak(4, 5, 1)
#pybind11#        footprint.addPeak(3, 2, 5)
#pybind11#        footprint.addPeak(7, 8, -2)
#pybind11#        footprint.addPeak(5, 7, 4)
#pybind11#        footprint.sortPeaks()
#pybind11#        self.assertEqual([peak.getIx() for peak in footprint.getPeaks()],
#pybind11#                         [3, 5, 4, 7])
#pybind11#
#pybind11#    def testClipToNonzero(self):
#pybind11#        # create a circular footprint
#pybind11#        ellipse = afwGeomEllipses.Ellipse(afwGeomEllipses.Axes(6, 6, 0), afwGeom.Point2D(9, 15))
#pybind11#        bb = afwGeom.Box2I(afwGeom.Point2I(0, 0), afwGeom.Extent2I(20, 30))
#pybind11#        foot = afwDetect.Footprint(ellipse, bb)
#pybind11#
#pybind11#        a0 = foot.getArea()
#pybind11#
#pybind11#        plots = False
#pybind11#        if plots:
#pybind11#            import matplotlib
#pybind11#            matplotlib.use('Agg')
#pybind11#            import pylab as plt
#pybind11#
#pybind11#            plt.clf()
#pybind11#            img = afwImage.ImageU(bb)
#pybind11#            foot.insertIntoImage(img, 1)
#pybind11#            ima = dict(interpolation='nearest', origin='lower', cmap='gray')
#pybind11#            plt.imshow(img.getArray(), **ima)
#pybind11#            plt.savefig('clipnz1.png')
#pybind11#
#pybind11#        source = afwImage.ImageF(bb)
#pybind11#        source.getArray()[:, :] = 1.
#pybind11#        source.getArray()[:, 0:10] = 0.
#pybind11#
#pybind11#        foot.clipToNonzero(source)
#pybind11#        foot.normalize()
#pybind11#        a1 = foot.getArea()
#pybind11#        self.assertLess(a1, a0)
#pybind11#
#pybind11#        img = afwImage.ImageU(bb)
#pybind11#        foot.insertIntoImage(img, 1)
#pybind11#        self.assertTrue(numpy.all(img.getArray()[source.getArray() == 0] == 0))
#pybind11#
#pybind11#        if plots:
#pybind11#            plt.clf()
#pybind11#            plt.subplot(1, 2, 1)
#pybind11#            plt.imshow(source.getArray(), **ima)
#pybind11#            plt.subplot(1, 2, 2)
#pybind11#            plt.imshow(img.getArray(), **ima)
#pybind11#            plt.savefig('clipnz2.png')
#pybind11#
#pybind11#        source.getArray()[:12, :] = 0.
#pybind11#        foot.clipToNonzero(source)
#pybind11#        foot.normalize()
#pybind11#
#pybind11#        a2 = foot.getArea()
#pybind11#        self.assertLess(a2, a1)
#pybind11#
#pybind11#        img = afwImage.ImageU(bb)
#pybind11#        foot.insertIntoImage(img, 1)
#pybind11#        self.assertTrue(numpy.all(img.getArray()[source.getArray() == 0] == 0))
#pybind11#
#pybind11#        if plots:
#pybind11#            plt.clf()
#pybind11#            plt.subplot(1, 2, 1)
#pybind11#            plt.imshow(source.getArray(), **ima)
#pybind11#            plt.subplot(1, 2, 2)
#pybind11#            img = afwImage.ImageU(bb)
#pybind11#            foot.insertIntoImage(img, 1)
#pybind11#            plt.imshow(img.getArray(), **ima)
#pybind11#            plt.savefig('clipnz3.png')
#pybind11#
#pybind11#    def testInclude(self):
#pybind11#        """Test that we can expand a Footprint to include the union of itself and all others provided."""
#pybind11#        region = afwGeom.Box2I(afwGeom.Point2I(-6, -6), afwGeom.Point2I(6, 6))
#pybind11#        parent = afwDetect.Footprint(afwGeom.Box2I(afwGeom.Point2I(-2, -2), afwGeom.Point2I(2, 2)), region)
#pybind11#        parent.addPeak(0, 0, float("NaN"))
#pybind11#        child1 = afwDetect.Footprint(afwGeom.Box2I(afwGeom.Point2I(-3, 0), afwGeom.Point2I(0, 3)), region)
#pybind11#        child1.addPeak(-1, 1, float("NaN"))
#pybind11#        child2 = afwDetect.Footprint(afwGeom.Box2I(afwGeom.Point2I(-4, -3), afwGeom.Point2I(-1, 0)), region)
#pybind11#        child3 = afwDetect.Footprint(afwGeom.Box2I(afwGeom.Point2I(4, -1), afwGeom.Point2I(6, 1)))
#pybind11#        merge123 = afwDetect.Footprint(parent)
#pybind11#        merge123.include([child1, child2, child3])
#pybind11#        self.assertTrue(merge123.getBBox().contains(parent.getBBox()))
#pybind11#        self.assertTrue(merge123.getBBox().contains(child1.getBBox()))
#pybind11#        self.assertTrue(merge123.getBBox().contains(child2.getBBox()))
#pybind11#        self.assertTrue(merge123.getBBox().contains(child3.getBBox()))
#pybind11#        mask123a = afwImage.MaskU(region)
#pybind11#        mask123b = afwImage.MaskU(region)
#pybind11#        afwDetect.setMaskFromFootprint(mask123a, parent, 1)
#pybind11#        afwDetect.setMaskFromFootprint(mask123a, child1, 1)
#pybind11#        afwDetect.setMaskFromFootprint(mask123a, child2, 1)
#pybind11#        afwDetect.setMaskFromFootprint(mask123a, child3, 1)
#pybind11#        afwDetect.setMaskFromFootprint(mask123b, merge123, 1)
#pybind11#        self.assertEqual(mask123a.getArray().sum(), merge123.getArea())
#pybind11#        self.assertClose(mask123a.getArray(), mask123b.getArray(), rtol=0, atol=0)
#pybind11#
#pybind11#        # Test that ignoreSelf=True works for include
#pybind11#        ignoreParent = True
#pybind11#        childOnly = afwDetect.Footprint()
#pybind11#        childOnly.include([child1, child2, child3])
#pybind11#        merge123 = afwDetect.Footprint(parent)
#pybind11#        merge123.include([child1, child2, child3], ignoreParent)
#pybind11#        maskChildren = afwImage.MaskU(region)
#pybind11#        mask123 = afwImage.MaskU(region)
#pybind11#        afwDetect.setMaskFromFootprint(maskChildren, childOnly, 1)
#pybind11#        afwDetect.setMaskFromFootprint(mask123, merge123, 1)
#pybind11#        self.assertTrue(numpy.all(maskChildren.getArray() == mask123.getArray()))
#pybind11#
#pybind11#    def checkEdge(self, footprint):
#pybind11#        """Check that Footprint::findEdgePixels() works"""
#pybind11#        bbox = footprint.getBBox()
#pybind11#        bbox.grow(3)
#pybind11#
#pybind11#        def makeImage(footprint):
#pybind11#            """Make an ImageF with 1 in the footprint, and 0 elsewhere"""
#pybind11#            ones = afwImage.ImageI(bbox)
#pybind11#            ones.set(1)
#pybind11#            image = afwImage.ImageI(bbox)
#pybind11#            image.set(0)
#pybind11#            afwDetect.copyWithinFootprintImage(footprint, ones, image)
#pybind11#            return image
#pybind11#
#pybind11#        edges = self.foot.findEdgePixels()
#pybind11#        edgeImage = makeImage(edges)
#pybind11#
#pybind11#        # Find edges with an edge-detection kernel
#pybind11#        image = makeImage(self.foot)
#pybind11#        kernel = afwImage.ImageD(3, 3)
#pybind11#        kernel.set(1, 1, 4)
#pybind11#        for x, y in [(1, 2), (0, 1), (1, 0), (2, 1)]:
#pybind11#            kernel.set(x, y, -1)
#pybind11#        kernel.setXY0(1, 1)
#pybind11#        result = afwImage.ImageI(bbox)
#pybind11#        result.set(0)
#pybind11#        afwMath.convolve(result, image, afwMath.FixedKernel(kernel), afwMath.ConvolutionControl(False))
#pybind11#        result.getArray().__imul__(image.getArray())
#pybind11#        trueEdges = numpy.where(result.getArray() > 0, 1, 0)
#pybind11#
#pybind11#        self.assertTrue(numpy.all(trueEdges == edgeImage.getArray()))
#pybind11#
#pybind11#    def testEdge(self):
#pybind11#        """Test for Footprint::findEdgePixels()"""
#pybind11#        foot = afwDetect.Footprint()
#pybind11#        for span in ((3, 3, 9),
#pybind11#                     (4, 2, 4),
#pybind11#                     (4, 6, 7),
#pybind11#                     (4, 9, 11),
#pybind11#                     (5, 3, 9),
#pybind11#                     (6, 6, 7),
#pybind11#                     ):
#pybind11#            foot.addSpanInSeries(*span)
#pybind11#        foot.normalize()
#pybind11#        self.checkEdge(foot)
#pybind11#
#pybind11#        # This footprint came from a very large Footprint in a deep HSC coadd patch
#pybind11#        self.checkEdge(afwDetect.Footprint.readFits(os.path.join(testPath,"testFootprintEdge.fits")))
#pybind11#
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#
#pybind11#class FootprintSetTestCase(unittest.TestCase):
#pybind11#    """A test case for FootprintSet"""
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        self.ms = afwImage.MaskedImageF(afwGeom.Extent2I(12, 8))
#pybind11#        im = self.ms.getImage()
#pybind11#        #
#pybind11#        # Objects that we should detect
#pybind11#        #
#pybind11#        self.objects = []
#pybind11#        self.objects += [Object(10, [(1, 4, 4), (2, 3, 5), (3, 4, 4)])]
#pybind11#        self.objects += [Object(20, [(5, 7, 8), (5, 10, 10), (6, 8, 9)])]
#pybind11#        self.objects += [Object(20, [(6, 3, 3)])]
#pybind11#
#pybind11#        self.ms.set((0, 0x0, 4.0))      # clear image; set variance
#pybind11#        for obj in self.objects:
#pybind11#            obj.insert(im)
#pybind11#
#pybind11#        if False and display:
#pybind11#            ds9.mtv(im, frame=0)
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        del self.ms
#pybind11#
#pybind11#    def testGC(self):
#pybind11#        """Check that FootprintSets are automatically garbage collected (when MemoryTestCase runs)"""
#pybind11#        afwDetect.FootprintSet(afwImage.MaskedImageF(afwGeom.Extent2I(10, 20)), afwDetect.Threshold(10))
#pybind11#
#pybind11#    def testFootprints(self):
#pybind11#        """Check that we found the correct number of objects and that they are correct"""
#pybind11#        ds = afwDetect.FootprintSet(self.ms, afwDetect.Threshold(10))
#pybind11#
#pybind11#        objects = ds.getFootprints()
#pybind11#
#pybind11#        self.assertEqual(len(objects), len(self.objects))
#pybind11#        for i in range(len(objects)):
#pybind11#            self.assertEqual(objects[i], self.objects[i])
#pybind11#
#pybind11#    def testFootprints2(self):
#pybind11#        """Check that we found the correct number of objects using FootprintSet"""
#pybind11#        ds = afwDetect.FootprintSet(self.ms, afwDetect.Threshold(10))
#pybind11#
#pybind11#        objects = ds.getFootprints()
#pybind11#
#pybind11#        self.assertEqual(len(objects), len(self.objects))
#pybind11#        for i in range(len(objects)):
#pybind11#            self.assertEqual(objects[i], self.objects[i])
#pybind11#
#pybind11#    def testFootprints3(self):
#pybind11#        """Check that we found the correct number of objects using FootprintSet and PIXEL_STDEV"""
#pybind11#        threshold = 4.5                 # in units of sigma
#pybind11#
#pybind11#        self.ms.set(2, 4, (10, 0x0, 36))  # not detected (high variance)
#pybind11#
#pybind11#        y, x = self.objects[2].spans[0][0:2]
#pybind11#        self.ms.set(x, y, (threshold, 0x0, 1.0))
#pybind11#
#pybind11#        ds = afwDetect.FootprintSet(self.ms,
#pybind11#                                    afwDetect.createThreshold(threshold, "pixel_stdev"), "OBJECT")
#pybind11#
#pybind11#        objects = ds.getFootprints()
#pybind11#
#pybind11#        self.assertEqual(len(objects), len(self.objects))
#pybind11#        for i in range(len(objects)):
#pybind11#            self.assertEqual(objects[i], self.objects[i])
#pybind11#
#pybind11#    def testFootprintsMasks(self):
#pybind11#        """Check that detectionSets have the proper mask bits set"""
#pybind11#        ds = afwDetect.FootprintSet(self.ms, afwDetect.Threshold(10), "OBJECT")
#pybind11#        objects = ds.getFootprints()
#pybind11#
#pybind11#        if display:
#pybind11#            ds9.mtv(self.ms, frame=1)
#pybind11#
#pybind11#        mask = self.ms.getMask()
#pybind11#        for i in range(len(objects)):
#pybind11#            for sp in objects[i].getSpans():
#pybind11#                for x in range(sp.getX0(), sp.getX1() + 1):
#pybind11#                    self.assertEqual(mask.get(x, sp.getY()), mask.getPlaneBitMask("OBJECT"))
#pybind11#
#pybind11#    def testFootprintsImageId(self):
#pybind11#        """Check that we can insert footprints into an Image"""
#pybind11#        ds = afwDetect.FootprintSet(self.ms, afwDetect.Threshold(10))
#pybind11#        objects = ds.getFootprints()
#pybind11#
#pybind11#        idImage = afwImage.ImageU(self.ms.getDimensions())
#pybind11#        idImage.set(0)
#pybind11#
#pybind11#        for foot in objects:
#pybind11#            foot.insertIntoImage(idImage, foot.getId())
#pybind11#
#pybind11#        if False:
#pybind11#            ds9.mtv(idImage, frame=2)
#pybind11#
#pybind11#        for i in range(len(objects)):
#pybind11#            for sp in objects[i].getSpans():
#pybind11#                for x in range(sp.getX0(), sp.getX1() + 1):
#pybind11#                    self.assertEqual(idImage.get(x, sp.getY()), objects[i].getId())
#pybind11#
#pybind11#    def testFootprintSetImageId(self):
#pybind11#        """Check that we can insert a FootprintSet into an Image, setting relative IDs"""
#pybind11#        ds = afwDetect.FootprintSet(self.ms, afwDetect.Threshold(10))
#pybind11#        objects = ds.getFootprints()
#pybind11#
#pybind11#        idImage = ds.insertIntoImage(True)
#pybind11#        if display:
#pybind11#            ds9.mtv(idImage, frame=2)
#pybind11#
#pybind11#        for i in range(len(objects)):
#pybind11#            for sp in objects[i].getSpans():
#pybind11#                for x in range(sp.getX0(), sp.getX1() + 1):
#pybind11#                    self.assertEqual(idImage.get(x, sp.getY()), i + 1)
#pybind11#
#pybind11#    def testFootprintsImage(self):
#pybind11#        """Check that we can search Images as well as MaskedImages"""
#pybind11#        ds = afwDetect.FootprintSet(self.ms.getImage(), afwDetect.Threshold(10))
#pybind11#
#pybind11#        objects = ds.getFootprints()
#pybind11#
#pybind11#        self.assertEqual(len(objects), len(self.objects))
#pybind11#        for i in range(len(objects)):
#pybind11#            self.assertEqual(objects[i], self.objects[i])
#pybind11#
#pybind11#    def testGrow2(self):
#pybind11#        """Grow some more interesting shaped Footprints.  Informative with display, but no numerical tests"""
#pybind11#        ds = afwDetect.FootprintSet(self.ms, afwDetect.Threshold(10), "OBJECT")
#pybind11#
#pybind11#        idImage = afwImage.ImageU(self.ms.getDimensions())
#pybind11#        idImage.set(0)
#pybind11#
#pybind11#        i = 1
#pybind11#        for foot in ds.getFootprints()[0:1]:
#pybind11#            gfoot = afwDetect.growFootprint(foot, 3, False)
#pybind11#            gfoot.insertIntoImage(idImage, i)
#pybind11#            i += 1
#pybind11#
#pybind11#        if display:
#pybind11#            ds9.mtv(self.ms, frame=0)
#pybind11#            ds9.mtv(idImage, frame=1)
#pybind11#
#pybind11#    def testFootprintPeaks(self):
#pybind11#        """Test that we can extract the peaks from a Footprint"""
#pybind11#        fs = afwDetect.FootprintSet(self.ms, afwDetect.Threshold(10), "OBJECT")
#pybind11#
#pybind11#        foot = fs.getFootprints()[0]
#pybind11#
#pybind11#        self.assertEqual(len(foot.getPeaks()), 5)
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#
#pybind11#
#pybind11#class MaskFootprintSetTestCase(unittest.TestCase):
#pybind11#    """A test case for generating FootprintSet from Masks"""
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        self.mim = afwImage.MaskedImageF(afwGeom.ExtentI(12, 8))
#pybind11#        #
#pybind11#        # Objects that we should detect
#pybind11#        #
#pybind11#        self.objects = []
#pybind11#        self.objects += [Object(0x2, [(1, 4, 4), (2, 3, 5), (3, 4, 4)])]
#pybind11#        self.objects += [Object(0x41, [(5, 7, 8), (6, 8, 8)])]
#pybind11#        self.objects += [Object(0x42, [(5, 10, 10)])]
#pybind11#        self.objects += [Object(0x82, [(6, 3, 3)])]
#pybind11#
#pybind11#        self.mim.set((0, 0, 0))                 # clear image
#pybind11#        for obj in self.objects:
#pybind11#            obj.insert(self.mim.getImage())
#pybind11#            obj.insert(self.mim.getMask())
#pybind11#
#pybind11#        if display:
#pybind11#            ds9.mtv(self.mim, frame=0)
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        del self.mim
#pybind11#
#pybind11#    def testFootprints(self):
#pybind11#        """Check that we found the correct number of objects using FootprintSet"""
#pybind11#        level = 0x2
#pybind11#        ds = afwDetect.FootprintSet(self.mim.getMask(), afwDetect.createThreshold(level, "bitmask"))
#pybind11#
#pybind11#        objects = ds.getFootprints()
#pybind11#
#pybind11#        if 0 and display:
#pybind11#            ds9.mtv(self.mim, frame=0)
#pybind11#
#pybind11#        self.assertEqual(len(objects), len([o for o in self.objects if (o.val & level)]))
#pybind11#
#pybind11#        i = 0
#pybind11#        for o in self.objects:
#pybind11#            if o.val & level:
#pybind11#                self.assertEqual(o, objects[i])
#pybind11#                i += 1
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#
#pybind11#
#pybind11#class NaNFootprintSetTestCase(unittest.TestCase):
#pybind11#    """A test case for FootprintSet when the image contains NaNs"""
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        self.ms = afwImage.MaskedImageF(afwGeom.Extent2I(12, 8))
#pybind11#        im = self.ms.getImage()
#pybind11#        #
#pybind11#        # Objects that we should detect
#pybind11#        #
#pybind11#        self.objects = []
#pybind11#        self.objects += [Object(10, [(1, 4, 4), (2, 3, 5), (3, 4, 4)])]
#pybind11#        self.objects += [Object(20, [(5, 7, 8), (6, 8, 8)])]
#pybind11#        self.objects += [Object(20, [(5, 10, 10)])]
#pybind11#        self.objects += [Object(30, [(6, 3, 3)])]
#pybind11#
#pybind11#        im.set(0)                       # clear image
#pybind11#        for obj in self.objects:
#pybind11#            obj.insert(im)
#pybind11#
#pybind11#        self.NaN = float("NaN")
#pybind11#        im.set(3, 7, self.NaN)
#pybind11#        im.set(0, 0, self.NaN)
#pybind11#        im.set(8, 2, self.NaN)
#pybind11#
#pybind11#        im.set(9, 6, self.NaN)          # connects the two objects with value==20 together if NaN is detected
#pybind11#
#pybind11#        if False and display:
#pybind11#            ds9.mtv(im, frame=0)
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        del self.ms
#pybind11#
#pybind11#    def testFootprints(self):
#pybind11#        """Check that we found the correct number of objects using FootprintSet"""
#pybind11#        ds = afwDetect.FootprintSet(self.ms, afwDetect.Threshold(10), "DETECTED")
#pybind11#
#pybind11#        objects = ds.getFootprints()
#pybind11#
#pybind11#        if display:
#pybind11#            ds9.mtv(self.ms, frame=0)
#pybind11#
#pybind11#        self.assertEqual(len(objects), len(self.objects))
#pybind11#        for i in range(len(objects)):
#pybind11#            self.assertEqual(objects[i], self.objects[i])
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
