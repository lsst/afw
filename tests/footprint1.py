#!/usr/bin/env python2
from __future__ import absolute_import, division

# 
# LSST Data Management System
# Copyright 2008-2015 LSST Corporation.
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
Tests for Footprints, and FootprintSets

Run with:
   footprint1.py
or
   python
   >>> import footprint1; footprint1.run()
"""

import math, sys
import unittest
import numpy
import lsst.utils.tests as utilsTests
import lsst.pex.logging as logging
import lsst.pex.exceptions as pexExcept
import lsst.afw.geom as afwGeom
import lsst.afw.geom.ellipses as afwGeomEllipses
import lsst.afw.coord as afwCoord
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.detection as afwDetect
import lsst.afw.detection.utils as afwDetectUtils
import lsst.afw.display.ds9 as ds9
import lsst.afw.display.utils as displayUtils

try:
    type(verbose)
except NameError:
    verbose = 0
    logging.Debug("afwDetect.Footprint", verbose)

try:
    type(display)
except NameError:
    display = False

def toString(*args):
    """toString written in python"""
    if len(args) == 1:
        args = args[0]

    y, x0, x1 = args
    return "%d: %d..%d" % (y, x0, x1)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class Object(object):
    def __init__(self, val, spans):
        self.val = val
        self.spans = spans

    def __str__(self):
        return ", ".join([str(s) for s in self.spans])

    def insert(self, im):
        """Insert self into an image"""
        for sp in self.spans:
            y, x0, x1 = sp
            for x in range(x0, x1+1):
                im.set(x, y, self.val)

    def __eq__(self, other):
        for osp, sp in zip(other.getSpans(), self.spans):
            if osp.toString() != toString(sp):
                return False

        return True

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class SpanTestCase(unittest.TestCase):
    def testLessThan(self):
        span1 = afwDetect.Span(42, 0, 100);
        span2 = afwDetect.Span(41, 0, 100);
        span3 = afwDetect.Span(43, 0, 100);
        span4 = afwDetect.Span(42, -100, 100);
        span5 = afwDetect.Span(42, 100, 200);
        span6 = afwDetect.Span(42, 0, 10);
        span7 = afwDetect.Span(42, 0, 200);
        span8 = afwDetect.Span(42, 0, 100);

        def assertOrder(x1, x2):
            self.assertTrue(x1 < x2)
            self.assertFalse(x2 < x1)

        assertOrder(span2, span1)
        assertOrder(span1, span3)
        assertOrder(span4, span1)
        assertOrder(span1, span5)
        assertOrder(span6, span1)
        assertOrder(span1, span7)
        self.assertFalse(span1 < span8)
        self.assertFalse(span8 < span1)


class ThresholdTestCase(unittest.TestCase):
    def testThresholdFactory(self):
        """
        Test the creation of a Threshold object

        This is a white-box test.
        -tests missing parameters
        -tests mal-formed parameters
        """
        try:
            afwDetect.createThreshold(3.4)
        except:
            self.fail("Failed to build Threshold with proper parameters")

        try:
            afwDetect.createThreshold(3.4, "foo bar")
        except:
            pass
        else:
            self.fail("Threhold parameters not properly validated")

        try:
            afwDetect.createThreshold(3.4, "variance")
        except:
            self.fail("Failed to build Threshold with proper parameters")

        try:
            afwDetect.createThreshold(3.4, "stdev")
        except:
            self.fail("Failed to build Threshold with proper parameters")

        try:
            afwDetect.createThreshold(3.4, "value")
        except:
            self.fail("Failed to build Threshold with proper parameters")

        try:
            afwDetect.createThreshold(3.4, "value", False)
        except:
            self.fail("Failed to build Threshold with VALUE, False parameters")

        try:
            afwDetect.createThreshold(0x4, "bitmask")
        except:
            self.fail("Failed to build Threshold with BITMASK parameters")

        try:
            afwDetect.createThreshold(5, "pixel_stdev")
        except:
            self.fail("Failed to build Threshold with PIXEL_STDEV parameters")

class FootprintTestCase(utilsTests.TestCase):
    """A test case for Footprint"""
    def setUp(self):
        self.foot = afwDetect.Footprint()

    def tearDown(self):
        del self.foot

    def testToString(self):
        y, x0, x1 = 10, 100, 101
        s = afwDetect.Span(y, x0, x1)
        self.assertEqual(s.toString(), toString(y, x0, x1))

    def testGC(self):
        """Check that Footprints are automatically garbage collected (when MemoryTestCase runs)"""

        f = afwDetect.Footprint()

    def testId(self):
        """Test uniqueness of IDs"""
        self.assertNotEqual(self.foot.getId(), afwDetect.Footprint().getId())

    def testIntersectMask(self):
        bbox = afwGeom.BoxI(afwGeom.PointI(0,0), afwGeom.ExtentI(10))
        fp = afwDetect.Footprint(bbox)
        maskBBox = afwGeom.BoxI(bbox)
        maskBBox.grow(-2)
        mask = afwImage.MaskU(maskBBox)
        innerBBox = afwGeom.BoxI(maskBBox)
        innerBBox.grow(-2)
        subMask = mask.Factory(mask, innerBBox)
        subMask.set(1)

        fp.intersectMask(mask)
        fpBBox = fp.getBBox()
        self.assertEqual(fpBBox.getMinX(), maskBBox.getMinX())
        self.assertEqual(fpBBox.getMinY(), maskBBox.getMinY())
        self.assertEqual(fpBBox.getMaxX(), maskBBox.getMaxX())
        self.assertEqual(fpBBox.getMaxY(), maskBBox.getMaxY())

        self.assertEqual(fp.getArea(), maskBBox.getArea() - innerBBox.getArea())

    def testTablePersistence(self):
        ellipse = afwGeomEllipses.Ellipse(afwGeomEllipses.Axes(8, 6, 0.25), afwGeom.Point2D(9,15))
        fp1 = afwDetect.Footprint(ellipse)
        fp1.addPeak(6, 7, 2)
        fp1.addPeak(8, 9, 3)
        with utilsTests.getTempFilePath(".fits") as tmpFile:
            fp1.writeFits(tmpFile)
            fp2 = afwDetect.Footprint.readFits(tmpFile)
            self.assertEqual(fp1.getArea(), fp2.getArea())
            self.assertEqual(list(fp1.getSpans()), list(fp2.getSpans()))
            # can't use Peak operator== for comparison because it compares IDs, not positions/values
            self.assertEqual(len(fp1.getPeaks()), len(fp2.getPeaks()))
            for peak1, peak2 in zip(fp1.getPeaks(), fp2.getPeaks()):
                self.assertEqual(peak1.getIx(), peak2.getIx())
                self.assertEqual(peak1.getIy(), peak2.getIy())
                self.assertEqual(peak1.getFx(), peak2.getFx())
                self.assertEqual(peak1.getFy(), peak2.getFy())
                self.assertEqual(peak1.getPeakValue(), peak2.getPeakValue())

    def testAddSpans(self):
        """Add spans to a Footprint"""
        for y, x0, x1 in [(10, 100, 105), (11, 99, 104)]:
            self.foot.addSpan(y, x0, x1)

        sp = self.foot.getSpans()

        self.assertEqual(sp[-1].toString(), toString(y, x0, x1))

    def testBbox(self):
        """Add Spans and check bounding box"""
        foot = afwDetect.Footprint()
        for y, x0, x1 in [(10, 100, 105),
                          (11, 99, 104)]:
            foot.addSpan(y, x0, x1)

        bbox = foot.getBBox()
        self.assertEqual(bbox.getWidth(), 7)
        self.assertEqual(bbox.getHeight(), 2)
        self.assertEqual(bbox.getMinX(), 99)
        self.assertEqual(bbox.getMinY(), 10)
        self.assertEqual(bbox.getMaxX(), 105)
        self.assertEqual(bbox.getMaxY(), 11)
        # clip with a bbox that doesn't overlap at all
        bbox2 = afwGeom.Box2I(afwGeom.Point2I(5, 90), afwGeom.Extent2I(1, 2))
        foot.clipTo(bbox2)
        self.assert_(foot.getBBox().isEmpty())
        self.assertEqual(foot.getArea(), 0)

    def testSpanShift(self):
        """Test our ability to shift spans"""
        span = afwDetect.Span(10, 100, 105)
        foot = afwDetect.Footprint()

        foot.addSpan(span, 1, 2)

        bbox = foot.getBBox()
        self.assertEqual(bbox.getWidth(), 6)
        self.assertEqual(bbox.getHeight(), 1)
        self.assertEqual(bbox.getMinX(), 101)
        self.assertEqual(bbox.getMinY(), 12)
        #
        # Shift that span using Span.shift
        #
        foot = afwDetect.Footprint()
        span.shift(-1, -2)
        foot.addSpan(span)

        bbox = foot.getBBox()
        self.assertEqual(bbox.getWidth(), 6)
        self.assertEqual(bbox.getHeight(), 1)
        self.assertEqual(bbox.getMinX(), 99)
        self.assertEqual(bbox.getMinY(), 8)

    def testFootprintFromBBox1(self):
        """Create a rectangular Footprint"""
        x0, y0, w, h = 9, 10, 7, 4
        foot = afwDetect.Footprint(afwGeom.Box2I(afwGeom.Point2I(x0, y0), afwGeom.Extent2I(w, h)))

        bbox = foot.getBBox()

        self.assertEqual(bbox.getWidth(), w)
        self.assertEqual(bbox.getHeight(), h)
        self.assertEqual(bbox.getMinX(), x0)
        self.assertEqual(bbox.getMinY(), y0)
        self.assertEqual(bbox.getMaxX(), x0 + w - 1)
        self.assertEqual(bbox.getMaxY(), y0 + h - 1)

        if False:
            idImage = afwImage.ImageU(w, h)
            idImage.set(0)
            foot.insertIntoImage(idImage, foot.getId(), bbox)
            ds9.mtv(idImage, frame=2)

    def testGetBBox(self):
        """Check that Footprint.getBBox() returns a copy"""
        x0, y0, w, h = 9, 10, 7, 4
        foot = afwDetect.Footprint(afwGeom.Box2I(afwGeom.Point2I(x0, y0), afwGeom.Extent2I(w, h)))
        bbox = foot.getBBox()

        dx, dy = 10, 20
        bbox.shift(afwGeom.Extent2I(dx, dy))

        self.assertEqual(bbox.getMinX(), x0 + dx)
        self.assertEqual(foot.getBBox().getMinX(), x0)

    def testFootprintFromCircle(self):
        """Create an elliptical Footprint"""
        ellipse = afwGeomEllipses.Ellipse(afwGeomEllipses.Axes(6, 6, 0), afwGeom.Point2D(9,15))
        foot = afwDetect.Footprint(ellipse, afwGeom.Box2I(afwGeom.Point2I(0, 0), afwGeom.Extent2I(20, 30)))

        idImage = afwImage.ImageU(afwGeom.Extent2I(foot.getRegion().getWidth(), foot.getRegion().getHeight()))
        idImage.set(0)

        foot.insertIntoImage(idImage, foot.getId())

        if False:
            ds9.mtv(idImage, frame=2)

    def testFootprintFromEllipse(self):
        """Create an elliptical Footprint"""
        cen = afwGeom.Point2D(23, 25)
        a, b, theta = 25, 15, 30
        ellipse = afwGeomEllipses.Ellipse(afwGeomEllipses.Axes(a, b, math.radians(theta)),  cen)
        foot = afwDetect.Footprint(ellipse, afwGeom.Box2I(afwGeom.Point2I(0, 0), afwGeom.Extent2I(50, 60)))

        idImage = afwImage.ImageU(afwGeom.Extent2I(foot.getRegion().getWidth(), foot.getRegion().getHeight()))
        idImage.set(0)

        foot.insertIntoImage(idImage, foot.getId())

        if display:
            ds9.mtv(idImage, frame=2)
            displayUtils.drawFootprint(foot, frame=2)
            shape = foot.getShape()
            shape.scale(2)              # <r^2> = 1/2 for a disk
            ds9.dot(shape, *cen, frame=2, ctype=ds9.RED)

            shape = foot.getShape()
            shape.scale(2)              # <r^2> = 1/2 for a disk
            ds9.dot(shape, *cen, frame=2, ctype=ds9.MAGENTA)

        axes = afwGeom.ellipses.Axes(foot.getShape())
        axes.scale(2)                   # <r^2> = 1/2 for a disk

        self.assertEqual(foot.getCentroid(), cen)
        self.assertTrue(abs(a - axes.getA()) < 0.15, "a: %g v. %g" % (a, axes.getA()))
        self.assertTrue(abs(b - axes.getB()) < 0.02, "b: %g v. %g" % (b, axes.getB()))
        self.assertTrue(abs(theta - math.degrees(axes.getTheta())) < 0.2,
                        "theta: %g v. %g" % (theta, math.degrees(axes.getTheta())))

    def testCopy(self):
        bbox = afwGeom.BoxI(afwGeom.PointI(0,2), afwGeom.PointI(5,6))

        fp = afwDetect.Footprint(bbox, bbox)

        #test copy construct
        fp2 = afwDetect.Footprint(fp)

        self.assertEqual(fp2.getBBox(), bbox)
        self.assertEqual(fp2.getRegion(), bbox)
        self.assertEqual(fp2.getArea(), bbox.getArea())
        self.assertEqual(fp2.isNormalized(), True)

        y = bbox.getMinY()
        for s in fp2.getSpans():
            self.assertEqual(s.getY(), y)
            self.assertEqual(s.getX0(), bbox.getMinX())
            self.assertEqual(s.getX1(), bbox.getMaxX())
            y+=1

        #test assignment
        fp3 = afwDetect.Footprint()
        fp3.assign(fp)
        self.assertEqual(fp3.getBBox(), bbox)
        self.assertEqual(fp3.getRegion(), bbox)
        self.assertEqual(fp3.getArea(), bbox.getArea())
        self.assertEqual(fp3.isNormalized(), True)

        y = bbox.getMinY()
        for s in fp3.getSpans():
            self.assertEqual(s.getY(), y)
            self.assertEqual(s.getX0(), bbox.getMinX())
            self.assertEqual(s.getX1(), bbox.getMaxX())
            y+=1

    def testShrink(self):
        width, height = 5, 10 # Size of footprint
        x0, y0 = 50, 50 # Position of footprint
        imwidth, imheight = 100, 100 # Size of image

        foot = afwDetect.Footprint(afwGeom.Box2I(afwGeom.Point2I(x0, y0), afwGeom.Extent2I(width, height)),
                                   afwGeom.Box2I(afwGeom.Point2I(0, 0), afwGeom.Extent2I(imwidth, imheight)))
        self.assertEqual(foot.getNpix(), width*height)

        # Add some peaks to the original footprint and check that those lying outside
        # the shrunken footprint are omitted from the returned shrunken footprint.
        foot.addPeak(50, 50, 1) # should be omitted in shrunken footprint
        foot.addPeak(52, 52, 2) # should be kept in shrunken footprint
        foot.addPeak(50, 59, 3) # should be omitted in shrunken footprint
        self.assertEqual(len(foot.getPeaks()), 3) # check that all three peaks were added

        # Shrinking by one pixel makes each dimension *two* pixels shorter.
        shrunk = afwDetect.shrinkFootprint(foot, 1, True)
        self.assertEqual(3*8, shrunk.getNpix())

        # Shrunken footprint should now only contain one peak at (52, 52)
        self.assertEqual(len(shrunk.getPeaks()), 1)
        peak = shrunk.getPeaks()[0]
        self.assertEqual((peak.getIx(), peak.getIy()), (52, 52))

        # Without shifting the centroid
        self.assertEqual(shrunk.getCentroid(), foot.getCentroid())

        # Get the same result from a Manhattan shrink
        shrunk = afwDetect.shrinkFootprint(foot, 1, False)
        self.assertEqual(3*8, shrunk.getNpix())
        self.assertEqual(shrunk.getCentroid(), foot.getCentroid())

        # Shrinking by a large amount leaves nothing.
        self.assertEqual(afwDetect.shrinkFootprint(foot, 100, True).getNpix(), 0)

    def testShrinkIsoVsManhattan(self):
        # Demonstrate that isotropic and Manhattan shrinks are different.
        radius = 8
        imwidth, imheight = 100, 100
        x0, y0 = imwidth//2, imheight//2
        nshrink = 4

        ellipse = afwGeomEllipses.Ellipse(afwGeomEllipses.Axes(1.5*radius, 2*radius, 0),
                                          afwGeom.Point2D(x0,y0))
        foot = afwDetect.Footprint(ellipse, afwGeom.Box2I(afwGeom.Point2I(0, 0),
                                   afwGeom.Extent2I(imwidth, imheight)))
        self.assertNotEqual(afwDetect.shrinkFootprint(foot, nshrink, False),
                            afwDetect.shrinkFootprint(foot, nshrink, True))

    def _fig8Test(self, x1, y1, x2, y2):
        # Construct a "figure of 8" consisting of two circles touching at the
        # centre of an image, then demonstrate that it shrinks correctly.
        # (Helper method for tests below.)
        radius = 3
        imwidth, imheight = 100, 100
        nshrink = 1

        # These are the correct values for footprint sizes given the paramters
        # above.
        circle_npix = 29
        initial_npix = circle_npix * 2 - 1 # touch at one pixel
        shrunk_npix = 26

        box = afwGeom.Box2I(afwGeom.Point2I(0, 0), afwGeom.Extent2I(imwidth, imheight))

        e1 = afwGeomEllipses.Ellipse(afwGeomEllipses.Axes(radius, radius, 0),
                                          afwGeom.Point2D(x1, y1))
        f1 = afwDetect.Footprint(e1,box)
        self.assertEqual(f1.getNpix(), circle_npix)

        e2 = afwGeomEllipses.Ellipse(afwGeomEllipses.Axes(radius, radius, 0),
                                          afwGeom.Point2D(x2, y2))
        f2 = afwDetect.Footprint(e2,box)
        self.assertEqual(f2.getNpix(), circle_npix)

        initial = afwDetect.mergeFootprints(f1, f2)
        initial.setRegion(f2.getRegion()) # merge does not propagate the region
        self.assertEqual(initial_npix, initial.getNpix())

        shrunk = afwDetect.shrinkFootprint(initial, nshrink, True)
        self.assertEqual(shrunk_npix, shrunk.getNpix())

        if display:
            idImage = afwImage.ImageU(imwidth, imheight)
            for i, foot in enumerate([initial, shrunk]):
                print foot.getNpix()
                foot.insertIntoImage(idImage, i+1);
            ds9.mtv(idImage)

    def testShrinkEightVertical(self):
        # Test a "vertical" figure of 8.
        radius = 3
        imwidth, imheight = 100, 100
        self._fig8Test(imwidth//2, imheight//2-radius, imwidth//2, imheight//2+radius)

    def testShrinkEightHorizontal(self):
        # Test a "horizontal" figure of 8.
        radius = 3
        imwidth, imheight = 100, 100
        self._fig8Test(imwidth//2-radius, imheight//2, imwidth//2+radius, imheight//2)

    def testGrow(self):
        """Test growing a footprint"""
        x0, y0 = 20, 20
        width, height = 20, 30
        foot1 = afwDetect.Footprint(afwGeom.Box2I(afwGeom.Point2I(x0, y0), afwGeom.Extent2I(width, height)),
                                    afwGeom.Box2I(afwGeom.Point2I(0, 0), afwGeom.Extent2I(100, 100)))

        # Add some peaks and check that they get copied into the new grown footprint
        foot1.addPeak(20, 20, 1)
        foot1.addPeak(30, 35, 2)
        foot1.addPeak(25, 45, 3)
        self.assertEqual(len(foot1.getPeaks()), 3)

        bbox1 = foot1.getBBox()

        self.assertEqual(bbox1.getMinX(), x0)
        self.assertEqual(bbox1.getMaxX(), x0 + width - 1)
        self.assertEqual(bbox1.getWidth(), width)

        self.assertEqual(bbox1.getMinY(), y0)
        self.assertEqual(bbox1.getMaxY(), y0 + height - 1)
        self.assertEqual(bbox1.getHeight(), height)

        ngrow = 5
        for isotropic in (True, False):
            foot2 = afwDetect.growFootprint(foot1, ngrow, isotropic)

            # Check that peaks got copied into grown footprint
            self.assertEqual(len(foot2.getPeaks()), 3)
            for peak in foot2.getPeaks():
                self.assertTrue((peak.getIx(), peak.getIy()) in [(20, 20), (30, 35), (25, 45)])

            bbox2 = foot2.getBBox()

            if False and display:
                idImage = afwImage.ImageU(width, height)
                idImage.set(0)

                i = 1
                for foot in [foot1, foot2]:
                    foot.insertIntoImage(idImage, i)
                    i += 1

                metricImage = afwImage.ImageF("foo.fits")
                ds9.mtv(metricImage, frame=1)
                ds9.mtv(idImage)

            # check bbox2
            self.assertEqual(bbox2.getMinX(), x0 - ngrow)
            self.assertEqual(bbox2.getWidth(), width + 2*ngrow)

            self.assertEqual(bbox2.getMinY(), y0 - ngrow)
            self.assertEqual(bbox2.getHeight(), height + 2*ngrow)
            # Check that region was preserved
            self.assertEqual(foot1.getRegion(), foot2.getRegion())

    def testFootprintToBBoxList(self):
        """Test footprintToBBoxList"""
        region = afwGeom.Box2I(afwGeom.Point2I(0,0), afwGeom.Extent2I(12,10))
        foot = afwDetect.Footprint(0, region)
        for y, x0, x1 in [(3, 3, 5), (3, 7, 7),
                          (4, 2, 3), (4, 5, 7),
                          (5, 2, 3), (5, 5, 8),
                          (6, 3, 5),
                          ]:
            foot.addSpan(y, x0, x1)

        idImage = afwImage.ImageU(region.getDimensions())
        idImage.set(0)

        foot.insertIntoImage(idImage, 1)
        if display:
            ds9.mtv(idImage)

        idImageFromBBox = idImage.Factory(idImage, True)
        idImageFromBBox.set(0)
        bboxes = afwDetect.footprintToBBoxList(foot)
        for bbox in bboxes:
            x0, y0, x1, y1 = bbox.getMinX(), bbox.getMinY(), bbox.getMaxX(), bbox.getMaxY()

            for y in range(y0, y1 + 1):
                for x in range(x0, x1 + 1):
                    idImageFromBBox.set(x, y, 1)

            if display:
                x0 -= 0.5
                y0 -= 0.5
                x1 += 0.5
                y1 += 0.5

                ds9.line([(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)], ctype=ds9.RED)

        idImageFromBBox -= idImage      # should be blank
        stats = afwMath.makeStatistics(idImageFromBBox, afwMath.MAX)

        self.assertEqual(stats.getValue(), 0)

    def testWriteDefect(self):
        """Write a Footprint as a set of Defects"""
        region = afwGeom.Box2I(afwGeom.Point2I(0,0), afwGeom.Extent2I(12,10))
        foot = afwDetect.Footprint(0, region)
        for y, x0, x1 in [(3, 3, 5), (3, 7, 7),
                          (4, 2, 3), (4, 5, 7),
                          (5, 2, 3), (5, 5, 8),
                          (6, 3, 5),
                          ]:
            foot.addSpan(y, x0, x1)

        if True:
            fd = open("/dev/null", "w")
        else:
            fd = sys.stdout

        afwDetectUtils.writeFootprintAsDefects(fd, foot)


    def testNormalize(self):
        """Test Footprint.normalize"""
        w, h = 12, 10
        region = afwGeom.Box2I(afwGeom.Point2I(0,0), afwGeom.Extent2I(w,h))
        im = afwImage.ImageU(afwGeom.Extent2I(w, h))
        im.set(0)
        #
        # Create a footprint;  note that these Spans overlap
        #
        for spans, box in (([(3, 5, 6),
                             (4, 7, 7), ], afwGeom.Box2I(afwGeom.Point2I(5,3), afwGeom.Point2I(7,4))),
                           ([(3, 3, 5), (3, 6, 9),
                             (4, 2, 3), (4, 5, 7), (4, 8, 8),
                             (5, 2, 3), (5, 5, 8), (5, 6, 7),
                             (6, 3, 5),
                             ], afwGeom.Box2I(afwGeom.Point2I(2,3), afwGeom.Point2I(9,6)))
                      ):

            foot = afwDetect.Footprint(0, region)
            for y, x0, x1 in spans:
                foot.addSpan(y, x0, x1)

                for x in range(x0, x1 + 1): # also insert into im
                    im.set(x, y, 1)

            idImage = afwImage.ImageU(afwGeom.Extent2I(w, h))
            idImage.set(0)

            foot.insertIntoImage(idImage, 1)
            if display:             # overlaping pixels will be > 1
                ds9.mtv(idImage)
            #
            # Normalise the Footprint, removing overlapping spans
            #
            foot.normalize()

            idImage.set(0)
            foot.insertIntoImage(idImage, 1)
            if display:
                ds9.mtv(idImage, frame=1)

            idImage -= im

            self.assertTrue(box == foot.getBBox())
            self.assertEqual(afwMath.makeStatistics(idImage, afwMath.MAX).getValue(), 0)

    def testSetFromFootprint(self):
        """Test setting mask/image pixels from a Footprint list"""
        mi = afwImage.MaskedImageF(afwGeom.Extent2I(12, 8))
        im = mi.getImage()
        #
        # Objects that we should detect
        #
        self.objects = []
        self.objects += [Object(10, [(1, 4, 4), (2, 3, 5), (3, 4, 4)])]
        self.objects += [Object(20, [(5, 7, 8), (5, 10, 10), (6, 8, 9)])]
        self.objects += [Object(20, [(6, 3, 3)])]

        im.set(0)                       # clear image
        for obj in self.objects:
            obj.insert(im)

        if False and display:
            ds9.mtv(mi, frame=0)

        ds = afwDetect.FootprintSet(mi, afwDetect.Threshold(15))

        objects = ds.getFootprints()
        afwDetect.setMaskFromFootprintList(mi.getMask(), objects, 0x1)

        self.assertEqual(mi.getMask().get(4, 2), 0x0)
        self.assertEqual(mi.getMask().get(3, 6), 0x1)

        self.assertEqual(mi.getImage().get(3, 6), 20)
        afwDetect.setImageFromFootprintList(mi.getImage(), objects, 5.0)
        self.assertEqual(mi.getImage().get(4, 2), 10)
        self.assertEqual(mi.getImage().get(3, 6), 5)

        if display:
            ds9.mtv(mi, frame=1)
        #
        # Check Footprint.contains() while we are about it
        #
        self.assertTrue(objects[0].contains(afwGeom.Point2I(7, 5)))
        self.assertFalse(objects[0].contains(afwGeom.Point2I(10, 6)))
        self.assertFalse(objects[0].contains(afwGeom.Point2I(7, 6)))
        self.assertFalse(objects[0].contains(afwGeom.Point2I(4, 2)))

        self.assertTrue(objects[1].contains(afwGeom.Point2I(3, 6)))

    def testMakeFootprintSetXY0(self):
        """Test setting mask/image pixels from a Footprint list"""
        mi = afwImage.MaskedImageF(afwGeom.Extent2I(12, 8))
        im = mi.getImage()
        im.set(100)

        mi.setXY0(afwGeom.PointI(2, 2))
        ds = afwDetect.FootprintSet(mi, afwDetect.Threshold(1), "DETECTED")

        bitmask = mi.getMask().getPlaneBitMask("DETECTED")
        for y in range(im.getHeight()):
            for x in range(im.getWidth()):
                self.assertEqual(mi.getMask().get(x, y), bitmask)

    def testTransform(self):
        dims = afwGeom.Extent2I(512, 512)
        bbox = afwGeom.Box2I(afwGeom.Point2I(0,0), dims)
        radius = 5
        offset = afwGeom.Extent2D(123, 456)
        crval = afwCoord.Coord(0*afwGeom.degrees, 0*afwGeom.degrees)
        crpix = afwGeom.Point2D(0, 0)
        cdMatrix = [1.0e-5, 0.0, 0.0, 1.0e-5]
        source = afwImage.makeWcs(crval, crpix, *cdMatrix)
        target = afwImage.makeWcs(crval, crpix + offset, *cdMatrix)
        fpSource = afwDetect.Footprint(afwGeom.Point2I(12, 34), radius, bbox)

        fpTarget = fpSource.transform(source, target, bbox)

        self.assertEqual(len(fpSource.getSpans()), len(fpTarget.getSpans()))
        self.assertEqual(fpSource.getNpix(), fpTarget.getNpix())
        self.assertEqual(fpSource.getArea(), fpTarget.getArea())

        imSource = afwImage.ImageU(dims)
        fpSource.insertIntoImage(imSource, 1)

        imTarget = afwImage.ImageU(dims)
        fpTarget.insertIntoImage(imTarget, 1)

        subSource = imSource.Factory(imSource, fpSource.getBBox())
        subTarget = imTarget.Factory(imTarget, fpTarget.getBBox())
        self.assertTrue(numpy.all(subSource.getArray() == subTarget.getArray()))

        # make a bbox smaller than the target footprint
        bbox2 = afwGeom.Box2I(fpTarget.getBBox())
        bbox2.grow(-1)
        fpTarget2 = fpSource.transform(source, target, bbox2)  # this one clips
        fpTarget3 = fpSource.transform(source, target, bbox2, False)  # this one doesn't
        self.assertTrue(bbox2.contains(fpTarget2.getBBox()))
        self.assertFalse(bbox2.contains(fpTarget3.getBBox()))
        self.assertNotEqual(fpTarget.getArea(), fpTarget2.getArea())
        self.assertEqual(fpTarget.getArea(), fpTarget3.getArea())

    def testCopyWithinFootprintImage(self):
        W,H = 10,10
        dims = afwGeom.Extent2I(W,H)
        source = afwImage.ImageF(dims)
        dest = afwImage.ImageF(dims)
        sa = source.getArray()
        for i in range(H):
            for j in range(W):
                sa[i,j] = 100 * i + j

        self.foot.addSpan(4, 3, 6)
        self.foot.addSpan(5, 2, 4)

        afwDetect.copyWithinFootprintImage(self.foot, source, dest)

        da = dest.getArray()
        self.assertEqual(da[4,2], 0)
        self.assertEqual(da[4,3], 403)
        self.assertEqual(da[4,4], 404)
        self.assertEqual(da[4,5], 405)
        self.assertEqual(da[4,6], 406)
        self.assertEqual(da[4,7], 0)
        self.assertEqual(da[5,1], 0)
        self.assertEqual(da[5,2], 502)
        self.assertEqual(da[5,3], 503)
        self.assertEqual(da[5,4], 504)
        self.assertEqual(da[5,5], 0)
        self.assertTrue(numpy.all(da[:4,:] == 0))
        self.assertTrue(numpy.all(da[6:,:] == 0))

    def testCopyWithinFootprintMaskedImage(self):
        W,H = 10,10
        dims = afwGeom.Extent2I(W,H)
        source = afwImage.MaskedImageF(dims)
        dest = afwImage.MaskedImageF(dims)
        sa = source.getImage().getArray()
        sv = source.getVariance().getArray()
        sm = source.getMask().getArray()
        for i in range(H):
            for j in range(W):
                sa[i,j] = 100 * i + j
                sv[i,j] = 100 * j + i
                sm[i,j] = 1

        self.foot.addSpan(4, 3, 6)
        self.foot.addSpan(5, 2, 4)

        afwDetect.copyWithinFootprintMaskedImage(self.foot, source, dest)

        da = dest.getImage().getArray()
        dv = dest.getVariance().getArray()
        dm = dest.getMask().getArray()

        self.assertEqual(da[4,2], 0)
        self.assertEqual(da[4,3], 403)
        self.assertEqual(da[4,4], 404)
        self.assertEqual(da[4,5], 405)
        self.assertEqual(da[4,6], 406)
        self.assertEqual(da[4,7], 0)
        self.assertEqual(da[5,1], 0)
        self.assertEqual(da[5,2], 502)
        self.assertEqual(da[5,3], 503)
        self.assertEqual(da[5,4], 504)
        self.assertEqual(da[5,5], 0)
        self.assertTrue(numpy.all(da[:4,:] == 0))
        self.assertTrue(numpy.all(da[6:,:] == 0))

        self.assertEqual(dv[4,2], 0)
        self.assertEqual(dv[4,3], 304)
        self.assertEqual(dv[4,4], 404)
        self.assertEqual(dv[4,5], 504)
        self.assertEqual(dv[4,6], 604)
        self.assertEqual(dv[4,7], 0)
        self.assertEqual(dv[5,1], 0)
        self.assertEqual(dv[5,2], 205)
        self.assertEqual(dv[5,3], 305)
        self.assertEqual(dv[5,4], 405)
        self.assertEqual(dv[5,5], 0)
        self.assertTrue(numpy.all(dv[:4,:] == 0))
        self.assertTrue(numpy.all(dv[6:,:] == 0))

        self.assertTrue(numpy.all(dm[4, 3:7] == 1))
        self.assertTrue(numpy.all(dm[5, 2:5] == 1))
        self.assertTrue(numpy.all(dm[:4,:] == 0))
        self.assertTrue(numpy.all(dm[6:,:] == 0))
        self.assertTrue(numpy.all(dm[4, :3] == 0))
        self.assertTrue(numpy.all(dm[4, 7:] == 0))

    def testMergeFootprints(self):
        f1 = self.foot
        f2 = afwDetect.Footprint()

        f1.addSpan(10, 10, 20)
        f1.addSpan(10, 30, 40)
        f1.addSpan(10, 50, 60)

        f1.addSpan(11, 30, 50)
        f1.addSpan(12, 30, 50)

        f1.addSpan(13, 10, 20)
        f1.addSpan(13, 30, 40)
        f1.addSpan(13, 50, 60)

        f1.addSpan(15, 10,20)
        f1.addSpan(15, 31,40)
        f1.addSpan(15, 51,60)

        f2.addSpan(8,  10, 20)
        f2.addSpan(9,  20, 30)
        f2.addSpan(10,  0,  9)
        f2.addSpan(10, 35, 65)
        f2.addSpan(10, 70, 80)

        f2.addSpan(13, 49, 54)
        f2.addSpan(14, 10, 30)

        f2.addSpan(15, 21,30)
        f2.addSpan(15, 41,50)
        f2.addSpan(15, 61,70)

        f1.normalize()
        f2.normalize()

        fA = afwDetect.mergeFootprints(f1, f2)
        fB = afwDetect.mergeFootprints(f2, f1)

        ims = []
        for i,f in enumerate([f1,f2,fA,fB]):
            im1 = afwImage.ImageU(100, 100)
            im1.set(0)
            imbb = im1.getBBox()
            f.setRegion(imbb)
            f.insertIntoImage(im1, 1)
            ims.append(im1)

        for i,merged in enumerate([ims[2],ims[3]]):
            m = merged.getArray()
            a1 = ims[0].getArray()
            a2 = ims[1].getArray()
            # Slightly looser tests to start...
            # Every pixel in f1 is in f[AB]
            self.assertTrue(numpy.all(m.flat[numpy.flatnonzero(a1)] == 1))
            # Every pixel in f2 is in f[AB]
            self.assertTrue(numpy.all(m.flat[numpy.flatnonzero(a2)] == 1))
            # merged == a1 | a2.
            self.assertTrue(numpy.all(m == numpy.maximum(a1, a2)))

        if False:
            import matplotlib
            matplotlib.use('Agg')
            import pylab as plt
            plt.clf()
            for i,im1 in enumerate(ims):
                plt.subplot(4,1, i+1)
                plt.imshow(im1.getArray(), interpolation='nearest', origin='lower')
                plt.axis([0, 100, 0, 20])
            plt.savefig('merge2.png')


    def testClipToNonzero(self):
        # create a circular footprint
        ellipse = afwGeomEllipses.Ellipse(afwGeomEllipses.Axes(6, 6, 0), afwGeom.Point2D(9,15))
        bb = afwGeom.Box2I(afwGeom.Point2I(0, 0), afwGeom.Extent2I(20, 30))
        foot = afwDetect.Footprint(ellipse, bb)

        a0 = foot.getArea()

        plots = False
        if plots:
            import matplotlib
            matplotlib.use('Agg')
            import pylab as plt

            plt.clf()
            img = afwImage.ImageU(bb)
            foot.insertIntoImage(img, 1)
            ima = dict(interpolation='nearest', origin='lower', cmap='gray')
            plt.imshow(img.getArray(), **ima)
            plt.savefig('clipnz1.png')

        source = afwImage.ImageF(bb)
        source.getArray()[:,:] = 1.
        source.getArray()[:,0:10] = 0.

        foot.clipToNonzero(source)
        foot.normalize()
        a1 = foot.getArea()
        self.assertLess(a1, a0)

        img = afwImage.ImageU(bb)
        foot.insertIntoImage(img, 1)
        self.assertTrue(numpy.all(img.getArray()[source.getArray() == 0] == 0))

        if plots:
            plt.clf()
            plt.subplot(1,2,1)
            plt.imshow(source.getArray(), **ima)
            plt.subplot(1,2,2)
            plt.imshow(img.getArray(), **ima)
            plt.savefig('clipnz2.png')

        source.getArray()[:12,:] = 0.
        foot.clipToNonzero(source)
        foot.normalize()

        a2 = foot.getArea()
        self.assertLess(a2, a1)

        img = afwImage.ImageU(bb)
        foot.insertIntoImage(img, 1)
        self.assertTrue(numpy.all(img.getArray()[source.getArray() == 0] == 0))

        if plots:
            plt.clf()
            plt.subplot(1,2,1)
            plt.imshow(source.getArray(), **ima)
            plt.subplot(1,2,2)
            img = afwImage.ImageU(bb)
            foot.insertIntoImage(img, 1)
            plt.imshow(img.getArray(), **ima)
            plt.savefig('clipnz3.png')

    def testInclude(self):
        """Test that we can expand a Footprint to include the union of itself and all others
        provided (must be non-disjoint).
        """
        region = afwGeom.Box2I(afwGeom.Point2I(-6, -6), afwGeom.Point2I(6, 6))
        parent = afwDetect.Footprint(afwGeom.Box2I(afwGeom.Point2I(-2, -2), afwGeom.Point2I(2, 2)), region)
        parent.addPeak(0, 0, float("NaN"))
        child1 = afwDetect.Footprint(afwGeom.Box2I(afwGeom.Point2I(-3, 0), afwGeom.Point2I(0, 3)), region)
        child1.addPeak(-1, 1, float("NaN"))
        child2 = afwDetect.Footprint(afwGeom.Box2I(afwGeom.Point2I(-4, -3), afwGeom.Point2I(-1, 0)), region)
        child3 = afwDetect.Footprint(afwGeom.Box2I(afwGeom.Point2I(4, -1), afwGeom.Point2I(6, 1)))
        merge12 = afwDetect.Footprint(parent)
        merge12.include([child1, child2])
        self.assertTrue(merge12.getBBox().contains(parent.getBBox()))
        self.assertTrue(merge12.getBBox().contains(child1.getBBox()))
        self.assertTrue(merge12.getBBox().contains(child2.getBBox()))
        mask12a = afwImage.MaskU(region)
        mask12b = afwImage.MaskU(region)
        afwDetect.setMaskFromFootprint(mask12a, parent, 1)
        afwDetect.setMaskFromFootprint(mask12a, child1, 1)
        afwDetect.setMaskFromFootprint(mask12a, child2, 1)
        afwDetect.setMaskFromFootprint(mask12b, merge12, 1)
        self.assertEqual(mask12a.getArray().sum(), merge12.getArea())
        self.assertClose(mask12a.getArray(), mask12b.getArray(), rtol=0, atol=0)
        self.assertRaisesLsstCpp(pexExcept.RuntimeError, parent.include, [child1, child2, child3])

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class FootprintSetTestCase(unittest.TestCase):
    """A test case for FootprintSet"""

    def setUp(self):
        self.ms = afwImage.MaskedImageF(afwGeom.Extent2I(12, 8))
        im = self.ms.getImage()
        #
        # Objects that we should detect
        #
        self.objects = []
        self.objects += [Object(10, [(1, 4, 4), (2, 3, 5), (3, 4, 4)])]
        self.objects += [Object(20, [(5, 7, 8), (5, 10, 10), (6, 8, 9)])]
        self.objects += [Object(20, [(6, 3, 3)])]

        self.ms.set((0, 0x0, 4.0))      # clear image; set variance
        for obj in self.objects:
            obj.insert(im)

        if False and display:
            ds9.mtv(im, frame=0)

    def tearDown(self):
        del self.ms

    def testGC(self):
        """Check that FootprintSets are automatically garbage collected (when MemoryTestCase runs)"""
        ds = afwDetect.FootprintSet(afwImage.MaskedImageF(afwGeom.Extent2I(10, 20)), afwDetect.Threshold(10))

    def testFootprints(self):
        """Check that we found the correct number of objects and that they are correct"""
        ds = afwDetect.FootprintSet(self.ms, afwDetect.Threshold(10))

        objects = ds.getFootprints()

        self.assertEqual(len(objects), len(self.objects))
        for i in range(len(objects)):
            self.assertEqual(objects[i], self.objects[i])

    def testFootprints2(self):
        """Check that we found the correct number of objects using FootprintSet"""
        ds = afwDetect.FootprintSet(self.ms, afwDetect.Threshold(10))

        objects = ds.getFootprints()

        self.assertEqual(len(objects), len(self.objects))
        for i in range(len(objects)):
            self.assertEqual(objects[i], self.objects[i])

    def testFootprints3(self):
        """Check that we found the correct number of objects using FootprintSet and PIXEL_STDEV"""
        threshold = 4.5                 # in units of sigma

        self.ms.set(2, 4, (10, 0x0, 36)) # not detected (high variance)

        y, x = self.objects[2].spans[0][0:2]
        self.ms.set(x, y, (threshold, 0x0, 1.0))

        ds = afwDetect.FootprintSet(self.ms,
                                        afwDetect.createThreshold(threshold, "pixel_stdev"), "OBJECT")

        objects = ds.getFootprints()

        self.assertEqual(len(objects), len(self.objects))
        for i in range(len(objects)):
            self.assertEqual(objects[i], self.objects[i])

    def testFootprintsMasks(self):
        """Check that detectionSets have the proper mask bits set"""
        ds = afwDetect.FootprintSet(self.ms, afwDetect.Threshold(10), "OBJECT")
        objects = ds.getFootprints()

        if display:
            ds9.mtv(self.ms, frame=1)

        mask = self.ms.getMask()
        for i in range(len(objects)):
            for sp in objects[i].getSpans():
                for x in range(sp.getX0(), sp.getX1() + 1):
                    self.assertEqual(mask.get(x, sp.getY()), mask.getPlaneBitMask("OBJECT"))

    def testFootprintsImageId(self):
        """Check that we can insert footprints into an Image"""
        ds = afwDetect.FootprintSet(self.ms, afwDetect.Threshold(10))
        objects = ds.getFootprints()

        idImage = afwImage.ImageU(self.ms.getDimensions())
        idImage.set(0)

        for foot in objects:
            foot.insertIntoImage(idImage, foot.getId())

        if False:
            ds9.mtv(idImage, frame=2)

        for i in range(len(objects)):
            for sp in objects[i].getSpans():
                for x in range(sp.getX0(), sp.getX1() + 1):
                    self.assertEqual(idImage.get(x, sp.getY()), objects[i].getId())


    def testFootprintSetImageId(self):
        """Check that we can insert a FootprintSet into an Image, setting relative IDs"""
        ds = afwDetect.FootprintSet(self.ms, afwDetect.Threshold(10))
        objects = ds.getFootprints()

        idImage = ds.insertIntoImage(True)
        if display:
            ds9.mtv(idImage, frame=2)

        for i in range(len(objects)):
            for sp in objects[i].getSpans():
                for x in range(sp.getX0(), sp.getX1() + 1):
                    self.assertEqual(idImage.get(x, sp.getY()), i + 1)

    def testFootprintsImage(self):
        """Check that we can search Images as well as MaskedImages"""
        ds = afwDetect.FootprintSet(self.ms.getImage(), afwDetect.Threshold(10))

        objects = ds.getFootprints()

        self.assertEqual(len(objects), len(self.objects))
        for i in range(len(objects)):
            self.assertEqual(objects[i], self.objects[i])

    def testGrow2(self):
        """Grow some more interesting shaped Footprints.  Informative with display, but no numerical tests"""
        ds = afwDetect.FootprintSet(self.ms, afwDetect.Threshold(10), "OBJECT")

        idImage = afwImage.ImageU(self.ms.getDimensions())
        idImage.set(0)

        i = 1
        for foot in ds.getFootprints()[0:1]:
            gfoot = afwDetect.growFootprint(foot, 3, False)
            gfoot.insertIntoImage(idImage, i)
            i += 1

        if display:
            ds9.mtv(self.ms, frame=0)
            ds9.mtv(idImage, frame=1)

    def testFootprintPeaks(self):
        """Test that we can extract the peaks from a Footprint"""
        fs = afwDetect.FootprintSet(self.ms, afwDetect.Threshold(10), "OBJECT")

        foot = fs.getFootprints()[0]

        self.assertEqual(len(foot.getPeaks()), 5)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class MaskFootprintSetTestCase(unittest.TestCase):
    """A test case for generating FootprintSet from Masks"""

    def setUp(self):
        self.mim = afwImage.MaskedImageF(afwGeom.ExtentI(12, 8))
        #
        # Objects that we should detect
        #
        self.objects = []
        self.objects += [Object(0x2, [(1, 4, 4), (2, 3, 5), (3, 4, 4)])]
        self.objects += [Object(0x41, [(5, 7, 8), (6, 8, 8)])]
        self.objects += [Object(0x42, [(5, 10, 10)])]
        self.objects += [Object(0x82, [(6, 3, 3)])]

        self.mim.set((0, 0, 0))                 # clear image
        for obj in self.objects:
            obj.insert(self.mim.getImage())
            obj.insert(self.mim.getMask())

        if display:
            ds9.mtv(self.mim, frame=0)

    def tearDown(self):
        del self.mim

    def testFootprints(self):
        """Check that we found the correct number of objects using FootprintSet"""
        level = 0x2
        ds = afwDetect.FootprintSet(self.mim.getMask(), afwDetect.createThreshold(level, "bitmask"))

        objects = ds.getFootprints()

        if 0 and display:
            ds9.mtv(self.mim, frame=0)

        self.assertEqual(len(objects), len([o for o in self.objects if (o.val & level)]))

        i = 0
        for o in self.objects:
            if o.val & level:
                self.assertEqual(o, objects[i])
                i += 1

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class NaNFootprintSetTestCase(unittest.TestCase):
    """A test case for FootprintSet when the image contains NaNs"""

    def setUp(self):
        self.ms = afwImage.MaskedImageF(afwGeom.Extent2I(12, 8))
        im = self.ms.getImage()
        #
        # Objects that we should detect
        #
        self.objects = []
        self.objects += [Object(10, [(1, 4, 4), (2, 3, 5), (3, 4, 4)])]
        self.objects += [Object(20, [(5, 7, 8), (6, 8, 8)])]
        self.objects += [Object(20, [(5, 10, 10)])]
        self.objects += [Object(30, [(6, 3, 3)])]

        im.set(0)                       # clear image
        for obj in self.objects:
            obj.insert(im)

        self.NaN = float("NaN")
        im.set(3, 7, self.NaN)
        im.set(0, 0, self.NaN)
        im.set(8, 2, self.NaN)

        im.set(9, 6, self.NaN)          # connects the two objects with value==20 together if NaN is detected

        if False and display:
            ds9.mtv(im, frame=0)

    def tearDown(self):
        del self.ms

    def testFootprints(self):
        """Check that we found the correct number of objects using FootprintSet"""
        ds = afwDetect.FootprintSet(self.ms, afwDetect.Threshold(10), "DETECTED")

        objects = ds.getFootprints()

        if display:
            ds9.mtv(self.ms, frame=0)

        self.assertEqual(len(objects), len(self.objects))
        for i in range(len(objects)):
            self.assertEqual(objects[i], self.objects[i])

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(ThresholdTestCase)
    suites += unittest.makeSuite(SpanTestCase)
    suites += unittest.makeSuite(FootprintTestCase)
    suites += unittest.makeSuite(FootprintSetTestCase)
    suites += unittest.makeSuite(NaNFootprintSetTestCase)
    suites += unittest.makeSuite(MaskFootprintSetTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)


def run(shouldExit=False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)
