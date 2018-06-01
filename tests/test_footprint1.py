#
# LSST Data Management System
# Copyright 2008-2017 LSST Corporation.
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

import math
import sys
import unittest
import os

import numpy as np

import lsst.utils.tests
import lsst.geom
import lsst.afw.geom as afwGeom
import lsst.afw.geom.ellipses as afwGeomEllipses
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.detection as afwDetect
import lsst.afw.detection.utils as afwDetectUtils
import lsst.afw.display.ds9 as ds9
import lsst.afw.display.utils as displayUtils

try:
    type(display)
except NameError:
    display = False

testPath = os.path.abspath(os.path.dirname(__file__))


def toString(*args):
    """toString written in python"""
    if len(args) == 1:
        args = args[0]

    y, x0, x1 = args
    return "%d: %d..%d" % (y, x0, x1)


class Object:

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


class SpanTestCase(unittest.TestCase):

    def testLessThan(self):
        span1 = afwDetect.Span(42, 0, 100)
        span2 = afwDetect.Span(41, 0, 100)
        span3 = afwDetect.Span(43, 0, 100)
        span4 = afwDetect.Span(42, -100, 100)
        span5 = afwDetect.Span(42, 100, 200)
        span6 = afwDetect.Span(42, 0, 10)
        span7 = afwDetect.Span(42, 0, 200)
        span8 = afwDetect.Span(42, 0, 100)

        # Cannot use assertLess and friends here
        # because Span only has operator <
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
        except Exception:
            self.fail("Failed to build Threshold with proper parameters")

        try:
            afwDetect.createThreshold(3.4, "foo bar")
        except Exception:
            pass
        else:
            self.fail("Threhold parameters not properly validated")

        try:
            afwDetect.createThreshold(3.4, "variance")
        except Exception:
            self.fail("Failed to build Threshold with proper parameters")

        try:
            afwDetect.createThreshold(3.4, "stdev")
        except Exception:
            self.fail("Failed to build Threshold with proper parameters")

        try:
            afwDetect.createThreshold(3.4, "value")
        except Exception:
            self.fail("Failed to build Threshold with proper parameters")

        try:
            afwDetect.createThreshold(3.4, "value", False)
        except Exception:
            self.fail("Failed to build Threshold with VALUE, False parameters")

        try:
            afwDetect.createThreshold(0x4, "bitmask")
        except Exception:
            self.fail("Failed to build Threshold with BITMASK parameters")

        try:
            afwDetect.createThreshold(5, "pixel_stdev")
        except Exception:
            self.fail("Failed to build Threshold with PIXEL_STDEV parameters")


class FootprintTestCase(lsst.utils.tests.TestCase):
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

        afwDetect.Footprint()

    def testId(self):
        """Test uniqueness of IDs"""
        self.assertNotEqual(self.foot.getId(), afwDetect.Footprint().getId())

    def testIntersectMask(self):
        bbox = lsst.geom.BoxI(lsst.geom.PointI(0, 0), lsst.geom.ExtentI(10))
        fp = afwDetect.Footprint(afwGeom.SpanSet(bbox))
        maskBBox = lsst.geom.BoxI(bbox)
        maskBBox.grow(-2)
        mask = afwImage.Mask(maskBBox)
        innerBBox = lsst.geom.BoxI(maskBBox)
        innerBBox.grow(-2)
        subMask = mask.Factory(mask, innerBBox)
        subMask.set(1)

        # We only want the pixels that are unmasked, and lie in the bounding box
        # of the mask, so not the mask (selecting only zero values) and clipped
        fp.spans = fp.spans.intersectNot(mask).clippedTo(mask.getBBox())
        fp.removeOrphanPeaks()
        fpBBox = fp.getBBox()
        self.assertEqual(fpBBox.getMinX(), maskBBox.getMinX())
        self.assertEqual(fpBBox.getMinY(), maskBBox.getMinY())
        self.assertEqual(fpBBox.getMaxX(), maskBBox.getMaxX())
        self.assertEqual(fpBBox.getMaxY(), maskBBox.getMaxY())

        self.assertEqual(fp.getArea(), maskBBox.getArea() - innerBBox.getArea())

    def testTablePersistence(self):
        ellipse = afwGeom.Ellipse(afwGeomEllipses.Axes(8, 6, 0.25),
                                  lsst.geom.Point2D(9, 15))
        fp1 = afwDetect.Footprint(afwGeom.SpanSet.fromShape(ellipse))
        fp1.addPeak(6, 7, 2)
        fp1.addPeak(8, 9, 3)
        with lsst.utils.tests.getTempFilePath(".fits") as tmpFile:
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

    def testBbox(self):
        """Add Spans and check bounding box"""
        foot = afwDetect.Footprint()
        spanLists = [afwGeom.Span(10, 100, 105), afwGeom.Span(11, 99, 104)]
        spanSet = afwGeom.SpanSet(spanLists)
        foot.spans = spanSet

        bbox = foot.getBBox()
        self.assertEqual(bbox.getWidth(), 7)
        self.assertEqual(bbox.getHeight(), 2)
        self.assertEqual(bbox.getMinX(), 99)
        self.assertEqual(bbox.getMinY(), 10)
        self.assertEqual(bbox.getMaxX(), 105)
        self.assertEqual(bbox.getMaxY(), 11)
        # clip with a bbox that doesn't overlap at all
        bbox2 = lsst.geom.Box2I(lsst.geom.Point2I(5, 90), lsst.geom.Extent2I(1, 2))
        foot.clipTo(bbox2)
        self.assertTrue(foot.getBBox().isEmpty())
        self.assertEqual(foot.getArea(), 0)

    def testFootprintFromBBox1(self):
        """Create a rectangular Footprint"""
        x0, y0, w, h = 9, 10, 7, 4
        spanSet = afwGeom.SpanSet(lsst.geom.Box2I(lsst.geom.Point2I(x0, y0),
                                                  lsst.geom.Extent2I(w, h)))
        foot = afwDetect.Footprint(spanSet)

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
        spanSet = afwGeom.SpanSet(lsst.geom.Box2I(lsst.geom.Point2I(x0, y0),
                                                  lsst.geom.Extent2I(w, h)))
        foot = afwDetect.Footprint(spanSet)
        bbox = foot.getBBox()

        dx, dy = 10, 20
        bbox.shift(lsst.geom.Extent2I(dx, dy))

        self.assertEqual(bbox.getMinX(), x0 + dx)
        self.assertEqual(foot.getBBox().getMinX(), x0)

    def testFootprintFromCircle(self):
        """Create an elliptical Footprint"""
        ellipse = afwGeom.Ellipse(afwGeomEllipses.Axes(6, 6, 0),
                                  lsst.geom.Point2D(9, 15))
        spanSet = afwGeom.SpanSet.fromShape(ellipse)
        foot = afwDetect.Footprint(spanSet,
                                   lsst.geom.Box2I(lsst.geom.Point2I(0, 0),
                                                   lsst.geom.Extent2I(20, 30)))

        idImage = afwImage.ImageU(
            lsst.geom.Extent2I(foot.getRegion().getWidth(),
                               foot.getRegion().getHeight()))
        idImage.set(0)

        foot.spans.setImage(idImage, foot.getId())

        if False:
            ds9.mtv(idImage, frame=2)

    def testFootprintFromEllipse(self):
        """Create an elliptical Footprint"""
        cen = lsst.geom.Point2D(23, 25)
        a, b, theta = 25, 15, 30
        ellipse = afwGeom.Ellipse(
            afwGeomEllipses.Axes(a, b, math.radians(theta)),
            cen)
        spanSet = afwGeom.SpanSet.fromShape(ellipse)
        foot = afwDetect.Footprint(spanSet,
                                   lsst.geom.Box2I(lsst.geom.Point2I(0, 0),
                                                   lsst.geom.Extent2I(50, 60)))

        idImage = afwImage.ImageU(lsst.geom.Extent2I(
            foot.getRegion().getWidth(), foot.getRegion().getHeight()))
        idImage.set(0)

        foot.spans.setImage(idImage, foot.getId())

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
        self.assertLess(abs(a - axes.getA()), 0.15, "a: %g v. %g" % (a, axes.getA()))
        self.assertLess(abs(b - axes.getB()), 0.02, "b: %g v. %g" % (b, axes.getB()))
        self.assertLess(abs(theta - math.degrees(axes.getTheta())), 0.2,
                        "theta: %g v. %g" % (theta, math.degrees(axes.getTheta())))

    def testCopy(self):
        bbox = lsst.geom.BoxI(lsst.geom.PointI(0, 2), lsst.geom.PointI(5, 6))

        fp = afwDetect.Footprint(afwGeom.SpanSet(bbox), bbox)

        # test copy construct
        fp2 = afwDetect.Footprint(fp)

        self.assertEqual(fp2.getBBox(), bbox)
        self.assertEqual(fp2.getRegion(), bbox)
        self.assertEqual(fp2.getArea(), bbox.getArea())

        y = bbox.getMinY()
        for s in fp2.getSpans():
            self.assertEqual(s.getY(), y)
            self.assertEqual(s.getX0(), bbox.getMinX())
            self.assertEqual(s.getX1(), bbox.getMaxX())
            y += 1

        # test assignment
        fp3 = afwDetect.Footprint()
        fp3.assign(fp)
        self.assertEqual(fp3.getBBox(), bbox)
        self.assertEqual(fp3.getRegion(), bbox)
        self.assertEqual(fp3.getArea(), bbox.getArea())

        y = bbox.getMinY()
        for s in fp3.getSpans():
            self.assertEqual(s.getY(), y)
            self.assertEqual(s.getX0(), bbox.getMinX())
            self.assertEqual(s.getX1(), bbox.getMaxX())
            y += 1

    def testShrink(self):
        width, height = 5, 10  # Size of footprint
        x0, y0 = 50, 50  # Position of footprint
        imwidth, imheight = 100, 100  # Size of image

        spanSet = afwGeom.SpanSet(lsst.geom.Box2I(lsst.geom.Point2I(x0, y0),
                                                  lsst.geom.Extent2I(width, height)))
        region = lsst.geom.Box2I(lsst.geom.Point2I(0, 0),
                                 lsst.geom.Extent2I(imwidth, imheight))
        foot = afwDetect.Footprint(spanSet, region)
        self.assertEqual(foot.getArea(), width*height)

        # Add some peaks to the original footprint and check that those lying outside
        # the shrunken footprint are omitted from the returned shrunken footprint.
        foot.addPeak(50, 50, 1)  # should be omitted in shrunken footprint
        foot.addPeak(52, 52, 2)  # should be kept in shrunken footprint
        foot.addPeak(50, 59, 3)  # should be omitted in shrunken footprint
        self.assertEqual(len(foot.getPeaks()), 3)  # check that all three peaks were added

        # Shrinking by one pixel makes each dimension *two* pixels shorter.
        shrunk = afwDetect.Footprint().assign(foot)
        shrunk.erode(1)
        self.assertEqual(3*8, shrunk.getArea())

        # Shrunken footprint should now only contain one peak at (52, 52)
        self.assertEqual(len(shrunk.getPeaks()), 1)
        peak = shrunk.getPeaks()[0]
        self.assertEqual((peak.getIx(), peak.getIy()), (52, 52))

        # Without shifting the centroid
        self.assertEqual(shrunk.getCentroid(), foot.getCentroid())

        # Get the same result from a Manhattan shrink
        shrunk = afwDetect.Footprint().assign(foot)
        shrunk.erode(1, afwGeom.Stencil.MANHATTAN)
        self.assertEqual(3*8, shrunk.getArea())
        self.assertEqual(shrunk.getCentroid(), foot.getCentroid())

        # Shrinking by a large amount leaves nothing.
        shrunkToNothing = afwDetect.Footprint().assign(foot)
        shrunkToNothing.erode(100)
        self.assertEqual(shrunkToNothing.getArea(), 0)

    def testShrinkIsoVsManhattan(self):
        # Demonstrate that isotropic and Manhattan shrinks are different.
        radius = 8
        imwidth, imheight = 100, 100
        x0, y0 = imwidth//2, imheight//2
        nshrink = 4

        ellipse = afwGeom.Ellipse(
            afwGeomEllipses.Axes(1.5*radius, 2*radius, 0),
            lsst.geom.Point2D(x0, y0))
        spanSet = afwGeom.SpanSet.fromShape(ellipse)
        foot = afwDetect.Footprint(
            spanSet,
            lsst.geom.Box2I(lsst.geom.Point2I(0, 0),
                            lsst.geom.Extent2I(imwidth, imheight)))
        footIsotropic = afwDetect.Footprint()
        footIsotropic.assign(foot)

        foot.erode(nshrink, afwGeom.Stencil.MANHATTAN)
        footIsotropic.erode(nshrink)
        self.assertNotEqual(foot, footIsotropic)

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
        initial_npix = circle_npix * 2 - 1  # touch at one pixel
        shrunk_npix = 26

        box = lsst.geom.Box2I(lsst.geom.Point2I(0, 0),
                              lsst.geom.Extent2I(imwidth, imheight))

        e1 = afwGeom.Ellipse(afwGeomEllipses.Axes(radius, radius, 0),
                             lsst.geom.Point2D(x1, y1))
        spanSet1 = afwGeom.SpanSet.fromShape(e1)
        f1 = afwDetect.Footprint(spanSet1, box)
        self.assertEqual(f1.getArea(), circle_npix)

        e2 = afwGeom.Ellipse(afwGeomEllipses.Axes(radius, radius, 0),
                             lsst.geom.Point2D(x2, y2))
        spanSet2 = afwGeom.SpanSet.fromShape(e2)
        f2 = afwDetect.Footprint(spanSet2, box)
        self.assertEqual(f2.getArea(), circle_npix)

        initial = afwDetect.mergeFootprints(f1, f2)
        initial.setRegion(f2.getRegion())  # merge does not propagate the region
        self.assertEqual(initial_npix, initial.getArea())

        shrunk = afwDetect.Footprint().assign(initial)
        shrunk.erode(nshrink)
        self.assertEqual(shrunk_npix, shrunk.getArea())

        if display:
            idImage = afwImage.ImageU(imwidth, imheight)
            for i, foot in enumerate([initial, shrunk]):
                print(foot.getArea())
                foot.spans.setImage(idImage, i+1)
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
        spanSet = afwGeom.SpanSet(lsst.geom.Box2I(lsst.geom.Point2I(x0, y0),
                                                  lsst.geom.Extent2I(width, height)))
        foot1 = afwDetect.Footprint(spanSet,
                                    lsst.geom.Box2I(lsst.geom.Point2I(0, 0),
                                                    lsst.geom.Extent2I(100, 100)))

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
            foot2 = afwDetect.Footprint().assign(foot1)
            stencil = afwGeom.Stencil.CIRCLE if isotropic else \
                afwGeom.Stencil.MANHATTAN
            foot2.dilate(ngrow, stencil)

            # Check that the grown footprint is bigger than the original
            self.assertGreater(foot2.getArea(), foot1.getArea())

            # Check that peaks got copied into grown footprint
            self.assertEqual(len(foot2.getPeaks()), 3)
            for peak in foot2.getPeaks():
                self.assertIn((peak.getIx(), peak.getIy()),
                              [(20, 20), (30, 35), (25, 45)])

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
        region = lsst.geom.Box2I(lsst.geom.Point2I(0, 0), lsst.geom.Extent2I(12, 10))
        foot = afwDetect.Footprint(afwGeom.SpanSet(), region)
        spanList = [afwGeom.Span(*span) for span in ((3, 3, 5), (3, 7, 7),
                                                     (4, 2, 3), (4, 5, 7),
                                                     (5, 2, 3), (5, 5, 8),
                                                     (6, 3, 5))]
        foot.spans = afwGeom.SpanSet(spanList)

        idImage = afwImage.ImageU(region.getDimensions())
        idImage.set(0)

        foot.spans.setImage(idImage, 1)
        if display:
            ds9.mtv(idImage)

        idImageFromBBox = idImage.Factory(idImage, True)
        idImageFromBBox.set(0)
        bboxes = afwDetect.footprintToBBoxList(foot)
        for bbox in bboxes:
            x0, y0, x1, y1 = bbox.getMinX(), bbox.getMinY(), \
                bbox.getMaxX(), bbox.getMaxY()

            for y in range(y0, y1 + 1):
                for x in range(x0, x1 + 1):
                    idImageFromBBox.set(x, y, 1)

            if display:
                x0 -= 0.5
                y0 -= 0.5
                x1 += 0.5
                y1 += 0.5

                ds9.line([(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)],
                         ctype=ds9.RED)

        idImageFromBBox -= idImage      # should be blank
        stats = afwMath.makeStatistics(idImageFromBBox, afwMath.MAX)

        self.assertEqual(stats.getValue(), 0)

    def testWriteDefect(self):
        """Write a Footprint as a set of Defects"""
        region = lsst.geom.Box2I(lsst.geom.Point2I(0, 0), lsst.geom.Extent2I(12, 10))
        spanSet = afwGeom.SpanSet([afwGeom.Span(*span) for span in [(3, 3, 5),
                                                                    (3, 7, 7),
                                                                    (4, 2, 3),
                                                                    (4, 5, 7),
                                                                    (5, 2, 3),
                                                                    (5, 5, 8),
                                                                    (6, 3, 5)]])
        foot = afwDetect.Footprint(spanSet, region)

        openedFile = False
        if True:
            fd = open("/dev/null", "w")
            openedFile = True
        else:
            fd = sys.stdout

        afwDetectUtils.writeFootprintAsDefects(fd, foot)
        if openedFile:
            fd.close()

    def testSetFromFootprint(self):
        """Test setting mask/image pixels from a Footprint list"""
        mi = afwImage.MaskedImageF(lsst.geom.Extent2I(12, 8))
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
        for ft in objects:
            ft.spans.setImage(mi.getImage(), 5.0)
        self.assertEqual(mi.getImage().get(4, 2), 10)
        self.assertEqual(mi.getImage().get(3, 6), 5)

        if display:
            ds9.mtv(mi, frame=1)
        #
        # Check Footprint.contains() while we are about it
        #
        self.assertTrue(objects[0].contains(lsst.geom.Point2I(7, 5)))
        self.assertFalse(objects[0].contains(lsst.geom.Point2I(10, 6)))
        self.assertFalse(objects[0].contains(lsst.geom.Point2I(7, 6)))
        self.assertFalse(objects[0].contains(lsst.geom.Point2I(4, 2)))

        self.assertTrue(objects[1].contains(lsst.geom.Point2I(3, 6)))

        # Verify the FootprintSet footprint list setter can accept inputs from
        # the footprint list getter
        # Create a copy of the ds' FootprintList
        dsFpList = ds.getFootprints()
        footprintListCopy = [afwDetect.Footprint().assign(f) for f in dsFpList]
        # Use the FootprintList setter with the output from the getter
        ds.setFootprints(ds.getFootprints()[:-1])
        dsFpListNew = ds.getFootprints()
        self.assertTrue(len(dsFpListNew) == len(footprintListCopy)-1)
        for new, old in zip(dsFpListNew, footprintListCopy[:-1]):
            self.assertEqual(new, old)

    def testMakeFootprintSetXY0(self):
        """Test setting mask/image pixels from a Footprint list"""
        mi = afwImage.MaskedImageF(lsst.geom.Extent2I(12, 8))
        im = mi.getImage()
        im.set(100)

        mi.setXY0(lsst.geom.PointI(2, 2))
        afwDetect.FootprintSet(mi, afwDetect.Threshold(1), "DETECTED")

        bitmask = mi.getMask().getPlaneBitMask("DETECTED")
        for y in range(im.getHeight()):
            for x in range(im.getWidth()):
                self.assertEqual(mi.getMask().get(x, y), bitmask)

    def testTransform(self):
        dims = lsst.geom.Extent2I(512, 512)
        bbox = lsst.geom.Box2I(lsst.geom.Point2I(0, 0), dims)
        radius = 5
        offset = lsst.geom.Extent2D(123, 456)
        crval = lsst.geom.SpherePoint(0, 0, lsst.geom.degrees)
        crpix = lsst.geom.Point2D(0, 0)
        cdMatrix = np.array([1.0e-5, 0.0, 0.0, 1.0e-5])
        cdMatrix.shape = (2, 2)
        source = afwGeom.makeSkyWcs(crval=crval, crpix=crpix, cdMatrix=cdMatrix)
        target = afwGeom.makeSkyWcs(crval=crval, crpix=crpix + offset, cdMatrix=cdMatrix)
        sourceSpanSet = afwGeom.SpanSet.fromShape(radius,
                                                  afwGeom.Stencil.CIRCLE)
        sourceSpanSet = sourceSpanSet.shiftedBy(12, 34)
        fpSource = afwDetect.Footprint(sourceSpanSet, bbox)

        fpTarget = fpSource.transform(source, target, bbox)

        self.assertEqual(len(fpSource.getSpans()), len(fpTarget.getSpans()))
        self.assertEqual(fpSource.getArea(), fpTarget.getArea())
        imSource = afwImage.ImageU(dims)
        fpSource.spans.setImage(imSource, 1)

        imTarget = afwImage.ImageU(dims)
        fpTarget.spans.setImage(imTarget, 1)

        subSource = imSource.Factory(imSource, fpSource.getBBox())
        subTarget = imTarget.Factory(imTarget, fpTarget.getBBox())
        self.assertTrue(np.all(subSource.getArray() == subTarget.getArray()))

        # make a bbox smaller than the target footprint
        bbox2 = lsst.geom.Box2I(fpTarget.getBBox())
        bbox2.grow(-1)
        fpTarget2 = fpSource.transform(source, target, bbox2)  # this one clips
        fpTarget3 = fpSource.transform(source, target, bbox2, False)  # this one doesn't
        self.assertTrue(bbox2.contains(fpTarget2.getBBox()))
        self.assertFalse(bbox2.contains(fpTarget3.getBBox()))
        self.assertNotEqual(fpTarget.getArea(), fpTarget2.getArea())
        self.assertEqual(fpTarget.getArea(), fpTarget3.getArea())

        # Test that peakCatalogs get Transformed correctly
        truthList = [(x, y, 10) for x, y in zip(range(-2, 2), range(-1, 3))]
        for value in truthList:
            fpSource.addPeak(*value)
        scaleFactor = 2
        linTrans = lsst.geom.LinearTransform(np.matrix([[scaleFactor, 0],
                                                        [0, scaleFactor]],
                                                       dtype=float))
        linTransFootprint = fpSource.transform(linTrans, fpSource.getBBox(),
                                               False)
        for peak, truth in zip(linTransFootprint.peaks, truthList):
            # Multiplied by two because that is the linear transform scaling
            # factor
            self.assertEqual(peak.getIx(), truth[0]*scaleFactor)
            self.assertEqual(peak.getIy(), truth[1]*scaleFactor)

    def testCopyWithinFootprintImage(self):
        W, H = 10, 10
        dims = lsst.geom.Extent2I(W, H)
        source = afwImage.ImageF(dims)
        dest = afwImage.ImageF(dims)
        sa = source.getArray()
        for i in range(H):
            for j in range(W):
                sa[i, j] = 100 * i + j

        footSpans = [s for s in self.foot.spans]
        footSpans.append(afwGeom.Span(4, 3, 6))
        footSpans.append(afwGeom.Span(5, 2, 4))
        self.foot.spans = afwGeom.SpanSet(footSpans)

        self.foot.spans.copyImage(source, dest)

        da = dest.getArray()
        self.assertEqual(da[4, 2], 0)
        self.assertEqual(da[4, 3], 403)
        self.assertEqual(da[4, 4], 404)
        self.assertEqual(da[4, 5], 405)
        self.assertEqual(da[4, 6], 406)
        self.assertEqual(da[4, 7], 0)
        self.assertEqual(da[5, 1], 0)
        self.assertEqual(da[5, 2], 502)
        self.assertEqual(da[5, 3], 503)
        self.assertEqual(da[5, 4], 504)
        self.assertEqual(da[5, 5], 0)
        self.assertTrue(np.all(da[:4, :] == 0))
        self.assertTrue(np.all(da[6:, :] == 0))

    def testCopyWithinFootprintOutside(self):
        """Copy a footprint that is larger than the image"""
        target = afwImage.ImageF(100, 100)
        target.set(0)
        subTarget = afwImage.ImageF(target, lsst.geom.Box2I(lsst.geom.Point2I(40, 40),
                                                            lsst.geom.Extent2I(20, 20)))
        source = afwImage.ImageF(10, 30)
        source.setXY0(45, 45)
        source.set(1.0)

        foot = afwDetect.Footprint()
        spanList = [afwGeom.Span(*s) for s in (
            (50, 50, 60),  # Oversized on the source image, right; only some pixels overlap
            (60, 0, 100),  # Oversized on the source, left and right; and on sub-target image, top
            (99, 0, 1000),  # Oversized on the source image, top, left and right; aiming for segfault
        )]
        foot.spans = afwGeom.SpanSet(spanList)

        foot.spans.clippedTo(subTarget.getBBox()).clippedTo(source.getBBox()).\
            copyImage(source, subTarget)

        expected = np.zeros((100, 100))
        expected[50, 50:55] = 1.0

        self.assertTrue(np.all(target.getArray() == expected))

    def testCopyWithinFootprintMaskedImage(self):
        W, H = 10, 10
        dims = lsst.geom.Extent2I(W, H)
        source = afwImage.MaskedImageF(dims)
        dest = afwImage.MaskedImageF(dims)
        sa = source.getImage().getArray()
        sv = source.getVariance().getArray()
        sm = source.getMask().getArray()
        for i in range(H):
            for j in range(W):
                sa[i, j] = 100 * i + j
                sv[i, j] = 100 * j + i
                sm[i, j] = 1

        footSpans = [s for s in self.foot.spans]
        footSpans.append(afwGeom.Span(4, 3, 6))
        footSpans.append(afwGeom.Span(5, 2, 4))
        self.foot.spans = afwGeom.SpanSet(footSpans)

        self.foot.spans.copyMaskedImage(source, dest)

        da = dest.getImage().getArray()
        dv = dest.getVariance().getArray()
        dm = dest.getMask().getArray()

        self.assertEqual(da[4, 2], 0)
        self.assertEqual(da[4, 3], 403)
        self.assertEqual(da[4, 4], 404)
        self.assertEqual(da[4, 5], 405)
        self.assertEqual(da[4, 6], 406)
        self.assertEqual(da[4, 7], 0)
        self.assertEqual(da[5, 1], 0)
        self.assertEqual(da[5, 2], 502)
        self.assertEqual(da[5, 3], 503)
        self.assertEqual(da[5, 4], 504)
        self.assertEqual(da[5, 5], 0)
        self.assertTrue(np.all(da[:4, :] == 0))
        self.assertTrue(np.all(da[6:, :] == 0))

        self.assertEqual(dv[4, 2], 0)
        self.assertEqual(dv[4, 3], 304)
        self.assertEqual(dv[4, 4], 404)
        self.assertEqual(dv[4, 5], 504)
        self.assertEqual(dv[4, 6], 604)
        self.assertEqual(dv[4, 7], 0)
        self.assertEqual(dv[5, 1], 0)
        self.assertEqual(dv[5, 2], 205)
        self.assertEqual(dv[5, 3], 305)
        self.assertEqual(dv[5, 4], 405)
        self.assertEqual(dv[5, 5], 0)
        self.assertTrue(np.all(dv[:4, :] == 0))
        self.assertTrue(np.all(dv[6:, :] == 0))

        self.assertTrue(np.all(dm[4, 3:7] == 1))
        self.assertTrue(np.all(dm[5, 2:5] == 1))
        self.assertTrue(np.all(dm[:4, :] == 0))
        self.assertTrue(np.all(dm[6:, :] == 0))
        self.assertTrue(np.all(dm[4, :3] == 0))
        self.assertTrue(np.all(dm[4, 7:] == 0))

    def testMergeFootprints(self):
        f1 = self.foot
        f2 = afwDetect.Footprint()

        spanList1 = [(10, 10, 20),
                     (10, 30, 40),
                     (10, 50, 60),
                     (11, 30, 50),
                     (12, 30, 50),
                     (13, 10, 20),
                     (13, 30, 40),
                     (13, 50, 60),
                     (15, 10, 20),
                     (15, 31, 40),
                     (15, 51, 60)]
        spanSet1 = afwGeom.SpanSet([afwGeom.Span(*span) for span in spanList1])
        f1.spans = spanSet1

        spanList2 = [(8, 10, 20),
                     (9, 20, 30),
                     (10, 0, 9),
                     (10, 35, 65),
                     (10, 70, 80),
                     (13, 49, 54),
                     (14, 10, 30),
                     (15, 21, 30),
                     (15, 41, 50),
                     (15, 61, 70)]
        spanSet2 = afwGeom.SpanSet([afwGeom.Span(*span) for span in spanList2])
        f2.spans = spanSet2

        fA = afwDetect.mergeFootprints(f1, f2)
        fB = afwDetect.mergeFootprints(f2, f1)

        ims = []
        for i, f in enumerate([f1, f2, fA, fB]):
            im1 = afwImage.ImageU(100, 100)
            im1.set(0)
            imbb = im1.getBBox()
            f.setRegion(imbb)
            f.spans.setImage(im1, 1)
            ims.append(im1)

        for i, merged in enumerate([ims[2], ims[3]]):
            m = merged.getArray()
            a1 = ims[0].getArray()
            a2 = ims[1].getArray()
            # Slightly looser tests to start...
            # Every pixel in f1 is in f[AB]
            self.assertTrue(np.all(m.flat[np.flatnonzero(a1)] == 1))
            # Every pixel in f2 is in f[AB]
            self.assertTrue(np.all(m.flat[np.flatnonzero(a2)] == 1))
            # merged == a1 | a2.
            self.assertTrue(np.all(m == np.maximum(a1, a2)))

        if False:
            import matplotlib
            matplotlib.use('Agg')
            import pylab as plt
            plt.clf()
            for i, im1 in enumerate(ims):
                plt.subplot(4, 1, i+1)
                plt.imshow(im1.getArray(), interpolation='nearest',
                           origin='lower')
                plt.axis([0, 100, 0, 20])
            plt.savefig('merge2.png')

    def testPeakSort(self):
        spanSet = afwGeom.SpanSet(lsst.geom.Box2I(lsst.geom.Point2I(0, 0),
                                                  lsst.geom.Point2I(10, 10)))
        footprint = afwDetect.Footprint(spanSet)
        footprint.addPeak(4, 5, 1)
        footprint.addPeak(3, 2, 5)
        footprint.addPeak(7, 8, -2)
        footprint.addPeak(5, 7, 4)
        footprint.sortPeaks()
        self.assertEqual([peak.getIx() for peak in footprint.getPeaks()],
                         [3, 5, 4, 7])

    def testInclude(self):
        """Test that we can expand a Footprint to include the union of itself and all others provided."""
        region = lsst.geom.Box2I(lsst.geom.Point2I(-6, -6), lsst.geom.Point2I(6, 6))
        parentSpanSet = afwGeom.SpanSet(lsst.geom.Box2I(lsst.geom.Point2I(-2, -2),
                                                        lsst.geom.Point2I(2, 2)))
        parent = afwDetect.Footprint(parentSpanSet, region)
        parent.addPeak(0, 0, float("NaN"))
        child1SpanSet = afwGeom.SpanSet(lsst.geom.Box2I(lsst.geom.Point2I(-3, 0),
                                                        lsst.geom.Point2I(0, 3)))
        child1 = afwDetect.Footprint(child1SpanSet, region)
        child1.addPeak(-1, 1, float("NaN"))
        child2SpanSet = afwGeom.SpanSet(lsst.geom.Box2I(lsst.geom.Point2I(-4, -3),
                                                        lsst.geom.Point2I(-1, 0)))
        child2 = afwDetect.Footprint(child2SpanSet, region)
        child3SpanSet = afwGeom.SpanSet(lsst.geom.Box2I(lsst.geom.Point2I(4, -1),
                                                        lsst.geom.Point2I(6, 1)))
        child3 = afwDetect.Footprint(child3SpanSet)
        merge123 = afwDetect.Footprint(parent)
        merge123.spans = merge123.spans.union(child1.spans).union(child2.spans).union(child3.spans)
        self.assertTrue(merge123.getBBox().contains(parent.getBBox()))
        self.assertTrue(merge123.getBBox().contains(child1.getBBox()))
        self.assertTrue(merge123.getBBox().contains(child2.getBBox()))
        self.assertTrue(merge123.getBBox().contains(child3.getBBox()))
        mask123a = afwImage.Mask(region)
        mask123b = afwImage.Mask(region)
        parent.spans.setMask(mask123a, 1)
        child1.spans.setMask(mask123a, 1)
        child2.spans.setMask(mask123a, 1)
        child3.spans.setMask(mask123a, 1)
        merge123.spans.setMask(mask123b, 1)
        self.assertEqual(mask123a.getArray().sum(), merge123.getArea())
        self.assertFloatsAlmostEqual(mask123a.getArray(), mask123b.getArray(),
                                     rtol=0, atol=0)

        # Test that ignoreSelf=True works for include
        childOnly = afwDetect.Footprint()
        childOnly.spans = childOnly.spans.union(child1.spans).union(child2.spans).union(child3.spans)
        merge123 = afwDetect.Footprint(parent)
        merge123.spans = child1.spans.union(child2.spans).union(child3.spans)
        maskChildren = afwImage.Mask(region)
        mask123 = afwImage.Mask(region)
        childOnly.spans.setMask(maskChildren, 1)
        merge123.spans.setMask(mask123, 1)
        self.assertTrue(np.all(maskChildren.getArray() == mask123.getArray()))

    def checkEdge(self, footprint):
        """Check that Footprint::findEdgePixels() works"""
        bbox = footprint.getBBox()
        bbox.grow(3)

        def makeImage(area):
            """Make an ImageF with 1 in the footprint, and 0 elsewhere"""
            ones = afwImage.ImageI(bbox)
            ones.set(1)
            image = afwImage.ImageI(bbox)
            image.set(0)
            if isinstance(area, afwDetect.Footprint):
                area.spans.copyImage(ones, image)
            if isinstance(area, afwGeom.SpanSet):
                area.copyImage(ones, image)
            return image

        edges = self.foot.spans.findEdgePixels()
        edgeImage = makeImage(edges)

        # Find edges with an edge-detection kernel
        image = makeImage(self.foot)
        kernel = afwImage.ImageD(3, 3)
        kernel.set(1, 1, 4)
        for x, y in [(1, 2), (0, 1), (1, 0), (2, 1)]:
            kernel.set(x, y, -1)
        kernel.setXY0(1, 1)
        result = afwImage.ImageI(bbox)
        result.set(0)
        afwMath.convolve(result, image, afwMath.FixedKernel(kernel),
                         afwMath.ConvolutionControl(False))
        result.getArray().__imul__(image.getArray())
        trueEdges = np.where(result.getArray() > 0, 1, 0)

        self.assertTrue(np.all(trueEdges == edgeImage.getArray()))

    def testEdge(self):
        """Test for Footprint::findEdgePixels()"""
        foot = afwDetect.Footprint()
        spanList = [afwGeom.Span(*span) for span in ((3, 3, 9),
                                                     (4, 2, 4),
                                                     (4, 6, 7),
                                                     (4, 9, 11),
                                                     (5, 3, 9),
                                                     (6, 6, 7))]
        foot.spans = afwGeom.SpanSet(spanList)
        self.checkEdge(foot)

        # This footprint came from a very large Footprint in a deep HSC coadd patch
        self.checkEdge(afwDetect.Footprint.readFits(
            os.path.join(testPath, "testFootprintEdge.fits")))


class FootprintSetTestCase(unittest.TestCase):
    """A test case for FootprintSet"""

    def setUp(self):
        self.ms = afwImage.MaskedImageF(lsst.geom.Extent2I(12, 8))
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
        afwDetect.FootprintSet(afwImage.MaskedImageF(lsst.geom.Extent2I(10, 20)),
                               afwDetect.Threshold(10))

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

        self.ms.set(2, 4, (10, 0x0, 36))  # not detected (high variance)

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
                    self.assertEqual(mask.get(x, sp.getY()),
                                     mask.getPlaneBitMask("OBJECT"))

    def testFootprintsImageId(self):
        """Check that we can insert footprints into an Image"""
        ds = afwDetect.FootprintSet(self.ms, afwDetect.Threshold(10))
        objects = ds.getFootprints()

        idImage = afwImage.ImageU(self.ms.getDimensions())
        idImage.set(0)

        for foot in objects:
            foot.spans.setImage(idImage, foot.getId())

        if False:
            ds9.mtv(idImage, frame=2)

        for i in range(len(objects)):
            for sp in objects[i].getSpans():
                for x in range(sp.getX0(), sp.getX1() + 1):
                    self.assertEqual(idImage.get(x, sp.getY()),
                                     objects[i].getId())

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
            foot.dilate(3, afwGeom.Stencil.MANHATTAN)
            foot.spans.setImage(idImage, i, doClip=True)
            i += 1

        if display:
            ds9.mtv(self.ms, frame=0)
            ds9.mtv(idImage, frame=1)

    def testFootprintPeaks(self):
        """Test that we can extract the peaks from a Footprint"""
        fs = afwDetect.FootprintSet(self.ms, afwDetect.Threshold(10), "OBJECT")

        foot = fs.getFootprints()[0]

        self.assertEqual(len(foot.getPeaks()), 5)


class MaskFootprintSetTestCase(unittest.TestCase):
    """A test case for generating FootprintSet from Masks"""

    def setUp(self):
        self.mim = afwImage.MaskedImageF(lsst.geom.ExtentI(12, 8))
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
        ds = afwDetect.FootprintSet(self.mim.getMask(),
                                    afwDetect.createThreshold(level, "bitmask"))

        objects = ds.getFootprints()

        if 0 and display:
            ds9.mtv(self.mim, frame=0)

        self.assertEqual(len(objects),
                         len([o for o in self.objects if (o.val & level)]))

        i = 0
        for o in self.objects:
            if o.val & level:
                self.assertEqual(o, objects[i])
                i += 1


class NaNFootprintSetTestCase(unittest.TestCase):
    """A test case for FootprintSet when the image contains NaNs"""

    def setUp(self):
        self.ms = afwImage.MaskedImageF(lsst.geom.Extent2I(12, 8))
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

        # connects the two objects with value==20 together if NaN is detected
        im.set(9, 6, self.NaN)

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


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
