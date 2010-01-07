#!/usr/bin/env python
"""
Tests for Footprints, and FootprintSets

Run with:
   Footprint_1.py
or
   python
   >>> import Footprint_1; Footprint_1.run()
"""

import pdb                              # we may want to say pdb.set_trace()
import sys
import unittest
import lsst.utils.tests as tests
import lsst.pex.logging as logging
import lsst.afw.image.imageLib as afwImage
import lsst.afw.math.mathLib as afwMath
import lsst.afw.detection.detectionLib as afwDetect
import lsst.afw.detection.utils as afwDetectUtils
import lsst.afw.display.ds9 as ds9

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

class ThresholdTestCase(unittest.TestCase):
    def testTresholdFactory(self):
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
            self.fail("Failed to build Threshold with proper parameters")
        

class FootprintTestCase(unittest.TestCase):
    """A test case for Footprint"""
    def setUp(self):
        self.foot = afwDetect.Footprint()

    def tearDown(self):
        del self.foot

    def testToString(self):
        y, x0, x1 = 10, 100, 101
        s = afwDetect.Span(y, x0, x1)
        self.assertEqual(s.toString(), toString(y, x0, x1))

    def testSetBbox(self):
        """Test setBBox"""
        
        self.assertEqual(self.foot.setBBox(), None)

    def testGC(self):
        """Check that Footprints are automatically garbage collected (when MemoryTestCase runs)"""
        
        f = afwDetect.Footprint()

    def testId(self):
        """Test uniqueness of IDs"""
        
        self.assertNotEqual(self.foot.getId(), afwDetect.Footprint().getId())

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
        self.assertEqual(bbox.getX0(), 99)
        self.assertEqual(bbox.getY0(), 10)
        self.assertEqual(bbox.getX1(), 105)
        self.assertEqual(bbox.getY1(), 11)

    def testSpanShift(self):
        """Test our ability to shift spans"""

        span = afwDetect.Span(10, 100, 105)
        foot = afwDetect.Footprint()

        foot.addSpan(span, 1, 2)

        bbox = foot.getBBox()
        self.assertEqual(bbox.getWidth(), 6)
        self.assertEqual(bbox.getHeight(), 1)
        self.assertEqual(bbox.getX0(), 101)
        self.assertEqual(bbox.getY0(), 12)
        #
        # Shift that span using Span.shift
        #
        foot = afwDetect.Footprint()
        span.shift(-1, -2)
        foot.addSpan(span)

        bbox = foot.getBBox()
        self.assertEqual(bbox.getWidth(), 6)
        self.assertEqual(bbox.getHeight(), 1)
        self.assertEqual(bbox.getX0(), 99)
        self.assertEqual(bbox.getY0(), 8)

    def testFootprintFromBBox1(self):
        """Create a rectangular Footprint"""
        x0, y0, w, h = 9, 10, 7, 4
        foot = afwDetect.Footprint(afwImage.BBox(afwImage.PointI(x0, y0), w, h))

        bbox = foot.getBBox()

        self.assertEqual(bbox.getWidth(), w)
        self.assertEqual(bbox.getHeight(), h)
        self.assertEqual(bbox.getX0(), x0)
        self.assertEqual(bbox.getY0(), y0)
        self.assertEqual(bbox.getX1(), x0 + w - 1)
        self.assertEqual(bbox.getY1(), y0 + h - 1)

        idImage = afwImage.ImageU(foot.getRegion().getDimensions())
        idImage.set(0)
        
        foot.insertIntoImage(idImage, foot.getId())

        if False:
            ds9.mtv(idImage, frame=2)

    def testGetBBox(self):
        """Check that Footprint.getBBox() returns a copy"""
        
        x0, y0, w, h = 9, 10, 7, 4
        foot = afwDetect.Footprint(afwImage.BBox(afwImage.PointI(x0, y0), w, h))
        bbox = foot.getBBox()

        dx, dy = 10, 20
        bbox.shift(dx, dy)

        self.assertEqual(bbox.getX0(), x0 + dx)
        self.assertEqual(foot.getBBox().getX0(), x0)

    def testBCircle2i(self):
        """Test the BCircle2i constructors"""
        
        x = 100
        y = 200
        r = 1.5
        
        bc = afwImage.BCircle(afwImage.PointI(x, y), r)
        for i in range(2):
            c = bc.getCenter()
            self.assertEqual(c.getX(), x)
            self.assertEqual(c.getY(), y)
            self.assertAlmostEqual(bc.getRadius(), r)

            bc = afwImage.BCircle(afwImage.PointI(x, y), r)

    def testFootprintFromBCircle(self):
        """Create a circular Footprint"""

        foot = afwDetect.Footprint(afwImage.BCircle(afwImage.PointI(9, 15), 6),
                                      afwImage.BBox(afwImage.PointI(0, 0), 20, 30))

        idImage = afwImage.ImageU(foot.getRegion().getWidth(), foot.getRegion().getHeight())
        idImage.set(0)
        
        foot.insertIntoImage(idImage, foot.getId())

        if False:
            ds9.mtv(idImage, frame=2)

    def testGrow(self):
        """Test growing a footprint"""
        x0, y0 = 20, 20
        width, height = 20, 30
        foot1 = afwDetect.Footprint(afwImage.BBox(afwImage.PointI(x0, y0), width, height),
                                       afwImage.BBox(afwImage.PointI(0, 0), 100, 100))
        bbox1 = foot1.getBBox()

        self.assertEqual(bbox1.getX0(), x0)
        self.assertEqual(bbox1.getX1(), x0 + width - 1)
        self.assertEqual(bbox1.getWidth(), width)

        self.assertEqual(bbox1.getY0(), y0)
        self.assertEqual(bbox1.getY1(), y0 + height - 1)
        self.assertEqual(bbox1.getHeight(), height)

        ngrow = 5
        for isotropic in (True, False):
            foot2 = afwDetect.growFootprint(foot1, ngrow, isotropic)
            bbox2 = foot2.getBBox()

            if False and display:
                idImage = afwImage.ImageU(foot1.getRegion().getDimensions())
                idImage.set(0)

                i = 1
                for foot in [foot1, foot2]:
                    foot.insertIntoImage(idImage, i)
                    i += 1

                metricImage = afwImage.ImageF("foo.fits")
                ds9.mtv(metricImage, frame=1)
                ds9.mtv(idImage)

            # check bbox1
            self.assertEqual(bbox1.getX0(), x0)
            self.assertEqual(bbox1.getX1(), x0 + width - 1)
            self.assertEqual(bbox1.getWidth(), width)

            self.assertEqual(bbox1.getY0(), y0)
            self.assertEqual(bbox1.getY1(), y0 + height - 1)
            self.assertEqual(bbox1.getHeight(), height)
            # check bbox2
            self.assertEqual(bbox2.getX0(), bbox1.getX0() - ngrow)
            self.assertEqual(bbox2.getX1(), bbox1.getX1() + ngrow)
            self.assertEqual(bbox2.getWidth(), bbox1.getWidth() + 2*ngrow)

            self.assertEqual(bbox2.getY0(), bbox1.getY0() - ngrow)
            self.assertEqual(bbox2.getY1(), bbox1.getY1() + ngrow)
            self.assertEqual(bbox2.getHeight(), bbox1.getHeight() + 2*ngrow)
            # Check that region was preserved
            self.assertEqual(foot1.getRegion(), foot2.getRegion())

    def testFootprintToBBoxList(self):
        """Test footprintToBBoxList"""
        foot = afwDetect.Footprint(0, afwImage.BBox(afwImage.PointI(0, 0), 12, 10))
        for y, x0, x1 in [(3, 3, 5), (3, 7, 7),
                          (4, 2, 3), (4, 5, 7),
                          (5, 2, 3), (5, 5, 8),
                          (6, 3, 5), 
                          ]:
            foot.addSpan(y, x0, x1)

        idImage = afwImage.ImageU(foot.getRegion().getDimensions())
        idImage.set(0)

        foot.insertIntoImage(idImage, 1)
        if display:
            ds9.mtv(idImage)

        idImageFromBBox = idImage.Factory(idImage, True)
        idImageFromBBox.set(0)
        bboxes = afwDetect.footprintToBBoxList(foot)
        for bbox in bboxes:
            x0, y0, x1, y1 = bbox.getX0(), bbox.getY0(), bbox.getX1(), bbox.getY1()

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

        foot = afwDetect.Footprint(0, afwImage.BBox(afwImage.PointI(0, 0), 12, 10))
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
        im = afwImage.ImageU(w, h)
        im.set(0)
        #
        # Create a footprint;  note that these Spans overlap
        #
        for spans in ([(3, 5, 6), (4, 7, 7), ],
                      [(3, 3, 5), (3, 5, 7),
                       (4, 2, 3), (4, 5, 7), (4, 8, 9),
                       (5, 2, 3), (5, 5, 8), (5, 6, 7),
                       (6, 3, 5), 
                       ],
                      ):

            foot = afwDetect.Footprint(0, afwImage.BBox(afwImage.PointI(0, 0), w, h))
            for y, x0, x1 in spans:
                foot.addSpan(y, x0, x1)

                for x in range(x0, x1 + 1): # also insert into im
                    im.set(x, y, 1)

            idImage = afwImage.ImageU(foot.getRegion().getDimensions())
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

            self.assertEqual(afwMath.makeStatistics(idImage, afwMath.MAX).getValue(), 0)

    def testSetFromFootprint(self):
        """Test setting mask/image pixels from a Footprint list"""
        
        mi = afwImage.MaskedImageF(12, 8)
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

        ds = afwDetect.makeFootprintSet(mi, afwDetect.Threshold(15))

        objects = ds.getFootprints()
        afwDetect.setMaskFromFootprintList(mi.getMask(), objects, 0x1)

        self.assertEqual(mi.getMask().get(4, 2), 0x0)
        self.assertEqual(mi.getMask().get(3, 6), 0x1)
        
        self.assertEqual(mi.getImage().get(3, 6), 20)
        afwDetect.setImageFromFootprintList(mi.getImage(), objects, 5.0)
        self.assertEqual(mi.getImage().get(4, 2), 10)
        self.assertEqual(mi.getImage().get(3, 6), 5)
        
        if False and display:
            ds9.mtv(mi, frame=1)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class FootprintSetTestCase(unittest.TestCase):
    """A test case for FootprintSet"""

    def setUp(self):
        self.ms = afwImage.MaskedImageF(12, 8)
        im = self.ms.getImage()
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
            ds9.mtv(im, frame=0)
        
    def tearDown(self):
        del self.ms

    def testGC(self):
        """Check that FootprintSets are automatically garbage collected (when MemoryTestCase runs)"""
        
        ds = afwDetect.FootprintSetF(afwImage.MaskedImageF(10, 20), afwDetect.Threshold(10))

    def testFootprints(self):
        """Check that we found the correct number of objects and that they are correct"""
        ds = afwDetect.FootprintSetF(self.ms, afwDetect.Threshold(10))

        objects = ds.getFootprints()

        self.assertEqual(len(objects), len(self.objects))
        for i in range(len(objects)):
            self.assertEqual(objects[i], self.objects[i])
            
    def testFootprints2(self):
        """Check that we found the correct number of objects using makeFootprintSet"""
        ds = afwDetect.makeFootprintSet(self.ms, afwDetect.Threshold(10))

        objects = ds.getFootprints()

        self.assertEqual(len(objects), len(self.objects))
        for i in range(len(objects)):
            self.assertEqual(objects[i], self.objects[i])
            
    def testFootprintsMasks(self):
        """Check that detectionSets have the proper mask bits set"""
        ds = afwDetect.FootprintSetF(self.ms, afwDetect.Threshold(10), "OBJECT")
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
        ds = afwDetect.FootprintSetF(self.ms, afwDetect.Threshold(10))
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
        ds = afwDetect.FootprintSetF(self.ms, afwDetect.Threshold(10))
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
        ds = afwDetect.FootprintSetF(self.ms.getImage(), afwDetect.Threshold(10))

        objects = ds.getFootprints()

        self.assertEqual(len(objects), len(self.objects))
        for i in range(len(objects)):
            self.assertEqual(objects[i], self.objects[i])
            
    def testGrow2(self):
        """Grow some more interesting shaped Footprints.  Informative with display, but no numerical tests"""
        
        ds = afwDetect.FootprintSetF(self.ms, afwDetect.Threshold(10), "OBJECT")

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

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class NaNFootprintSetTestCase(unittest.TestCase):
    """A test case for FootprintSet when the image contains NaNs"""

    def setUp(self):
        self.ms = afwImage.MaskedImageF(12, 8)
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
        """Check that we found the correct number of objects using makeFootprintSet"""
        ds = afwDetect.makeFootprintSet(self.ms, afwDetect.Threshold(10), "DETECTED")

        objects = ds.getFootprints()

        if display:
            ds9.mtv(self.ms, frame=0)

        self.assertEqual(len(objects), len(self.objects))
        for i in range(len(objects)):
            self.assertEqual(objects[i], self.objects[i])
            
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    tests.init()

    suites = []
    suites += unittest.makeSuite(ThresholdTestCase)
    suites += unittest.makeSuite(FootprintTestCase)
    suites += unittest.makeSuite(FootprintSetTestCase)
    suites += unittest.makeSuite(NaNFootprintSetTestCase)
    suites += unittest.makeSuite(tests.MemoryTestCase)
    return unittest.TestSuite(suites)


def run(shouldExit=False):
    """Run the tests"""
    tests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)
