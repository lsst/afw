#!/usr/bin/env python
"""
Tests for Footprints, and DetectionSets

Run with:
   Footprint_1.py
or
   python
   >>> import Footprint_1; Footprint_1.run()
"""

import pdb                              # we may want to say pdb.set_trace()
import unittest
import lsst.utils.tests as tests
import lsst.pex.policy as pexPolicy
import lsst.pex.logging as logging
import lsst.afw.image.imageLib as afwImage
import lsst.afw.detection.detectionLib as afwDetection
import lsst.afw.display.ds9 as ds9

try:
    type(verbose)
except NameError:
    verbose = 0
    logging.Trace_setVerbosity("afwDetection.Footprint", verbose)

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

class ThresholdTestCase(unittest.TestCase):
    def testPolicyConstructor(self):
        """
        Test the creation of a Threshold object from a Policy Object

        This is a white-box test.
        -tests missing policy parameters
        -tests mal-formed parameters

        Cannot test polarity settings due to bug in lsst.pex.policy.Policy.
        lsst.pex.policy.Policy 
        """
        policy = pexPolicy.Policy()

        try:
            threshold = afwDetection.Threshold(policy)
        except:
            pass
        else:
            self.fail("Threhold policy not properly validated")

        policy.add("value", 3.4)
        try:
            threhold = afwDetection.Threshold(policy)
        except:
            self.fail("Threshold failed to build with proper policy")

        
        policy.add("type", "foo bar")
        try:
            threshold = afwDetection.Threshold(policy)
        except:
            pass
        else:
            self.fail("Threhold policy not properly validated")

        policy.set("type", "stdev")
        try:
            threhold = afwDetection.Threshold(policy)
        except:
            self.fail("Threshold failed to build with proper policy")

        policy.set("type", "value")
        try:
            threhold = afwDetection.Threshold(policy)
        except:
            self.fail("Threshold failed to build with proper policy")

        policy.set("type", "variance")
        try:
            threhold = afwDetection.Threshold(policy)
        except:
            self.fail("Threshold failed to build with proper policy")

class FootprintTestCase(unittest.TestCase):
    """A test case for Footprint"""
    def setUp(self):
        self.foot = afwDetection.Footprint()

    def tearDown(self):
        del self.foot

    def testToString(self):
        y, x0, x1 = 10, 100, 101
        s = afwDetection.Span(y, x0, x1)
        self.assertEqual(s.toString(), toString(y, x0, x1))

    def testBbox(self):
        """Test setBBox"""
        
        self.assertEqual(self.foot.setBBox(), None)

    def testGC(self):
        """Check that Footprints are automatically garbage collected (when MemoryTestCase runs)"""
        
        f = afwDetection.Footprint()

    def testId(self):
        """Test uniqueness of IDs"""
        
        self.assertNotEqual(self.foot.getId(), afwDetection.Footprint().getId())

    def testAddSpans(self):
        """Add spans to a Footprint"""
        for y, x0, x1 in [(10, 100, 105), (11, 99, 104)]:
            self.foot.addSpan(y, x0, x1)

        sp = self.foot.getSpans()
        
        self.assertEqual(sp[-1].toString(), toString(y, x0, x1))

    def testBbox(self):
        """Add Spans and check bounding box"""
        foot = afwDetection.Footprint()
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

        span = afwDetection.Span(10, 100, 105)
        foot = afwDetection.Footprint()

        foot.addSpan(span, 1, 2)

        bbox = foot.getBBox()
        self.assertEqual(bbox.getWidth(), 6)
        self.assertEqual(bbox.getHeight(), 1)
        self.assertEqual(bbox.getX0(), 101)
        self.assertEqual(bbox.getY0(), 12)
        #
        # Shift that span using Span.shift
        #
        foot = afwDetection.Footprint()
        span.shift(-1, -2)
        foot.addSpan(span)

        bbox = foot.getBBox()
        self.assertEqual(bbox.getWidth(), 6)
        self.assertEqual(bbox.getHeight(), 1)
        self.assertEqual(bbox.getX0(), 99)
        self.assertEqual(bbox.getY0(), 8)

    def testFootprintFromBBox(self):
        """Create a rectangular Footprint"""
        foot = afwDetection.Footprint(afwImage.BBox(afwImage.PointI(9, 10), 7, 4),
                                      afwImage.BBox(afwImage.PointI(0, 0), 30, 20))

        bbox = foot.getBBox()

        self.assertEqual(bbox.getWidth(), 7)
        self.assertEqual(bbox.getHeight(), 4)
        self.assertEqual(bbox.getX0(), 9)
        self.assertEqual(bbox.getY0(), 10)
        self.assertEqual(bbox.getX1(), 15)
        self.assertEqual(bbox.getY1(), 13)

        idImage = afwImage.ImageU(foot.getRegion().getDimensions())
        idImage.set(0)
        
        foot.insertIntoImage(idImage, foot.getId())

        if False:
            ds9.mtv(idImage, frame=2)

    def testBCircle2i(self):
        """Test the BCircle2i constructors"""
        
        x = 100; y = 200; r = 1.5
        
        bc = afwImage.BCircle(afwImage.PointI(x, y), r)
        for i in range(2):
            c = bc.getCenter()
            self.assertEqual(c.getX(), x)
            self.assertEqual(c.getY(), y)
            self.assertAlmostEqual(bc.getRadius(), r)

            bc = afwImage.BCircle(afwImage.PointI(x, y), r)

    def testFootprintFromBCircle(self):
        """Create a circular Footprint"""

        foot = afwDetection.Footprint(afwImage.BCircle(afwImage.PointI(9, 15), 6),
                                      afwImage.BBox(afwImage.PointI(0, 0), 20, 30))

        idImage = afwImage.ImageU(foot.getRegion().getWidth(), foot.getRegion().getHeight())
        idImage.set(0)
        
        foot.insertIntoImage(idImage, foot.getId())

        if False:
            ds9.mtv(idImage, frame=2)

    def testGrow(self):
        """Test growing a footprint"""
        x0, y0 = 20, 20;  width, height = 20, 30
        foot1 = afwDetection.Footprint(afwImage.BBox(afwImage.PointI(x0, y0), width, height),
                                       afwImage.BBox(afwImage.PointI(0, 0), 100, 100))
        bbox1 = foot1.getBBox()

        self.assertEqual(bbox1.getX0(), x0)
        self.assertEqual(bbox1.getX1(), x0 + width - 1)
        self.assertEqual(bbox1.getWidth(), width)

        self.assertEqual(bbox1.getY0(), y0)
        self.assertEqual(bbox1.getY1(), y0 + height - 1)
        self.assertEqual(bbox1.getHeight(), height)

        ngrow = 1
        foot2 = afwDetection.growFootprint(foot1, ngrow)
        bbox2 = foot2.getBBox()

        # check bbox1
        self.assertEqual(bbox1.getX0(), x0)
        self.assertEqual(bbox1.getX1(), x0 + width - ngrow)
        self.assertEqual(bbox1.getWidth(), width)

        self.assertEqual(bbox1.getY0(), y0)
        self.assertEqual(bbox1.getY1(), y0 + height - ngrow)
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

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class DetectionSetTestCase(unittest.TestCase):
    """A test case for DetectionSet"""
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
    
    def setUp(self):
        self.ms = afwImage.MaskedImageF(12, 8)
        im = self.ms.getImage()
        #
        # Objects that we should detect
        #
        self.objects = []
        self.objects += [self.Object(10, [(1, 4, 4), (2, 3, 5), (3, 4, 4)])]
        self.objects += [self.Object(20, [(5, 7, 8), (5, 10, 10), (6, 8, 9)])]
        self.objects += [self.Object(20, [(6, 3, 3)])]

        im.set(0)                       # clear image
        for obj in self.objects:
            obj.insert(im)

        if display:
            ds9.mtv(im, frame=0)
        
    def tearDown(self):
        del self.ms

    def testGC(self):
        """Check that DetectionSets are automatically garbage collected (when MemoryTestCase runs)"""
        
        ds = afwDetection.DetectionSetF(afwImage.MaskedImageF(10, 20), afwDetection.Threshold(10))

    def testFootprints(self):
        """Check that we found the correct number of objects and that they are correct"""
        ds = afwDetection.DetectionSetF(self.ms, afwDetection.Threshold(10))

        objects = ds.getFootprints()

        self.assertEqual(len(objects), len(self.objects))
        for i in range(len(objects)):
            self.assertEqual(objects[i], self.objects[i])
            
    def testFootprintsMasks(self):
        """Check that detectionSets have the proper mask bits set"""
        ds = afwDetection.DetectionSetF(self.ms, afwDetection.Threshold(10), "OBJECT")
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
        ds = afwDetection.DetectionSetF(self.ms, afwDetection.Threshold(10))
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


    def testDetectionSetImageId(self):
        """Check that we can insert a DetectionSet into an Image, setting relative IDs"""
        ds = afwDetection.DetectionSetF(self.ms, afwDetection.Threshold(10))
        objects = ds.getFootprints()

        idImage = ds.insertIntoImage(True)
        if display:
            ds9.mtv(idImage, frame=2)

        for i in range(len(objects)):
            for sp in objects[i].getSpans():
                for x in range(sp.getX0(), sp.getX1() + 1):
                    self.assertEqual(idImage.get(x, sp.getY()), i + 1)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    tests.init()

    suites = []
    suites += unittest.makeSuite(ThresholdTestCase)
    suites += unittest.makeSuite(FootprintTestCase)
    suites += unittest.makeSuite(DetectionSetTestCase)
    suites += unittest.makeSuite(tests.MemoryTestCase)
    return unittest.TestSuite(suites)


def run(exit=False):
    """Run the tests"""
    tests.run(suite(), exit)

if __name__ == "__main__":
    run(True)
