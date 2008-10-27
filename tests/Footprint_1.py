"""
Tests for Footprints, and DetectionSets

Run with:
   python Footprint_1.py
or
   python
   >>> import unittest; T=load("Footprint_1"); unittest.TextTestRunner(verbosity=1).run(T.suite())
"""

import pdb                              # we may want to say pdb.set_trace()
import unittest
import lsst.utils.tests as tests
import lsst.pex.logging as logging
import lsst.afw.image as imageLib
import lsst.afw.display.ds9 as ds9
import lsst.detection.detectionLib as detection

try:
    type(verbose)
except NameError:
    verbose = 0
    logging.Trace_setVerbosity("detection.Footprint", verbose)

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

class FootprintTestCase(unittest.TestCase):
    """A test case for Footprint"""
    def setUp(self):
        self.foot = detection.Footprint()

    def tearDown(self):
        del self.foot

    def testToString(self):
        y, x0, x1 = 10, 100, 101
        s = detection.Span(y, x0, x1)
        self.assertEqual(s.toString(), toString(y, x0, x1))

    def testBbox(self):
        """Test setBBox"""
        
        self.assertEqual(self.foot.setBBox(), None)

    def testGC(self):
        """Check that Footprints are automatically garbage collected (when MemoryTestCase runs)"""
        
        f = detection.Footprint()

    def testId(self):
        """Test uniqueness of IDs"""
        
        self.assertNotEqual(self.foot.getId(), detection.Footprint().getId())

    def testAddSpans(self):
        """Add spans to a Footprint"""
        for y, x0, x1 in [(10, 100, 105), (11, 99, 104)]:
            self.foot.addSpan(y, x0, x1)

        sp = self.foot.getSpans()
        
        self.assertEqual(sp[-1].toString(), toString(y, x0, x1))

    def testBbox(self):
        """Add Spans and check bounding box"""
        foot = detection.Footprint()
        for y, x0, x1 in [(10, 100, 105),
                          (11, 99, 104)]:
            foot.addSpan(y, x0, x1)

        bbox = foot.getBBox()
        self.assertEqual(bbox.width(), 7)
        self.assertEqual(bbox.height(), 2)
        self.assertEqual(bbox.min().x(), 99)
        self.assertEqual(bbox.min().y(), 10)
        self.assertEqual(bbox.max().x(), 106) # i.e. one past
        self.assertEqual(bbox.max().y(), 12) #                the last point

    def testFootprintFromBBox(self):
        """Create a rectangular Footprint"""
        foot = detection.Footprint(imageLib.BBox2i(9, 10, 7, 4),
                                   imageLib.BBox2i(0, 0, 30, 20))

        bbox = foot.getBBox()

        self.assertEqual(bbox.width(), 7)
        self.assertEqual(bbox.height(), 4)
        self.assertEqual(bbox.min().x(), 9)
        self.assertEqual(bbox.min().y(), 10)
        self.assertEqual(bbox.max().x(), 16) # i.e. one past
        self.assertEqual(bbox.max().y(), 14) #              the last point

        idImage = imageLib.ImageU(foot.getRegion().width(), foot.getRegion().height())
        idImage.set(0)
        
        foot.insertIntoImage(idImage, foot.getId())

        if False:
            ds9.mtv(idImage, frame=2)

    def testBCircle2i(self):
        """Test the BCircle2i constructors"""
        
        x = 100; y = 200; r = 1.5
        
        bc = detection.BCircle2i(detection.Vector2i(x, y), r)
        for i in range(2):
            c = bc.center()
            self.assertEqual(c.x(), x)
            self.assertEqual(c.y(), y)
            self.assertAlmostEqual(bc.radius(), r)

            bc = detection.BCircle2i(x, y, r)

    def testFootprintFromBCircle(self):
        """Create a circular Footprint"""

        foot = detection.Footprint(detection.BCircle2i(9, 15, 6),
                                   imageLib.BBox2i(0, 0, 20, 30))

        idImage = imageLib.ImageU(foot.getRegion().width(), foot.getRegion().height())
        idImage.set(0)
        
        foot.insertIntoImage(idImage, foot.getId())

        if False:
            ds9.mtv(idImage, frame=2)

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
        self.ms = imageLib.MaskedImageD(12, 8)
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
        
        ms = imageLib.MaskedImageD(10, 20)
        ds = detection.DetectionSetD(ms, detection.Threshold(10))

    def testFootprints(self):
        """Check that we found the correct number of objects and that they are correct"""
        ds = detection.DetectionSetD(self.ms, detection.Threshold(10))
        objects = ds.getFootprints()

        self.assertEqual(len(objects), len(self.objects))
        for i in range(len(objects)):
            self.assertEqual(objects[i], self.objects[i])
            
    def testFootprintsMasks(self):
        """Check that detectionSets have the proper mask bits set"""
        ds = detection.DetectionSetD(self.ms, detection.Threshold(10), "OBJECT")
        objects = ds.getFootprints()

        if display:
            ds9.mtv(self.ms, frame=1)

        mask = self.ms.getMask()
        for i in range(len(objects)):
            for sp in objects[i].getSpans():
                for x in range(sp.getX0(), sp.getX1() + 1):
                    self.assertEqual(mask.getVal(x, sp.getY()), mask.getPlaneBitMask("OBJECT"))

    def testFootprintsImageId(self):
        """Check that we can insert footprints into an Image"""
        ds = detection.DetectionSetD(self.ms, detection.Threshold(10))
        objects = ds.getFootprints()

        idImage = imageLib.ImageU(self.ms.getImage().getCols(), self.ms.getImage().getRows())
        idImage.set(0)
        
        for foot in objects:
            foot.insertIntoImage(idImage, foot.getId())

        if False:
            ds9.mtv(idImage, frame=2)

        for i in range(len(objects)):
            for sp in objects[i].getSpans():
                for x in range(sp.getX0(), sp.getX1() + 1):
                    self.assertEqual(idImage.getVal(x, sp.getY()), objects[i].getId())


    def testDetectionSetImageId(self):
        """Check that we can insert a DetectionSet into an Image, setting relative IDs"""
        ds = detection.DetectionSetD(self.ms, detection.Threshold(10))
        objects = ds.getFootprints()

        idImage = ds.insertIntoImage(True)
        if display:
            ds9.mtv(idImage, frame=2)

        for i in range(len(objects)):
            for sp in objects[i].getSpans():
                for x in range(sp.getX0(), sp.getX1() + 1):
                    self.assertEqual(idImage.getVal(x, sp.getY()), i + 1)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    tests.init()

    suites = []
    suites += unittest.makeSuite(FootprintTestCase)
    suites += unittest.makeSuite(DetectionSetTestCase)
    suites += unittest.makeSuite(tests.MemoryTestCase)
    return unittest.TestSuite(suites)


def run(exit=False):
    """Run the tests"""
    tests.run(suite(), exit)

if __name__ == "__main__":
    run(True)
