import os
import math
import pdb                          # we may want to say pdb.set_trace()
import unittest

import lsst.afw.image as afwImage
import lsst.utils.tests as utilsTests
import lsst.daf.base as dafBase
import afwTests
import lsst.pex.exceptions as pexEx

import lsst.afw.display.ds9 as ds9
try:
    type(display)
except NameError:
    display = False

try:
    type(verbose)
except NameError:
    verbose = 0

class MaskTestCase(unittest.TestCase):
    """A test case for Mask (based on Mask_1.cc)"""

    def setUp(self):
        maskImage = afwImage.ImageViewU(300,400)

        self.testMask = afwImage.MaskU(afwImage.MaskIVwPtrT(maskImage))
        self.testMask.clearMaskPlaneDict() # reset so tests will be deterministic

        for p in ("CR", "BP"):
            self.testMask.addMaskPlane(p)

        self.region = afwImage.BBox2i(100, 300, 10, 40)
        self.subTestMask = self.testMask.getSubMask(self.region)

        self.pixelList = afw.listPixelCoord()
        for x in range(0, 300):
            for y in range(300, 400, 20):
                self.pixelList.push_back(afw.PixelCoord(x, y))

    def tearDown(self):
        del self.subTestMask
        del self.testMask
        del self.region

    def testPlaneAddition(self):
        """Test mask plane addition"""

        nplane = self.testMask.getNumPlanesUsed()
        for p in ("XCR", "XBP"):
            self.assertEqual(self.testMask.addMaskPlane(p), nplane, "Assigning plane %s" % (p))
            nplane += 1

        for p in range(0,8):
            sp = "P%d" % p
            plane = self.testMask.addMaskPlane(sp)
            print "Assigned %s to plane %d" % (sp, plane)

        for p in range(0,8):
            sp = "P%d" % p
            self.testMask.removeMaskPlane(sp)

        self.assertEqual(nplane, self.testMask.getNumPlanesUsed(), "Adding and removing planes")

    def testMetaData(self):
        """Test mask plane metaData"""

        metaData = dafBase.DataProperty_createPropertyNode("testMetaData")

        self.testMask.addMaskPlaneMetaData(metaData)
        print "MaskPlane metadata:"
        print metaData.toString("\t");

        print "Printing metadata from Python:"
        d = self.testMask.getMaskPlaneDict()
        for p in d.keys():
            if d[p]:
                print "\t", d[p], p

        newPlane = dafBase.DataProperty("Whatever", 5)
        metaData.addProperty(newPlane)

        self.testMask.parseMaskPlaneMetaData(metaData)
        print "After loading metadata: "
        self.testMask.printMaskPlanes()

    def testPlaneOperations(self):
        """Test mask plane operations"""

        planes = lookupPlanes(self.testMask, ["CR", "BP"])
        self.testMask.clearMaskPlane(planes['CR'])

        for p in planes.keys():
            self.testMask.setMaskPlaneValues(planes[p], self.pixelList)

        printMaskPlane(self.testMask, planes['CR'])

        print "\nClearing mask"
        self.testMask.clearMaskPlane(planes['CR'])

        printMaskPlane(self.testMask, planes['CR'])

    def testOrEquals(self):
        """Test |= operator"""

        testMask3 = afwImage.MaskU(
            afwImage.MaskIVwPtrT(afwImage.ImageViewU(self.testMask.getCols(), self.testMask.getRows()))
            )

        testMask3.addMaskPlane("CR")

        self.testMask |= testMask3

        print "Applied |= operator"

    def testPlaneRemoval(self):
        """Test mask plane removal"""

        planes = lookupPlanes(self.testMask, ["CR", "BP"])
        self.testMask.clearMaskPlane(planes['BP'])
        self.testMask.removeMaskPlane("BP")

        self.assertEqual(self.testMask.getMaskPlane("BP"), -1, "Plane BP is removed")

    def testInvalidPlaneOperations(self):
        """Test mask plane operations invalidated by Mask changes"""

        testMask3 = afwImage.MaskU(self.testMask.getCols(), self.testMask.getRows())
        
        name = "Great Timothy"
        testMask3.addMaskPlane(name)
        testMask3.removeMaskPlane(name) # invalidates dictionary version

        def tst():
            self.testMask |= testMask3

        self.assertRaises(pexEx.LsstRuntime, tst)
        self.testMask.this.acquire()    # Work around bug in swig "mask |= mask;" leaks when |= throws

    def testInvalidPlaneOperations2(self):
        """Test mask plane operations invalidated by Mask changes"""

        testMask3 = afwImage.MaskU(self.testMask.getCols(), self.testMask.getRows())
        
        name = "Great Timothy"
        name2 = "Our Boss"
        testMask3.addMaskPlane(name)
        testMask3.addMaskPlane(name2)
        oldDict = testMask3.getMaskPlaneDict()

        self.testMask.removeMaskPlane(name)
        self.testMask.removeMaskPlane(name2)
        self.testMask.addMaskPlane(name2) # added in opposite order to testMask3
        self.testMask.addMaskPlane(name)

        self.assertNotEqual(self.testMask.getMaskPlaneDict()[name], oldDict[name])

        def tst():
            self.testMask |= testMask3

        self.assertRaises(pexEx.LsstRuntime, tst)
        self.testMask.this.acquire()    # Work around bug in swig "mask |= mask;" leaks when |= throws

    def XXtestConformMaskPlanes(self):
        """Test conformMaskPlanes() when the two planes are actually the same"""

        testMask3 = afwImage.MaskU(self.testMask.getCols(), self.testMask.getRows())
        oldDict = self.testMask.getMaskPlaneDict()

        name = "XXX"
        self.testMask.addMaskPlane(name)
        self.testMask.removeMaskPlane(name) # invalidates dictionary version

        testMask3.conformMaskPlanes(oldDict)

        self.testMask |= testMask3

    def testConformMaskPlanes2(self):
        """Test conformMaskPlanes() when the two planes are different"""

        testMask3 = afwImage.MaskU(self.testMask.getCols(), self.testMask.getRows())
        
        name1 = "Great Timothy"
        name2 = "Our Boss"
        p1 = testMask3.addMaskPlane(name1)
        p2 = testMask3.addMaskPlane(name2)
        oldDict = self.testMask.getMaskPlaneDict()

        testMask3.setMaskPlaneValues(p1, 0, 5, 0)
        testMask3.setMaskPlaneValues(p2, 0, 5, 1)

        if display:
            im = afwImage.ImageD(self.testMask.getCols(), self.testMask.getRows())
            ds9.mtv(im)                 # bug in Mask display; needs an Image first
            ds9.mtv(testMask3)

        self.assertEqual(testMask3.getVal(0,0), testMask3.getPlaneBitMask(name1))
        self.assertEqual(testMask3.getVal(0,1), testMask3.getPlaneBitMask(name2))

        self.testMask.removeMaskPlane(name1)
        self.testMask.removeMaskPlane(name2)
        self.testMask.addMaskPlane(name2) # added in opposite order to testMask3
        self.testMask.addMaskPlane(name1)

        self.assertEqual(self.testMask.getVal(0,0), 0)

        if display:
            ds9.mtv(im, frame=1)
            ds9.mtv(testMask3, frame=1)

        testMask3.conformMaskPlanes(oldDict)

        self.assertEqual(testMask3.getVal(0,0), testMask3.getPlaneBitMask(name1))
        self.assertEqual(testMask3.getVal(0,1), testMask3.getPlaneBitMask(name2))

        if display:
            ds9.mtv(im, frame=2)
            ds9.mtv(testMask3, frame=2)

        self.testMask |= testMask3

    def testSubmask(self):
        """Test submask methods"""

        planes = lookupPlanes(self.testMask, ["CR", "BP"])
        self.testMask.setMaskPlaneValues(planes['CR'], self.pixelList)

        self.testMask.clearMaskPlane(planes['CR'])

        self.testMask.replaceSubMask(self.region, self.subTestMask)

        printMaskPlane(self.testMask, planes['CR'], range(90, 120), range(295, 350, 5))

    def testMaskPixelBooleanFunc(self):
        """Test MaskPixelBooleanFunc"""
        testCrFuncInstance = fwTests.testCrFuncD(self.testMask)
        testCrFuncInstance.init() # get the latest plane info from testMask
        CR_plane = self.testMask.getMaskPlane("CR")
        self.assertNotEqual(CR_plane, -1)
        
        self.testMask.setMaskPlaneValues(CR_plane, self.pixelList)
        count = self.testMask.countMask(testCrFuncInstance, self.region)
        self.assertEqual(count, 20, "Saw %d pixels with CR set" % count)

        del testCrFuncInstance

        # should generate a vw exception - dims. of region and submask must be =
        self.region.expand(10)
        self.assertRaises(Exception, self.testMask.replaceSubMask, self.region, self.subTestMask)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

def lookupPlanes(mask, planeNames):
    planes = {}
    for p in planeNames:
        try:
            planes[p] = mask.getMaskPlane(p)
            print "%s plane is %d" % (p, planes[p])
        except Exception, e:
            print "No %s plane found: %s" % (p, e)

    return planes

def printMaskPlane(mask, plane,
                   xrange=range(250, 300, 10), yrange=range(300, 400, 20)):
    """Print parts of the specified plane of the mask"""
    
    if True:
        xrange = range(min(xrange), max(xrange), 25)
        yrange = range(min(yrange), max(yrange), 25)

    for x in xrange:
        for y in yrange:
            if False:                   # mask(x,y) confuses swig
                print x, y, mask(x, y), mask(x, y, plane)
            else:
                print x, y, mask(x, y, plane)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(MaskTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)

    return unittest.TestSuite(suites)

def run(exit=False):
    """Run the tests"""
    utilsTests.run(suite(), exit)

if __name__ == "__main__":
    run(True)
