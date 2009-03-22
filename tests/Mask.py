#!/usr/bin/env python
"""
Tests for Masks

Run with:
   python Mask.py
or
   python
   >>> import Mask; Mask.run()
"""

import os
import pdb  # we may want to say pdb.set_trace()
import sys
import unittest

import lsst.utils.tests as utilsTests
import lsst.pex.exceptions as pexExcept
import lsst.daf.base
import lsst.afw.image.imageLib as afwImage
import eups
import lsst.afw.display.ds9 as ds9

try:
    type(display)
except NameError:
    display = False

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class MaskTestCase(unittest.TestCase):
    """A test case for Mask"""

    def setUp(self):
        tmp = afwImage.MaskU()           # clearMaskPlaneDict isn't static
        tmp.clearMaskPlaneDict()        # reset so tests will be deterministic

        for p in ("BAD", "SAT", "INTRP", "CR", "EDGE"):
            afwImage.MaskU_addMaskPlane(p)

        self.BAD  = afwImage.MaskU_getPlaneBitMask("BAD")
        self.CR   = afwImage.MaskU_getPlaneBitMask("CR")
        self.EDGE = afwImage.MaskU_getPlaneBitMask("EDGE")

        self.val1 = self.BAD | self.CR
        self.val2 = self.val1 | self.EDGE

        self.mask1 = afwImage.MaskU(100, 200); self.mask1.set(self.val1)
        self.mask2 = afwImage.MaskU(self.mask1.getDimensions()); self.mask2.set(self.val2)

        dataDir = eups.productDir("afwdata")
        if dataDir:
            if True:
                self.maskFile = os.path.join(dataDir, "small_MI_msk.fits")
            else:
                self.maskFile = os.path.join(dataDir, "871034p_1_MI_msk.fits")
        else:
            self.maskFile = None

    def tearDown(self):
        del self.mask1
        del self.mask2

    def testInitializeMasks(self):
        val = 0x1234
        msk = afwImage.MaskU(10, 10, val)
        self.assertEqual(msk.get(0,0), val)

        msk2 = afwImage.MaskU(afwImage.pairIntInt(10, 10), val)
        self.assertEqual(msk2.get(0,0), val)
        
    def testSetGetMasks(self):
        self.assertEqual(self.mask1.get(0,0), self.val1)
    
    def testOrMasks(self):
        self.mask2 |= self.mask1
        self.mask1 |= self.val2
        
        self.assertEqual(self.mask1.get(0,0), self.val1 | self.val2)
        self.assertEqual(self.mask2.get(0,0), self.val1 | self.val2)
    
    def testAndMasks(self):
        self.mask2 &= self.mask1
        self.mask1 &= self.val2
        
        self.assertEqual(self.mask1.get(0,0), self.val1 & self.val2)
        self.assertEqual(self.mask1.get(0,0), self.BAD | self.CR)
        self.assertEqual(self.mask2.get(0,0), self.val1 & self.val2)

    def testXorMasks(self):
        self.mask2 ^= self.mask1
        self.mask1 ^= self.val2
        
        self.assertEqual(self.mask1.get(0,0), self.val1 ^ self.val2)
        self.assertEqual(self.mask2.get(0,0), self.val1 ^ self.val2)

    def testLogicalMasksMismatch(self):
        "Test logical operations on Masks of different sizes"
        i1 = afwImage.MaskU(100,100); i1.set(100)
        i2 = afwImage.MaskU(10,10);   i2.set(10)
        
        def tst(i1, i2): i1 |= i2
        utilsTests.assertRaisesLsstCpp(self, lsst.pex.exceptions.LengthErrorException, tst, i1, i2)

        def tst(i1, i2): i1 &= i2
        utilsTests.assertRaisesLsstCpp(self, lsst.pex.exceptions.LengthErrorException, tst, i1, i2)
    
    def testMaskPlanes(self):
        planes = afwImage.MaskU_getMaskPlaneDict()
        self.assertEqual(len(planes), afwImage.MaskU_getNumPlanesUsed())
        
        for k in sorted(planes.keys()):
            self.assertEqual(planes[k], afwImage.MaskU_getMaskPlane(k))
            
    def testCopyConstructors(self):
        dmask = afwImage.MaskU(self.mask1, True) # deep copy
        smask = afwImage.MaskU(self.mask1) # shallow copy
        
        self.mask1 |= 32767             # should only change dmask
        self.assertEqual(dmask.get(0,0), self.val1)
        self.assertEqual(smask.get(0,0), self.val1 | 32767)

    def testBBox(self):
        x0, y0, width, height = 1, 2, 10, 20
        x1, y1 = x0 + width - 1, y0 + height - 1
        llc = afwImage.PointI(x0, y0)
        
        bbox = afwImage.BBox(llc, width, height)

        self.assertEqual(bbox.getX0(), x0)
        self.assertEqual(bbox.getY0(), y0)
        self.assertEqual(bbox.getX1(), x1)
        self.assertEqual(bbox.getY1(), y1)
        self.assertEqual(bbox.getWidth(), width)
        self.assertEqual(bbox.getHeight(), height)

        urc = afwImage.PointI(x1, y1)
        bbox2 = afwImage.BBox(llc, urc)
        self.assertEqual(bbox, bbox2)
        
        bbox2 = afwImage.BBox(llc, width, height+1)
        self.assertNotEqual(bbox, bbox2)

    def testSubmasks(self):
        smask = afwImage.MaskU(self.mask1, afwImage.BBox(afwImage.PointI(1, 1), 3, 2))
        mask2 = afwImage.MaskU(smask.getDimensions())

        mask2.set(666)
        smask <<= mask2
        
        del smask; del mask2
        
        self.assertEqual(self.mask1.get(0, 0), self.val1)
        self.assertEqual(self.mask1.get(1, 1), 666)
        self.assertEqual(self.mask1.get(4, 1), self.val1)
        self.assertEqual(self.mask1.get(1, 2), 666)
        self.assertEqual(self.mask1.get(4, 2), self.val1)
        self.assertEqual(self.mask1.get(1, 3), self.val1)

    def testReadFits(self):
        if not self.maskFile:
            print >> sys.stderr, "Warning: afwdata is not set up; not running the FITS I/O tests"
            return

        nMaskPlanes0 = afwImage.MaskU_getNumPlanesUsed()
        mask = afwImage.MaskU(self.maskFile) # will take any unrecognised mask planes and shift them into unused slots

        if False:
            for (k, v) in afwImage.MaskU_getMaskPlaneDict().items():
                print k, v

        self.assertEqual(mask.get(32,1), 0)
        self.assertEqual(mask.get(50,50), 0)
        self.assertEqual(mask.get(0,0), (1<<nMaskPlanes0))

    def testReadFitsConform(self):
        if not self.maskFile:
            print >> sys.stderr, "Warning: afwdata is not set up; not running the FITS I/O tests"
            return

        hdu = 0
        mask = afwImage.MaskU(self.maskFile, hdu, None, afwImage.BBox(), True)

        if False:
            import lsst.afw.display.ds9 as ds9
            ds9.mtv(mask)

        if False:
            for (k, v) in afwImage.MaskU_getMaskPlaneDict().items():
                print k, v

        self.assertEqual(mask.get(32,1), 0)
        self.assertEqual(mask.get(50,50), 0)
        self.assertEqual(mask.get(0,0), 1)

    def testWriteFits(self):
        if not self.maskFile:
            print >> sys.stderr, "Warning: afwdata is not set up; not running the FITS I/O tests"
            return

        nMaskPlanes0 = afwImage.MaskU_getNumPlanesUsed()
        mask = afwImage.MaskU(self.maskFile)

        self.assertEqual(mask.get(32,1), 0)
        self.assertEqual(mask.get(50,50), 0)
        self.assertEqual(mask.get(0,0), (1<<nMaskPlanes0)) # as header had none of the canonical planes

        tmpFile = "foo.fits"
        mask.writeFits(tmpFile)
        #
        # Read it back
        #
        rmask = afwImage.MaskU(tmpFile)
        self.assertEqual(mask.get(0,0), rmask.get(0,0))
        #
        # Check that we wrote (and read) the metadata successfully
        #
        for (k, v) in afwImage.MaskU_getMaskPlaneDict().items():
            #self.assertEqual(rmask.getMetadata().findUnique(k, True).getValueInt(), v)
            pass

        if False:
            print rmask.getMetadata().toString("", True)
            rmask.printMaskPlanes()

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class OldMaskTestCase(unittest.TestCase):
    """A test case for Mask (based on MaskU_1.cc); these are taken over from the DC2 fw tests
    and modified to run with the new (DC3) APIs"""

    def setUp(self):
        self.testMask = afwImage.MaskU(300,400,0)
        #self.testMask.set(0)

        self.testMask.clearMaskPlaneDict() # reset so tests will be deterministic

        for p in ("CR", "BP"):
            self.testMask.addMaskPlane(p)

        self.region = afwImage.BBox(afwImage.PointI(100, 300), 10, 40)
        self.subTestMask = afwImage.MaskU(self.testMask, self.region)

        if False:
            self.pixelList = afwImage.listPixelCoord()
            for x in range(0, 300):
                for y in range(300, 400, 20):
                    self.pixelList.push_back(afwImage.PixelCoord(x, y))

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
            #print "Assigned %s to plane %d" % (sp, plane)

        for p in range(0,8):
            sp = "P%d" % p
            self.testMask.removeMaskPlane(sp)

        self.assertEqual(nplane, self.testMask.getNumPlanesUsed(), "Adding and removing planes")

    def testMetadata(self):
        """Test mask plane metadata interchange with MaskPlaneDict"""
        #
        # Demonstrate that we can extract a MaskPlaneDict into metadata
        #
        metadata = lsst.daf.base.PropertySet()

        afwImage.MaskU_addMaskPlanesToMetadata(metadata)
        for (k, v) in afwImage.MaskU_getMaskPlaneDict().items():
            self.assertEqual(metadata.getInt("MP_%s" % k), v)
        #
        # Now add another plane to metadata and make it appear in the mask Dict, albeit
        # in general at another location (hence the getNumPlanesUsed call)
        #
        metadata.addInt("MP_" + "Whatever", afwImage.MaskU_getNumPlanesUsed())

        self.testMask.conformMaskPlanes(afwImage.MaskU_parseMaskPlaneMetadata(metadata))
        for (k, v) in afwImage.MaskU_getMaskPlaneDict().items():
            self.assertEqual(metadata.getInt("MP_%s" % k), v)

    def testPlaneOperations(self):
        """Test mask plane operations"""

        planes = afwImage.MaskU_getMaskPlaneDict()
        self.testMask.clearMaskPlane(planes['CR'])

        if False:
            for p in planes.keys():
                self.testMask.setMaskPlaneValues(planes[p], self.pixelList)

        #printMaskPlane(self.testMask, planes['CR'])

        #print "\nClearing mask"
        self.testMask.clearMaskPlane(planes['CR'])

        #printMaskPlane(self.testMask, planes['CR'])

    def testPlaneRemoval(self):
        """Test mask plane removal"""

        planes = afwImage.MaskU_getMaskPlaneDict()
        self.testMask.clearMaskPlane(planes['BP'])
        self.testMask.removeMaskPlane("BP")

        def checkPlaneBP():
            self.testMask.getMaskPlane("BP")

        utilsTests.assertRaisesLsstCpp(self, pexExcept.InvalidParameterException, checkPlaneBP)

    def testInvalidPlaneOperations(self):
        """Test mask plane operations invalidated by Mask changes"""

        testMask3 = afwImage.MaskU(self.testMask.getDimensions())
        
        name = "Great Timothy"
        testMask3.addMaskPlane(name)
        testMask3.removeMaskPlane(name) # invalidates dictionary version

        def tst():
            self.testMask |= testMask3

        utilsTests.assertRaisesLsstCpp(self, pexExcept.RuntimeErrorException, tst)

    def testInvalidPlaneOperations2(self):
        """Test mask plane operations invalidated by Mask changes"""

        testMask3 = afwImage.MaskU(self.testMask.getDimensions())
        
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

        utilsTests.assertRaisesLsstCpp(self, pexExcept.RuntimeErrorException, tst)
        #
        # OK, that failed as it should.  Fixup the dictionaries and try again
        #
        testMask3.conformMaskPlanes(oldDict)

        self.testMask |= testMask3      # shouldn't throw

    def testConformMaskPlanes(self):
        """Test conformMaskPlanes() when the two planes are actually the same"""

        testMask3 = afwImage.MaskU(self.testMask.getDimensions())
        oldDict = self.testMask.getMaskPlaneDict()

        name = "XXX"
        self.testMask.addMaskPlane(name)
        self.testMask.removeMaskPlane(name) # invalidates dictionary version

        testMask3.conformMaskPlanes(oldDict)

        self.testMask |= testMask3

    def testConformMaskPlanes2(self):
        """Test conformMaskPlanes() when the two planes are different"""

        testMask3 = afwImage.MaskU(self.testMask.getDimensions())
        
        name1 = "Great Timothy"
        name2 = "Our Boss"
        p1 = testMask3.addMaskPlane(name1)
        p2 = testMask3.addMaskPlane(name2)
        oldDict = self.testMask.getMaskPlaneDict()

        testMask3.setMaskPlaneValues(p1, 0, 5, 0)
        testMask3.setMaskPlaneValues(p2, 0, 5, 1)

        if display:
            im = afwImage.ImageF(self.testMask3.getDimensions()); im.set(0)
            ds9.mtv(im)                 # bug in ds9's Mask display; needs an Image first
            ds9.mtv(testMask3)

        self.assertEqual(testMask3.get(0,0), testMask3.getPlaneBitMask(name1))
        self.assertEqual(testMask3.get(0,1), testMask3.getPlaneBitMask(name2))

        self.testMask.removeMaskPlane(name1)
        self.testMask.removeMaskPlane(name2)
        self.testMask.addMaskPlane(name2) # added in opposite order to testMask3
        self.testMask.addMaskPlane(name1)

        self.assertEqual(self.testMask.get(0,0), 0)

        if display:
            ds9.mtv(im, frame=1)
            ds9.mtv(testMask3, frame=1)

        self.assertNotEqual(testMask3.get(0,0), testMask3.getPlaneBitMask(name1))
        self.assertNotEqual(testMask3.get(0,1), testMask3.getPlaneBitMask(name2))

        testMask3.conformMaskPlanes(oldDict)

        self.assertEqual(testMask3.get(0,0), testMask3.getPlaneBitMask(name1))
        self.assertEqual(testMask3.get(0,1), testMask3.getPlaneBitMask(name2))

        if display:
            ds9.mtv(im, frame=2)
            ds9.mtv(testMask3, frame=2)

        self.testMask |= testMask3

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

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
    suites += unittest.makeSuite(OldMaskTestCase) # test suite from vw-based Masks
    suites += unittest.makeSuite(MaskTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(exit=False):
    """Run the tests"""
    utilsTests.run(suite(), exit)

if __name__ == "__main__":
    run(True)
