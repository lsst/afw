#pybind11##!/usr/bin/env python
#pybind11#from __future__ import absolute_import, division
#pybind11#from __future__ import print_function
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
#pybind11#
#pybind11#"""
#pybind11#Tests for Masks
#pybind11#
#pybind11#Run with:
#pybind11#   python Mask.py
#pybind11#or
#pybind11#   python
#pybind11#   >>> import Mask; Mask.run()
#pybind11#"""
#pybind11#
#pybind11#import os.path
#pybind11#
#pybind11#import sys
#pybind11#import unittest
#pybind11#import numpy as np
#pybind11#
#pybind11#import lsst.utils
#pybind11#import lsst.utils.tests as utilsTests
#pybind11#import lsst.pex.exceptions as pexExcept
#pybind11#import lsst.daf.base
#pybind11#import lsst.afw.image as afwImage
#pybind11#import lsst.afw.geom as afwGeom
#pybind11#import lsst.afw.display.ds9 as ds9
#pybind11#import lsst.pex.exceptions as pexExcept
#pybind11#
#pybind11#try:
#pybind11#    type(display)
#pybind11#except NameError:
#pybind11#    display = False
#pybind11#
#pybind11#try:
#pybind11#    afwdataDir = lsst.utils.getPackageDir("afwdata")
#pybind11#except pexExcept.NotFoundError:
#pybind11#    afwdataDir = None
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#
#pybind11#
#pybind11#def showMaskDict(d=None, msg=None):
#pybind11#    if not d:
#pybind11#        d = afwImage.MaskU(0, 0)
#pybind11#        if not msg:
#pybind11#            msg = "default"
#pybind11#
#pybind11#    try:
#pybind11#        d = d.getMaskPlaneDict()
#pybind11#    except AttributeError:
#pybind11#        pass
#pybind11#
#pybind11#    if msg:
#pybind11#        print("%-15s" % msg, end=' ')
#pybind11#    print(sorted([(d[p], p) for p in d]))
#pybind11#
#pybind11#
#pybind11#class MaskTestCase(utilsTests.TestCase):
#pybind11#    """A test case for Mask"""
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        np.random.seed(1)
#pybind11#        self.Mask = afwImage.MaskU
#pybind11#
#pybind11#        # Store the default mask planes for later use
#pybind11#        maskPlaneDict = self.Mask().getMaskPlaneDict()
#pybind11#        self.defaultMaskPlanes = sorted(maskPlaneDict, key=maskPlaneDict.__getitem__)
#pybind11#
#pybind11#        # reset so tests will be deterministic
#pybind11#        self.Mask.clearMaskPlaneDict()
#pybind11#        for p in ("BAD", "SAT", "INTRP", "CR", "EDGE"):
#pybind11#            self.Mask.addMaskPlane(p)
#pybind11#
#pybind11#        self.BAD = self.Mask.getPlaneBitMask("BAD")
#pybind11#        self.CR = self.Mask.getPlaneBitMask("CR")
#pybind11#        self.EDGE = self.Mask.getPlaneBitMask("EDGE")
#pybind11#
#pybind11#        self.val1 = self.BAD | self.CR
#pybind11#        self.val2 = self.val1 | self.EDGE
#pybind11#
#pybind11#        self.mask1 = afwImage.MaskU(100, 200)
#pybind11#        self.mask1.set(self.val1)
#pybind11#        self.mask2 = afwImage.MaskU(self.mask1.getDimensions())
#pybind11#        self.mask2.set(self.val2)
#pybind11#
#pybind11#        # TBD: #DM-609 this should be refactored to use @unittest.skipif checks
#pybind11#        # for afwData above the tests that need it.
#pybind11#        if afwdataDir is not None:
#pybind11#            self.maskFile = os.path.join(afwdataDir, "data", "small_MI.fits")
#pybind11#            # Below: what to expect from the mask plane in the above data.
#pybind11#            # For some tests, it is left-shifted by some number of mask planes.
#pybind11#            self.expect = np.zeros((256, 256), dtype='i8')
#pybind11#            self.expect[:, 0] = 1
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        del self.mask1
#pybind11#        del self.mask2
#pybind11#        # Reset the mask plane to the default
#pybind11#        self.Mask.clearMaskPlaneDict()
#pybind11#        for p in self.defaultMaskPlanes:
#pybind11#            self.Mask.addMaskPlane(p)
#pybind11#
#pybind11#    def testArrays(self):
#pybind11#        # could use MaskU(5, 6) but check extent(5, 6) form too
#pybind11#        image1 = afwImage.MaskU(afwGeom.ExtentI(5, 6))
#pybind11#        array1 = image1.getArray()
#pybind11#        self.assertEqual(array1.shape[0], image1.getHeight())
#pybind11#        self.assertEqual(array1.shape[1], image1.getWidth())
#pybind11#        image2 = afwImage.MaskU(array1, False)
#pybind11#        self.assertEqual(array1.shape[0], image2.getHeight())
#pybind11#        self.assertEqual(array1.shape[1], image2.getWidth())
#pybind11#        image3 = afwImage.makeMaskFromArray(array1)
#pybind11#        self.assertEqual(array1.shape[0], image2.getHeight())
#pybind11#        self.assertEqual(array1.shape[1], image2.getWidth())
#pybind11#        self.assertEqual(type(image3), afwImage.MaskU)
#pybind11#        array1[:, :] = np.random.uniform(low=0, high=10, size=array1.shape)
#pybind11#        self.assertMasksEqual(image1, array1)
#pybind11#
#pybind11#    def testInitializeMasks(self):
#pybind11#        val = 0x1234
#pybind11#        msk = afwImage.MaskU(afwGeom.ExtentI(10, 10), val)
#pybind11#        self.assertEqual(msk.get(0, 0), val)
#pybind11#
#pybind11#    def testSetGetMasks(self):
#pybind11#        self.assertEqual(self.mask1.get(0, 0), self.val1)
#pybind11#
#pybind11#    def testOrMasks(self):
#pybind11#        self.mask2 |= self.mask1
#pybind11#        self.mask1 |= self.val2
#pybind11#
#pybind11#        self.assertMasksEqual(self.mask1, self.mask2)
#pybind11#        expect = np.empty_like(self.mask1.getArray())
#pybind11#        expect[:] = self.val2 | self.val1
#pybind11#        self.assertMasksEqual(self.mask1, expect)
#pybind11#
#pybind11#    def testAndMasks(self):
#pybind11#        self.mask2 &= self.mask1
#pybind11#        self.mask1 &= self.val2
#pybind11#
#pybind11#        self.assertMasksEqual(self.mask1, self.mask2)
#pybind11#        expect = np.empty_like(self.mask1.getArray())
#pybind11#        expect[:] = self.val1 & self.val2
#pybind11#        self.assertMasksEqual(self.mask1, expect)
#pybind11#
#pybind11#    def testXorMasks(self):
#pybind11#        self.mask2 ^= self.mask1
#pybind11#        self.mask1 ^= self.val2
#pybind11#
#pybind11#        self.assertMasksEqual(self.mask1, self.mask2)
#pybind11#        expect = np.empty_like(self.mask1.getArray())
#pybind11#        expect[:] = self.val1 ^ self.val2
#pybind11#        self.assertMasksEqual(self.mask1, expect)
#pybind11#
#pybind11#    def testLogicalMasksMismatch(self):
#pybind11#        "Test logical operations on Masks of different sizes"
#pybind11#        i1 = afwImage.MaskU(afwGeom.ExtentI(100, 100))
#pybind11#        i1.set(100)
#pybind11#        i2 = afwImage.MaskU(afwGeom.ExtentI(10, 10))
#pybind11#        i2.set(10)
#pybind11#
#pybind11#        def tst(i1, i2): i1 |= i2
#pybind11#        self.assertRaises(lsst.pex.exceptions.LengthError, tst, i1, i2)
#pybind11#
#pybind11#        def tst2(i1, i2): i1 &= i2
#pybind11#        self.assertRaises(lsst.pex.exceptions.LengthError, tst2, i1, i2)
#pybind11#
#pybind11#        def tst2(i1, i2): i1 ^= i2
#pybind11#        self.assertRaises(lsst.pex.exceptions.LengthError, tst2, i1, i2)
#pybind11#
#pybind11#    def testMaskPlanes(self):
#pybind11#        planes = self.Mask().getMaskPlaneDict()
#pybind11#        self.assertEqual(len(planes), self.Mask.getNumPlanesUsed())
#pybind11#
#pybind11#        for k in sorted(planes.keys()):
#pybind11#            self.assertEqual(planes[k], self.Mask.getMaskPlane(k))
#pybind11#
#pybind11#    def testCopyConstructors(self):
#pybind11#        dmask = afwImage.MaskU(self.mask1, True)  # deep copy
#pybind11#        smask = afwImage.MaskU(self.mask1)  # shallow copy
#pybind11#
#pybind11#        self.mask1 |= 32767             # should only change smask
#pybind11#        temp = np.zeros_like(self.mask1.getArray()) | self.val1
#pybind11#        self.assertMasksEqual(dmask, temp)
#pybind11#        self.assertMasksEqual(smask, self.mask1)
#pybind11#        self.assertMasksEqual(smask, temp | 32767)
#pybind11#
#pybind11#    def testSubmasks(self):
#pybind11#        smask = afwImage.MaskU(self.mask1,
#pybind11#                               afwGeom.Box2I(afwGeom.Point2I(1, 1), afwGeom.ExtentI(3, 2)),
#pybind11#                               afwImage.LOCAL)
#pybind11#        mask2 = afwImage.MaskU(smask.getDimensions())
#pybind11#
#pybind11#        mask2.set(666)
#pybind11#        smask[:] = mask2
#pybind11#
#pybind11#        del smask
#pybind11#        del mask2
#pybind11#
#pybind11#        self.assertEqual(self.mask1.get(0, 0), self.val1)
#pybind11#        self.assertEqual(self.mask1.get(1, 1), 666)
#pybind11#        self.assertEqual(self.mask1.get(4, 1), self.val1)
#pybind11#        self.assertEqual(self.mask1.get(1, 2), 666)
#pybind11#        self.assertEqual(self.mask1.get(4, 2), self.val1)
#pybind11#        self.assertEqual(self.mask1.get(1, 3), self.val1)
#pybind11#
#pybind11#    @unittest.skipIf(afwdataDir is None, "afwdata not setup")
#pybind11#    def testReadFits(self):
#pybind11#        nMaskPlanes0 = self.Mask.getNumPlanesUsed()
#pybind11#        mask = self.Mask(self.maskFile, 3)  # will shift any unrecognised mask planes into unused slots
#pybind11#
#pybind11#        self.assertMasksEqual(mask, self.expect << nMaskPlanes0)
#pybind11#
#pybind11#    @unittest.skipIf(afwdataDir is None, "afwdata not setup")
#pybind11#    def testReadFitsConform(self):
#pybind11#        hdu = 3
#pybind11#        mask = afwImage.MaskU(self.maskFile, hdu, None, afwGeom.Box2I(), afwImage.LOCAL, True)
#pybind11#
#pybind11#        self.assertMasksEqual(mask, self.expect)
#pybind11#
#pybind11#    @unittest.skipIf(afwdataDir is None, "afwdata not setup")
#pybind11#    def testWriteFits(self):
#pybind11#        nMaskPlanes0 = self.Mask.getNumPlanesUsed()
#pybind11#        mask = self.Mask(self.maskFile, 3)
#pybind11#
#pybind11#        self.assertMasksEqual(mask, self.expect << nMaskPlanes0)
#pybind11#
#pybind11#        with utilsTests.getTempFilePath(".fits") as tmpFile:
#pybind11#            mask.writeFits(tmpFile)
#pybind11#
#pybind11#            # Read it back
#pybind11#            md = lsst.daf.base.PropertySet()
#pybind11#            rmask = self.Mask(tmpFile, 0, md)
#pybind11#
#pybind11#            self.assertMasksEqual(mask, rmask)
#pybind11#
#pybind11#            # Check that we wrote (and read) the metadata successfully
#pybind11#            mp_ = "MP_" if True else self.Mask.maskPlanePrefix()  # currently private
#pybind11#            for (k, v) in self.Mask().getMaskPlaneDict().items():
#pybind11#                self.assertEqual(md.get(mp_ + k), v)
#pybind11#
#pybind11#    def testReadWriteXY0(self):
#pybind11#        """Test that we read and write (X0, Y0) correctly"""
#pybind11#        mask = afwImage.MaskU(afwGeom.ExtentI(10, 20))
#pybind11#
#pybind11#        x0, y0 = 1, 2
#pybind11#        mask.setXY0(x0, y0)
#pybind11#        with utilsTests.getTempFilePath(".fits") as tmpFile:
#pybind11#            mask.writeFits(tmpFile)
#pybind11#
#pybind11#            mask2 = mask.Factory(tmpFile)
#pybind11#
#pybind11#            self.assertEqual(mask2.getX0(), x0)
#pybind11#            self.assertEqual(mask2.getY0(), y0)
#pybind11#
#pybind11#    def testMaskInitialisation(self):
#pybind11#        dims = self.mask1.getDimensions()
#pybind11#        factory = self.mask1.Factory
#pybind11#
#pybind11#        self.mask1.set(666)
#pybind11#
#pybind11#        del self.mask1                 # tempt C++ to reuse the memory
#pybind11#        self.mask1 = factory(dims)
#pybind11#        self.assertEqual(self.mask1.get(10, 10), 0)
#pybind11#
#pybind11#        del self.mask1
#pybind11#        self.mask1 = factory(afwGeom.ExtentI(20, 20))
#pybind11#        self.assertEqual(self.mask1.get(10, 10), 0)
#pybind11#
#pybind11#    def testBoundsChecking(self):
#pybind11#        """Check that pixel indexes are checked in python"""
#pybind11#        tsts = []
#pybind11#
#pybind11#        def tst():
#pybind11#            self.mask1.get(-1, 0)
#pybind11#        tsts.append(tst)
#pybind11#
#pybind11#        def tst():
#pybind11#            self.mask1.get(0, -1)
#pybind11#        tsts.append(tst)
#pybind11#
#pybind11#        def tst():
#pybind11#            self.mask1.get(self.mask1.getWidth(), 0)
#pybind11#        tsts.append(tst)
#pybind11#
#pybind11#        def tst():
#pybind11#            self.mask1.get(0, self.mask1.getHeight())
#pybind11#        tsts.append(tst)
#pybind11#
#pybind11#        for tst in tsts:
#pybind11#            self.assertRaises(lsst.pex.exceptions.LengthError, tst)
#pybind11#
#pybind11#    def testCtorWithPlaneDefs(self):
#pybind11#        """Test that we can create a Mask with a given MaskPlaneDict"""
#pybind11#        FOO, val = "FOO", 2
#pybind11#        mask = afwImage.MaskU(100, 200, {FOO: val}
#pybind11#                              )
#pybind11#        mpd = mask.getMaskPlaneDict()
#pybind11#        self.assertIn(FOO, mpd.keys())
#pybind11#        self.assertEqual(mpd[FOO], val)
#pybind11#
#pybind11#    def testImageSlices(self):
#pybind11#        """Test image slicing, which generate sub-images using Box2I under the covers"""
#pybind11#        im = afwImage.MaskU(10, 20)
#pybind11#        im[-3:, -2:] = 0x4
#pybind11#        im[4, 10] = 0x2
#pybind11#        sim = im[1:4, 6:10]
#pybind11#        sim[:] = 0x8
#pybind11#        im[0:4, 0:4] = im[2:6, 8:12]
#pybind11#
#pybind11#        if display:
#pybind11#            ds9.mtv(im)
#pybind11#
#pybind11#        self.assertEqual(im.get(0, 6), 0)
#pybind11#        self.assertEqual(im.get(6, 17), 0)
#pybind11#        self.assertEqual(im.get(7, 18), 0x4)
#pybind11#        self.assertEqual(im.get(9, 19), 0x4)
#pybind11#        self.assertEqual(im.get(1, 6), 0x8)
#pybind11#        self.assertEqual(im.get(3, 9), 0x8)
#pybind11#        self.assertEqual(im.get(4, 10), 0x2)
#pybind11#        self.assertEqual(im.get(4, 9), 0)
#pybind11#        self.assertEqual(im.get(2, 2), 0x2)
#pybind11#        self.assertEqual(im.get(0, 0), 0x8)
#pybind11#
#pybind11#    def testInterpret(self):
#pybind11#        """Interpretation of Mask values"""
#pybind11#        planes = self.Mask().getMaskPlaneDict()
#pybind11#        im = self.Mask(len(planes), 1)
#pybind11#
#pybind11#        allBits = 0
#pybind11#        for i, p in enumerate(planes):
#pybind11#            bitmask = self.Mask.getPlaneBitMask(p)
#pybind11#            allBits |= bitmask
#pybind11#            self.assertEqual(im.interpret(bitmask), p)
#pybind11#            im.getArray()[0, i] = bitmask
#pybind11#            self.assertEqual(im.getAsString(i, 0), p)
#pybind11#        self.assertEqual(self.Mask.interpret(allBits), ",".join(planes.keys()))
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#
#pybind11#
#pybind11#class OldMaskTestCase(unittest.TestCase):
#pybind11#    """A test case for Mask (based on MaskU_1.cc); these are taken over from the DC2 fw tests
#pybind11#    and modified to run with the new (DC3) APIs"""
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        self.Mask = afwImage.MaskU           # the class
#pybind11#
#pybind11#        self.testMask = self.Mask(afwGeom.Extent2I(300, 400), 0)
#pybind11#
#pybind11#        # Store the default mask planes for later use
#pybind11#        maskPlaneDict = self.Mask().getMaskPlaneDict()
#pybind11#        self.defaultMaskPlanes = sorted(maskPlaneDict, key=maskPlaneDict.__getitem__)
#pybind11#
#pybind11#        # reset so tests will be deterministic
#pybind11#        self.Mask.clearMaskPlaneDict()
#pybind11#        for p in ("CR", "BP"):
#pybind11#            self.Mask.addMaskPlane(p)
#pybind11#
#pybind11#        self.region = afwGeom.Box2I(afwGeom.Point2I(100, 300), afwGeom.Extent2I(10, 40))
#pybind11#        self.subTestMask = self.Mask(self.testMask, self.region, afwImage.LOCAL)
#pybind11#
#pybind11#        if False:
#pybind11#            self.pixelList = afwImage.listPixelCoord()
#pybind11#            for x in range(0, 300):
#pybind11#                for y in range(300, 400, 20):
#pybind11#                    self.pixelList.push_back(afwImage.PixelCoord(x, y))
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        del self.testMask
#pybind11#        del self.subTestMask
#pybind11#        del self.region
#pybind11#        # Reset the mask plane to the default
#pybind11#        self.Mask.clearMaskPlaneDict()
#pybind11#        for p in self.defaultMaskPlanes:
#pybind11#            self.Mask.addMaskPlane(p)
#pybind11#
#pybind11#    def testPlaneAddition(self):
#pybind11#        """Test mask plane addition"""
#pybind11#
#pybind11#        nplane = self.testMask.getNumPlanesUsed()
#pybind11#        for p in ("XCR", "XBP"):
#pybind11#            self.assertEqual(self.Mask.addMaskPlane(p), nplane, "Assigning plane %s" % (p))
#pybind11#            nplane += 1
#pybind11#
#pybind11#        def pname(p):
#pybind11#            return "P%d" % p
#pybind11#
#pybind11#        nextra = 8
#pybind11#        for p in range(0, nextra):
#pybind11#            plane = self.Mask.addMaskPlane(pname(p))
#pybind11#
#pybind11#        for p in range(0, nextra):
#pybind11#            self.testMask.removeAndClearMaskPlane(pname(p))
#pybind11#
#pybind11#        self.assertEqual(nplane + nextra, self.Mask.getNumPlanesUsed(), "Adding and removing planes")
#pybind11#
#pybind11#        for p in range(0, nextra):
#pybind11#            self.Mask.removeMaskPlane(pname(p))
#pybind11#
#pybind11#        self.assertEqual(nplane, self.testMask.getNumPlanesUsed(), "Adding and removing planes")
#pybind11#
#pybind11#    def testMetadata(self):
#pybind11#        """Test mask plane metadata interchange with MaskPlaneDict"""
#pybind11#        #
#pybind11#        # Demonstrate that we can extract a MaskPlaneDict into metadata
#pybind11#        #
#pybind11#        metadata = lsst.daf.base.PropertySet()
#pybind11#
#pybind11#        self.Mask.addMaskPlanesToMetadata(metadata)
#pybind11#        for (k, v) in self.Mask().getMaskPlaneDict().items():
#pybind11#            self.assertEqual(metadata.getInt("MP_%s" % k), v)
#pybind11#        #
#pybind11#        # Now add another plane to metadata and make it appear in the mask Dict, albeit
#pybind11#        # in general at another location (hence the getNumPlanesUsed call)
#pybind11#        #
#pybind11#        metadata.addInt("MP_" + "Whatever", self.Mask.getNumPlanesUsed())
#pybind11#
#pybind11#        self.testMask.conformMaskPlanes(self.Mask.parseMaskPlaneMetadata(metadata))
#pybind11#        for (k, v) in self.Mask().getMaskPlaneDict().items():
#pybind11#            self.assertEqual(metadata.getInt("MP_%s" % k), v)
#pybind11#
#pybind11#    def testPlaneOperations(self):
#pybind11#        """Test mask plane operations"""
#pybind11#
#pybind11#        planes = self.Mask().getMaskPlaneDict()
#pybind11#        self.testMask.clearMaskPlane(planes['CR'])
#pybind11#
#pybind11#        if False:
#pybind11#            for p in planes.keys():
#pybind11#                self.testMask.setMaskPlaneValues(planes[p], self.pixelList)
#pybind11#
#pybind11#        #printMaskPlane(self.testMask, planes['CR'])
#pybind11#
#pybind11#        # print "\nClearing mask"
#pybind11#        self.testMask.clearMaskPlane(planes['CR'])
#pybind11#
#pybind11#        #printMaskPlane(self.testMask, planes['CR'])
#pybind11#
#pybind11#    def testPlaneRemoval(self):
#pybind11#        """Test mask plane removal"""
#pybind11#
#pybind11#        def checkPlaneBP():
#pybind11#            self.Mask.getMaskPlane("BP")
#pybind11#
#pybind11#        testMask2 = self.Mask(self.testMask.getDimensions())
#pybind11#        self.testMask = self.Mask(self.testMask.getDimensions())
#pybind11#        self.testMask.removeAndClearMaskPlane("BP")
#pybind11#
#pybind11#        d = testMask2.getMaskPlaneDict()
#pybind11#
#pybind11#        checkPlaneBP()                                        # still present in default mask
#pybind11#        self.assertIn("BP", testMask2.getMaskPlaneDict())  # should still be in testMask2
#pybind11#
#pybind11#        self.Mask.removeMaskPlane("BP")  # remove from default mask too
#pybind11#
#pybind11#        self.assertRaises(pexExcept.InvalidParameterError, checkPlaneBP)
#pybind11#
#pybind11#        self.assertRaises(pexExcept.InvalidParameterError,
#pybind11#                          lambda: self.Mask.removeMaskPlane("BP"))  # Plane is already removed
#pybind11#        self.assertRaises(pexExcept.InvalidParameterError,
#pybind11#                          lambda: self.testMask.removeMaskPlane("RHL gets names right"))
#pybind11#        #
#pybind11#        self.Mask.clearMaskPlaneDict()
#pybind11#        p0 = self.Mask.addMaskPlane("P0")
#pybind11#        p1 = self.Mask.addMaskPlane("P1")
#pybind11#        p1 = self.Mask.addMaskPlane("P1")		# a no-op -- readding a plane has no effect
#pybind11#        #
#pybind11#        # Check that removing default mask planes doesn't affect pre-existing planes
#pybind11#        #
#pybind11#        msk = self.Mask()
#pybind11#        nmask = len(msk.getMaskPlaneDict())
#pybind11#        self.Mask.removeMaskPlane("P0")
#pybind11#        self.Mask.removeMaskPlane("P1")
#pybind11#        self.assertEqual(len(msk.getMaskPlaneDict()), nmask)
#pybind11#        del msk
#pybind11#        #
#pybind11#        # Check that removeAndClearMaskPlane can clear the default too
#pybind11#        #
#pybind11#        self.Mask.addMaskPlane("BP")
#pybind11#        self.testMask.removeAndClearMaskPlane("BP", True)
#pybind11#
#pybind11#        self.assertRaises(pexExcept.InvalidParameterError, checkPlaneBP)
#pybind11#
#pybind11#    def testInvalidPlaneOperations(self):
#pybind11#        """Test mask plane operations invalidated by Mask changes"""
#pybind11#
#pybind11#        testMask3 = self.Mask(self.testMask.getDimensions())
#pybind11#
#pybind11#        name = "Great Timothy"
#pybind11#        self.Mask.addMaskPlane(name)
#pybind11#        testMask3.removeAndClearMaskPlane(name)
#pybind11#
#pybind11#        self.Mask.getMaskPlane(name)    # should be fine
#pybind11#        self.assertRaises(IndexError, lambda: testMask3.getMaskPlaneDict()[name])
#pybind11#
#pybind11#        def tst():
#pybind11#            self.testMask |= testMask3
#pybind11#
#pybind11#        self.assertRaises(pexExcept.RuntimeError, tst)
#pybind11#
#pybind11#        self.Mask.addMaskPlane(name)    # The dictionary should be back to the same state, so ...
#pybind11#        tst                             # ... assertion should not fail
#pybind11#
#pybind11#        self.testMask.removeAndClearMaskPlane(name, True)
#pybind11#        self.Mask.addMaskPlane("Mario")  # takes name's slot
#pybind11#        self.Mask.addMaskPlane(name)
#pybind11#
#pybind11#        self.assertRaises(pexExcept.RuntimeError, tst)
#pybind11#
#pybind11#    def testInvalidPlaneOperations2(self):
#pybind11#        """Test mask plane operations invalidated by Mask changes"""
#pybind11#
#pybind11#        testMask3 = self.Mask(self.testMask.getDimensions())
#pybind11#
#pybind11#        name = "Great Timothy"
#pybind11#        name2 = "Our Boss"
#pybind11#        self.Mask.addMaskPlane(name)
#pybind11#        self.Mask.addMaskPlane(name2)
#pybind11#        oldDict = testMask3.getMaskPlaneDict()  # a description of the Mask's current dictionary
#pybind11#
#pybind11#        for n in (name, name2):
#pybind11#            self.testMask.removeAndClearMaskPlane(n, True)
#pybind11#
#pybind11#        self.Mask.addMaskPlane(name2)        # added in opposite order to the planes in testMask3
#pybind11#        self.Mask.addMaskPlane(name)
#pybind11#
#pybind11#        self.assertNotEqual(self.testMask.getMaskPlaneDict()[name], oldDict[name])
#pybind11#
#pybind11#        def tst():
#pybind11#            self.testMask |= testMask3
#pybind11#
#pybind11#        self.testMask.removeAndClearMaskPlane("BP")
#pybind11#
#pybind11#        self.assertRaises(pexExcept.RuntimeError, tst)
#pybind11#        #
#pybind11#        # OK, that failed as it should.  Fixup the dictionaries and try again
#pybind11#        #
#pybind11#        self.Mask.addMaskPlane("BP")
#pybind11#        testMask3.conformMaskPlanes(oldDict)  # convert testMask3 from oldDict to current default
#pybind11#
#pybind11#        self.testMask |= testMask3      # shouldn't throw
#pybind11#
#pybind11#    def testConformMaskPlanes(self):
#pybind11#        """Test conformMaskPlanes() when the two planes are actually the same"""
#pybind11#
#pybind11#        testMask3 = self.Mask(self.testMask.getDimensions())
#pybind11#
#pybind11#        name = "XXX"
#pybind11#        self.Mask.addMaskPlane(name)
#pybind11#        oldDict = testMask3.getMaskPlaneDict()
#pybind11#        testMask3.removeAndClearMaskPlane(name)  # invalidates dictionary version
#pybind11#
#pybind11#        testMask3.conformMaskPlanes(oldDict)
#pybind11#
#pybind11#        self.testMask |= testMask3
#pybind11#
#pybind11#    def testConformMaskPlanes2(self):
#pybind11#        """Test conformMaskPlanes() when the two planes are different"""
#pybind11#
#pybind11#        testMask3 = afwImage.MaskU(self.testMask.getDimensions())
#pybind11#
#pybind11#        name1 = "Great Timothy"
#pybind11#        name2 = "Our Boss"
#pybind11#        p1 = self.Mask.addMaskPlane(name1)
#pybind11#        p2 = self.Mask.addMaskPlane(name2)
#pybind11#        oldDict = self.testMask.getMaskPlaneDict()
#pybind11#
#pybind11#        testMask3.setMaskPlaneValues(p1, 0, 5, 0)
#pybind11#        testMask3.setMaskPlaneValues(p2, 0, 5, 1)
#pybind11#
#pybind11#        if display:
#pybind11#            im = afwImage.ImageF(testMask3.getDimensions())
#pybind11#            ds9.mtv(im)                 # bug in ds9's Mask display; needs an Image first
#pybind11#            ds9.mtv(testMask3)
#pybind11#
#pybind11#        self.assertEqual(testMask3.get(0, 0), testMask3.getPlaneBitMask(name1))
#pybind11#        self.assertEqual(testMask3.get(0, 1), testMask3.getPlaneBitMask(name2))
#pybind11#
#pybind11#        self.testMask.removeAndClearMaskPlane(name1, True)
#pybind11#        self.testMask.removeAndClearMaskPlane(name2, True)
#pybind11#        self.Mask.addMaskPlane(name2)  # added in opposite order to testMask3
#pybind11#        self.Mask.addMaskPlane(name1)
#pybind11#
#pybind11#        self.assertEqual(self.testMask.get(0, 0), 0)
#pybind11#
#pybind11#        if display:
#pybind11#            ds9.mtv(im, frame=1)
#pybind11#            ds9.mtv(testMask3, frame=1)
#pybind11#
#pybind11#        self.assertNotEqual(testMask3.get(0, 0), testMask3.getPlaneBitMask(name1))
#pybind11#        self.assertNotEqual(testMask3.get(0, 1), testMask3.getPlaneBitMask(name2))
#pybind11#
#pybind11#        testMask3.conformMaskPlanes(oldDict)
#pybind11#
#pybind11#        self.assertEqual(testMask3.get(0, 0), testMask3.getPlaneBitMask(name1))
#pybind11#        self.assertEqual(testMask3.get(0, 1), testMask3.getPlaneBitMask(name2))
#pybind11#
#pybind11#        if display:
#pybind11#            ds9.mtv(im, frame=2)
#pybind11#            ds9.mtv(testMask3, frame=2)
#pybind11#
#pybind11#        self.testMask |= testMask3
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#pybind11#
#pybind11#
#pybind11#def printMaskPlane(mask, plane,
#pybind11#                   xrange=list(range(250, 300, 10)), yrange=list(range(300, 400, 20))):
#pybind11#    """Print parts of the specified plane of the mask"""
#pybind11#
#pybind11#    xrange = list(range(min(xrange), max(xrange), 25))
#pybind11#    yrange = list(range(min(yrange), max(yrange), 25))
#pybind11#
#pybind11#    for x in xrange:
#pybind11#        for y in yrange:
#pybind11#            if False:                   # mask(x,y) confuses swig
#pybind11#                print(x, y, mask(x, y), mask(x, y, plane))
#pybind11#            else:
#pybind11#                print(x, y, mask(x, y, plane))
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#
#pybind11#
#pybind11#class TestMemory(lsst.utils.tests.MemoryTestCase):
#pybind11#    pass
#pybind11#
#pybind11#
#pybind11#def setup_module(module):
#pybind11#    lsst.utils.tests.init()
#pybind11#
#pybind11#if __name__ == "__main__":
#pybind11#    lsst.utils.tests.init()
#pybind11#    unittest.main()
