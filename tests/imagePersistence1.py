#!/usr/bin/env python2
from __future__ import absolute_import, division

#
# LSST Data Management System
# See the COPYRIGHT and LICENSE files in the top-level directory of this
# package for notices and licensing terms.
#

import os
import unittest

import lsst.utils
import lsst.utils.tests as utilsTests
import lsst.afw.image as afwImage
import lsst.daf.base as dafBase
import lsst.daf.persistence as dafPers
import lsst.pex.policy as pexPolicy

dataDir = lsst.utils.getPackageDir("afwdata")

class ImagePersistenceTestCase(unittest.TestCase):
    """A test case for Image Persistence"""

    def checkImages(self, image, image2):
        # Check that two images are identical (well, actually check only every 4th pixel)
        assert image.getHeight() == image2.getHeight()
        assert image.getWidth() == image2.getWidth()
        assert image.getY0() == image2.getY0()
        assert image.getX0() == image2.getX0()
        for x in xrange(0, image.getWidth(), 2):
            for y in xrange(0, image.getHeight(), 2):
                pixel1 = image.get(x, y)
                pixel2 = image2.get(x, y)
                # Persisting through Boost text archives causes conversion error!
                # assert abs(pixel1 - pixel2) / pixel1 < 1e-7, \
                assert pixel1 == pixel2, \
                        "Differing pixel2 at %d, %d: %f, %f" % (x, y, pixel1, pixel2)

    def setUp(self):
        # Create the additionalData PropertySet
        self.additionalData = dafBase.PropertySet()
        self.additionalData.addInt("sliceId", 0)
        self.additionalData.addString("visitId", "fov391")
        self.additionalData.addInt("universeSize", 100)
        self.additionalData.addString("itemName", "foo")

        # Create an empty Policy
        policy = pexPolicy.Policy()

        # Get a Persistence object
        self.persistence = dafPers.Persistence.getPersistence(policy)

        # Choose a file to manipulate
        self.infile = os.path.join(dataDir, "CFHT", "D4", "cal-53535-i-797722_1.fits")

        self.image = afwImage.ImageF(self.infile)

    def tearDown(self):
        del self.additionalData
        del self.persistence
        del self.infile
        del self.image

    def testFitsPersistence(self):
        """Test persisting to FITS"""

        # Set up the LogicalLocation.
        logicalLocation = dafPers.LogicalLocation(self.infile)

        # Create a FitsStorage and put it in a StorageList.
        storage = self.persistence.getRetrieveStorage("FitsStorage", logicalLocation)
        storageList = dafPers.StorageList([storage])

        # Let's do the retrieval!
        persPtr = self.persistence.unsafeRetrieve("ImageF", storageList, self.additionalData)
        image2 = afwImage.ImageF.swigConvert(persPtr)

        # Check the resulting Image
        self.checkImages(self.image, image2)

    def testBoostPersistence(self):
        """Persist the image using boost"""
        with utilsTests.getTempFilePath(".boost") as boostFilePath:
            logicalLocation = dafPers.LogicalLocation(boostFilePath)
            storage = self.persistence.getPersistStorage("BoostStorage", logicalLocation)
            storageList = dafPers.StorageList([storage])
            self.persistence.persist(self.image, storageList, self.additionalData)

            # Retrieve it again
            storage = self.persistence.getRetrieveStorage("BoostStorage", logicalLocation)
            storageList = dafPers.StorageList([storage])
            pers2Ptr = self.persistence.unsafeRetrieve("ImageF", storageList, self.additionalData)
            image2 = afwImage.ImageF.swigConvert(pers2Ptr)
            
            # Check the resulting Image
            self.checkImages(self.image, image2)

    def testBoostPersistenceU16(self):
        """Persist a U16 image using boost"""
        with utilsTests.getTempFilePath(".boost") as boostFilePath:
            logicalLocation = dafPers.LogicalLocation(boostFilePath)
            storage = self.persistence.getPersistStorage("BoostStorage", logicalLocation)
            storageList = dafPers.StorageList([storage])
            #
            # Read a U16 image
            #
            self.image = self.image.Factory(os.path.join(dataDir, "data", "small_MI.fits"))
            self.persistence.persist(self.image, storageList, self.additionalData)

            # Retrieve it again
            storage = self.persistence.getRetrieveStorage("BoostStorage", logicalLocation)
            storageList = dafPers.StorageList([storage])
            pers2Ptr = self.persistence.unsafeRetrieve("ImageF", storageList, self.additionalData)
            image2 = afwImage.ImageF.swigConvert(pers2Ptr)

            # Check the resulting Image
            self.checkImages(self.image, image2)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(ImagePersistenceTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)

    return unittest.TestSuite(suites)

def run(shouldExit=False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)
