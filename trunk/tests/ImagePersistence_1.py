#!/usr/bin/env python
import os
import pdb                          # we may want to say pdb.set_trace()
import unittest
import lsst.utils.tests as utilsTests
import lsst.afw.image as afwImage
import lsst.daf.base as dafBase
import lsst.daf.persistence as dafPers
import lsst.pex.policy as pexPolicy
import eups

dataDir = eups.productDir("afwdata")
if not dataDir:
    raise RuntimeError("You must set up afwdata to run these tests")

class ImagePersistenceTestCase(unittest.TestCase):
    """A test case for Image Persistence"""

    def checkImages(self, image, image2):
        # Check that two images are identical (well, actually check only every 4th pixel)
        assert image.getRows() == image2.getRows()
        assert image.getCols() == image2.getCols()
        assert image.getOffsetRows() == image2.getOffsetRows()
        assert image.getOffsetCols() == image2.getOffsetCols()
        for c in xrange(0, image.getCols(), 2):
            for r in xrange(0, image.getRows(), 2):
                pixel1 = image.getVal(c, r)
                pixel2 = image2.getVal(c, r)
                # Persisting through Boost text archives causes conversion error!
                # assert abs(pixel1 - pixel2) / pixel1 < 1e-7, \
                assert pixel1 == pixel2, \
                        "Differing pixel2 at %d, %d: %f, %f" % (c, r, pixel1, pixel2)

    def setUp(self):
        # Create the additionalData DataProperty
        self.additionalData = dafBase.DataProperty.createPropertyNode("root")
        self.additionalData.addProperty(dafBase.DataProperty("sliceId", 0))
        self.additionalData.addProperty(dafBase.DataProperty("visitId", "fov391"))
        self.additionalData.addProperty(dafBase.DataProperty("universeSize", 100))
        self.additionalData.addProperty(dafBase.DataProperty("itemName", "foo"))

        # Create an empty Policy
        policy = pexPolicy.PolicyPtr()

        # Get a Persistence object
        self.persistence = dafPers.Persistence.getPersistence(policy)

        # Choose a file to manipulate
        self.infile = os.path.join(dataDir, "CFHT", "D4", "cal-53535-i-797722_1_img.fits")

        self.image = afwImage.ImageF(); self.image.readFits(self.infile)

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
        logicalLocation = dafPers.LogicalLocation("image.boost")
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

    if False:                           #  Crashes on RHL's os/x box -- #336
        def testBoostPersistenceU16(self):
            """Persist a U16 image using boost"""
            logicalLocation = dafPers.LogicalLocation("image.boost")
            storage = self.persistence.getPersistStorage("BoostStorage", logicalLocation)
            storageList = dafPers.StorageList([storage])
            #
            # Read a U16 image
            #
            self.image.readFits(os.path.join(dataDir, "small_MI_img.fits"))
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

def run(exit=False):
    """Run the tests"""
    utilsTests.run(suite(), exit)

if __name__ == "__main__":
    run(True)
