#!/usr/bin/env python
import os
import pdb                          # we may want to say pdb.set_trace()
import unittest
import lsst.utils.tests as utilsTests

import lsst.afw.image as afwImage
import lsst.daf.base as dafBase
import lsst.daf.persistence as dafPers
import lsst.pex.policy as pexPolicy
import lsst.pex.exceptions as pexExceptions
import eups

dataDir = eups.productDir("afwdata")
if not dataDir:
    raise RuntimeError("You must set up afwdata to run these tests")

class MaskedImagePersistenceTestCase(unittest.TestCase):
    """A test case for MaskedImage Persistence"""

    def checkImages(self, maskedImage, maskedImage2):
        # Check that two MaskedImages are identical (well, actually check only every 4th pixel)
        pass

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
        self.infile = os.path.join(dataDir, "small_MI")

        self.maskedImage = afwImage.MaskedImageF(self.infile)
        
    def tearDown(self):
        del self.additionalData
        del self.persistence
        del self.infile
        del self.maskedImage

    def testFitsPersistence(self):
        """Test persisting to FITS"""

        # Set up the LogicalLocation.
        logicalLocation = dafPers.LogicalLocation(self.infile)

        # Create a FitsStorage and put it in a StorageList.
        storage = self.persistence.getRetrieveStorage("FitsStorage", logicalLocation)
        storageList = dafPers.StorageList([storage])

        # Let's do the retrieval!
        maskedImage2 = afwImage.MaskedImageF.swigConvert( \
            self.persistence.unsafeRetrieve("MaskedImageF", storageList, self.additionalData))

        # Check the resulting MaskedImage
        self.checkImages(self.maskedImage, maskedImage2)

    def testBoostPersistence(self):
        """Persist the MaskedImage using boost"""

        logicalLocation = dafPers.LogicalLocation("Dest")
        storage = self.persistence.getPersistStorage("FitsStorage", logicalLocation)
        storageList = dafPers.StorageList([storage])
        try:
            self.persistence.persist(self.maskedImage, storageList, self.additionalData)
        except pexExceptions.LsstInvalidParameter, e:
            print e.what()
            raise


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(MaskedImagePersistenceTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)

    return unittest.TestSuite(suites)

def run(exit=False):
    """Run the tests"""
    utilsTests.run(suite(), exit)

if __name__ == "__main__":
    run(True)
