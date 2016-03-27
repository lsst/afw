#!/usr/bin/env python2
from __future__ import absolute_import, division

#
# LSST Data Management System
# See the COPYRIGHT and LICENSE files in the top-level directory of this
# package for notices and licensing terms.
#

import os
import os.path
import unittest

import lsst.utils
import lsst.utils.tests as utilsTests
import lsst.afw.image as afwImage
import lsst.daf.base as dafBase
import lsst.daf.persistence as dafPers
import lsst.pex.policy as pexPolicy

dataDir = os.path.join(lsst.utils.getPackageDir("afwdata"), "data")
if not dataDir:
    raise RuntimeError("You must set up afwdata to run these tests")

class MaskedImagePersistenceTestCase(unittest.TestCase):
    """A test case for MaskedImage Persistence"""

    def checkImages(self, maskedImage, maskedImage2):
        # Check that two MaskedImages are identical (well, actually check only every 4th pixel)
        pass

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
        self.infile = os.path.join(dataDir, "small_MI.fits")

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

        miPath = os.path.join("tests", "data", "Dest")
        logicalLocation = dafPers.LogicalLocation(miPath)
        storage = self.persistence.getPersistStorage("BoostStorage", logicalLocation)
        storageList = dafPers.StorageList([storage])
        self.persistence.persist(self.maskedImage, storageList, self.additionalData)
      


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(MaskedImagePersistenceTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)

    return unittest.TestSuite(suites)

def run(shouldExit=False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)
