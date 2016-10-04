#!/usr/bin/env python
from __future__ import absolute_import, division

#
# LSST Data Management System
# Copyright 2008, 2009, 2010 LSST Corporation.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#

import os
import os.path
import unittest

import lsst.utils
import lsst.utils.tests
import lsst.afw.image as afwImage
import lsst.daf.base as dafBase
import lsst.daf.persistence as dafPers
import lsst.pex.policy as pexPolicy
import lsst.pex.exceptions as pexExcept

try:
    dataDir = os.path.join(lsst.utils.getPackageDir("afwdata"), "data")
except pexExcept.NotFoundError:
    dataDir = None


class MaskedImagePersistenceTestCase(lsst.utils.tests.TestCase):
    """A test case for MaskedImage Persistence"""

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
        if dataDir is not None:
            self.infile = os.path.join(dataDir, "small_MI.fits")

            self.maskedImage = afwImage.MaskedImageF(self.infile)

    def tearDown(self):
        if dataDir is not None:
            del self.additionalData
            del self.persistence
            del self.infile
            del self.maskedImage

    @unittest.skipIf(dataDir is None, "afwdata not setup")
    def testFitsPersistence(self):
        """Test persisting to FITS"""

        # Set up the LogicalLocation.
        logicalLocation = dafPers.LogicalLocation(self.infile)

        # Create a FitsStorage and put it in a StorageList.
        storage = self.persistence.getRetrieveStorage("FitsStorage", logicalLocation)
        storageList = dafPers.StorageList([storage])

        # Let's do the retrieval!
        maskedImage2 = afwImage.MaskedImageF.swigConvert(
            self.persistence.unsafeRetrieve("MaskedImageF", storageList, self.additionalData))

        # Check the resulting MaskedImage
        self.assertMaskedImagesEqual(self.maskedImage, maskedImage2)

    @unittest.skipIf(dataDir is None, "afwdata not setup")
    def testBoostPersistence(self):
        """Persist the MaskedImage using boost"""

        miPath = os.path.join("tests", "data", "Dest")
        logicalLocation = dafPers.LogicalLocation(miPath)
        storage = self.persistence.getPersistStorage("BoostStorage", logicalLocation)
        storageList = dafPers.StorageList([storage])
        self.persistence.persist(self.maskedImage, storageList, self.additionalData)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()

if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
