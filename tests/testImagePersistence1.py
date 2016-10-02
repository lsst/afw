#pybind11##!/usr/bin/env python
#pybind11#from __future__ import absolute_import, division
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
#pybind11#import os
#pybind11#import unittest
#pybind11#
#pybind11#import lsst.utils
#pybind11#import lsst.utils.tests
#pybind11#import lsst.afw.image as afwImage
#pybind11#import lsst.daf.base as dafBase
#pybind11#import lsst.daf.persistence as dafPers
#pybind11#import lsst.pex.policy as pexPolicy
#pybind11#import lsst.pex.exceptions as pexExcept
#pybind11#
#pybind11#try:
#pybind11#    dataDir = lsst.utils.getPackageDir("afwdata")
#pybind11#except pexExcept.NotFoundError:
#pybind11#    dataDir = None
#pybind11#
#pybind11#
#pybind11#@unittest.skipIf(dataDir is None, "afwdata not setup")
#pybind11#class ImagePersistenceTestCase(lsst.utils.tests.TestCase):
#pybind11#    """A test case for Image Persistence"""
#pybind11#
#pybind11#    def checkImages(self, image, image2):
#pybind11#        # Check that two images are identical (well, actually check only every 4th pixel)
#pybind11#        assert image.getHeight() == image2.getHeight()
#pybind11#        assert image.getWidth() == image2.getWidth()
#pybind11#        assert image.getY0() == image2.getY0()
#pybind11#        assert image.getX0() == image2.getX0()
#pybind11#        for x in range(0, image.getWidth(), 2):
#pybind11#            for y in range(0, image.getHeight(), 2):
#pybind11#                pixel1 = image.get(x, y)
#pybind11#                pixel2 = image2.get(x, y)
#pybind11#                # Persisting through Boost text archives causes conversion error!
#pybind11#                # assert abs(pixel1 - pixel2) / pixel1 < 1e-7, \
#pybind11#                assert pixel1 == pixel2, \
#pybind11#                    "Differing pixel2 at %d, %d: %f, %f" % (x, y, pixel1, pixel2)
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        # Create the additionalData PropertySet
#pybind11#        self.additionalData = dafBase.PropertySet()
#pybind11#        self.additionalData.addInt("sliceId", 0)
#pybind11#        self.additionalData.addString("visitId", "fov391")
#pybind11#        self.additionalData.addInt("universeSize", 100)
#pybind11#        self.additionalData.addString("itemName", "foo")
#pybind11#
#pybind11#        # Create an empty Policy
#pybind11#        policy = pexPolicy.Policy()
#pybind11#
#pybind11#        # Get a Persistence object
#pybind11#        self.persistence = dafPers.Persistence.getPersistence(policy)
#pybind11#
#pybind11#        # Choose a file to manipulate
#pybind11#        self.infile = os.path.join(dataDir, "CFHT", "D4", "cal-53535-i-797722_1.fits")
#pybind11#
#pybind11#        self.image = afwImage.ImageF(self.infile)
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        del self.additionalData
#pybind11#        del self.persistence
#pybind11#        del self.infile
#pybind11#        del self.image
#pybind11#
#pybind11#    def testFitsPersistence(self):
#pybind11#        """Test persisting to FITS"""
#pybind11#
#pybind11#        # Set up the LogicalLocation.
#pybind11#        logicalLocation = dafPers.LogicalLocation(self.infile)
#pybind11#
#pybind11#        # Create a FitsStorage and put it in a StorageList.
#pybind11#        storage = self.persistence.getRetrieveStorage("FitsStorage", logicalLocation)
#pybind11#        storageList = dafPers.StorageList([storage])
#pybind11#
#pybind11#        # Let's do the retrieval!
#pybind11#        persPtr = self.persistence.unsafeRetrieve("ImageF", storageList, self.additionalData)
#pybind11#        image2 = afwImage.ImageF.swigConvert(persPtr)
#pybind11#
#pybind11#        # Check the resulting Image
#pybind11#        self.checkImages(self.image, image2)
#pybind11#
#pybind11#    def testBoostPersistence(self):
#pybind11#        """Persist the image using boost"""
#pybind11#        with lsst.utils.tests.getTempFilePath(".boost") as boostFilePath:
#pybind11#            logicalLocation = dafPers.LogicalLocation(boostFilePath)
#pybind11#            storage = self.persistence.getPersistStorage("BoostStorage", logicalLocation)
#pybind11#            storageList = dafPers.StorageList([storage])
#pybind11#            self.persistence.persist(self.image, storageList, self.additionalData)
#pybind11#
#pybind11#            # Retrieve it again
#pybind11#            storage = self.persistence.getRetrieveStorage("BoostStorage", logicalLocation)
#pybind11#            storageList = dafPers.StorageList([storage])
#pybind11#            pers2Ptr = self.persistence.unsafeRetrieve("ImageF", storageList, self.additionalData)
#pybind11#            image2 = afwImage.ImageF.swigConvert(pers2Ptr)
#pybind11#
#pybind11#            # Check the resulting Image
#pybind11#            self.checkImages(self.image, image2)
#pybind11#
#pybind11#    def testBoostPersistenceU16(self):
#pybind11#        """Persist a U16 image using boost"""
#pybind11#        with lsst.utils.tests.getTempFilePath(".boost") as boostFilePath:
#pybind11#            logicalLocation = dafPers.LogicalLocation(boostFilePath)
#pybind11#            storage = self.persistence.getPersistStorage("BoostStorage", logicalLocation)
#pybind11#            storageList = dafPers.StorageList([storage])
#pybind11#            #
#pybind11#            # Read a U16 image
#pybind11#            #
#pybind11#            self.image = self.image.Factory(os.path.join(dataDir, "data", "small_MI.fits"))
#pybind11#            self.persistence.persist(self.image, storageList, self.additionalData)
#pybind11#
#pybind11#            # Retrieve it again
#pybind11#            storage = self.persistence.getRetrieveStorage("BoostStorage", logicalLocation)
#pybind11#            storageList = dafPers.StorageList([storage])
#pybind11#            pers2Ptr = self.persistence.unsafeRetrieve("ImageF", storageList, self.additionalData)
#pybind11#            image2 = afwImage.ImageF.swigConvert(pers2Ptr)
#pybind11#
#pybind11#            # Check the resulting Image
#pybind11#            self.checkImages(self.image, image2)
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#
#pybind11#
#pybind11#class TestMemory(lsst.utils.tests.MemoryTestCase):
#pybind11#    pass
#pybind11#
#pybind11#def setup_module(module):
#pybind11#    lsst.utils.tests.init()
#pybind11#
#pybind11#if __name__ == "__main__":
#pybind11#    lsst.utils.tests.init()
#pybind11#    unittest.main()
