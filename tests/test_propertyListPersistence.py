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
import unittest

import lsst.utils
import lsst.utils.tests
import lsst.daf.persistence as dafPers
import lsst.pex.policy as pexPolicy
import lsst.afw.image as afwImage  # noqa F401 import the formatter for PropertyList


class PropertyListPersistenceTestCase(lsst.utils.tests.TestCase):
    """A test case for PropertyList Persistence"""

    def setUp(self):
        # Get a Persistence object
        policy = pexPolicy.Policy()
        self.persistence = dafPers.Persistence.getPersistence(policy)

    def tearDown(self):
        del self.persistence

    def testFitsPersistence(self):
        """Test unpersisting from FITS"""

        # Set up the LogicalLocation.
        testPath = os.path.abspath(os.path.dirname(__file__))
        logicalLocation = dafPers.LogicalLocation(
            os.path.join(testPath, "data", "HSC-0908120-056-small.fits"))

        # Create a FitsStorage and put it in a StorageList.
        storage = self.persistence.getRetrieveStorage(
            "FitsStorage", logicalLocation)
        storageList = dafPers.StorageList([storage])

        # Let's do the retrieval!
        propertyList = self.persistence.unsafeRetrieve(
            "PropertyList", storageList, None)

        self.assertEqual(propertyList.get("AR_HDU"), 5)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
