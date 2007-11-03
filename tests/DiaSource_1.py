"""
Tests for C++ DiaSource and DiaSourceVector Python wrappers (including persistence)

Run with:
   python DiaSource_1.py
or
   python
   >>> import unittest; T=load("DiaSource_1"); unittest.TextTestRunner(verbosity=1).run(T.suite())
"""

import pdb
import unittest
import random
import time
import lsst.mwi.data as data
import lsst.mwi.policy as policy
import lsst.mwi.persistence as persistence
import lsst.mwi.tests as tests
import lsst.fw.Core.fwCatalog as cat

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class DiaSourceTestCase(unittest.TestCase):
    """A test case for DiaSource and DiaSourceVec"""

    def setUp(self):
        self.dsv1 = cat.DiaSourceVec(16)
        self.dsv2 = cat.DiaSourceVec()

        for m in xrange(16):
            self.dsv1[m].setId(m)
            ds = cat.DiaSource()
            ds.setId(m)
            ds.setRa(m*20)
            self.dsv2.push_back(ds)

    def tearDown(self):
        del self.dsv1
        del self.dsv2

    def testIterable(self):
        """Check that we can iterate over a DiaSourceVec"""
        j = 0
        for i in self.dsv1:
            assert i.getId() == j
            j += 1

    def testCopyAndCompare(self):
        dsv1Copy = cat.DiaSourceVec(self.dsv1)
        dsv2Copy = cat.DiaSourceVec(self.dsv2)
        assert dsv1Copy == self.dsv1
        assert dsv2Copy == self.dsv2
        dsv1Copy.swap(dsv2Copy)
        assert dsv1Copy == self.dsv2
        assert dsv2Copy == self.dsv1
        dsv1Copy.swap(dsv2Copy)
        if dsv1Copy.size() == 0:
            dsv1Copy.push_back(cat.DiaSource())
        else:
            dsv1Copy.pop_back()
        ds = cat.DiaSource()
        ds.setId(123476519374511136)
        dsv2Copy.push_back(ds)
        assert dsv1Copy != self.dsv1
        assert dsv2Copy != self.dsv2

    def testInsertErase(self):
        dsv1Copy = cat.DiaSourceVec(self.dsv1)
        dsv1Copy.insert(dsv1Copy.begin() + 8, cat.DiaSource())
        dsv1Copy.insert(dsv1Copy.begin() + 9, 3, cat.DiaSource())
        dsv1Copy.erase(dsv1Copy.begin() + 8)
        dsv1Copy.erase(dsv1Copy.begin() + 8, dsv1Copy.begin() + 11)
        assert dsv1Copy == self.dsv1

    def testSlice(self):
        slice = self.dsv1[0:3]
        j = 0
        for i in slice:
            print i
            assert i.getId() == j
            j += 1

    def testPersistence(self):
        if persistence.DbAuth.available():
            pol  = policy.PolicyPtr()
            pers = persistence.Persistence.getPersistence(pol)
            loc  =  persistence.LogicalLocation("mysql://lsst10.ncsa.uiuc.edu:3306/test")
            dp = data.SupportFactory.createPropertyNode("root")
            dp.addProperty(data.DataProperty("visitId", int(time.clock())*16384 + random.randint(0,16383)))
            dp.addProperty(data.DataProperty("sliceId", 0))
            dp.addProperty(data.DataProperty("numSlices", 1))
            dp.addProperty(data.DataProperty("itemName", "DiaSource"))
            stl = persistence.StorageList()
            stl.push_back(pers.getPersistStorage("DbStorage", loc))
            pers.persist(self.dsv1, stl, dp)
            stl = persistence.StorageList()
            stl.push_back(pers.getRetrieveStorage("DbStorage", loc))
            res = cat.DiaSourceVecSharedPtr(pers.retrieve("DiaSourceVector", stl, dp))
            cat.dropAllVisitSliceTables(loc, pol, dp)
            assert(res == self.dsv1)
        else:
            print "skipping database tests"


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    tests.init()

    suites = []
    suites += unittest.makeSuite(DiaSourceTestCase)
    suites += unittest.makeSuite(tests.MemoryTestCase)
    return unittest.TestSuite(suites)

if __name__ == "__main__":
    tests.run(suite())

