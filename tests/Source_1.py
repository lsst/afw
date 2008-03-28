"""
Tests for C++ Source and SourceVector Python wrappers (including persistence)

Run with:
   python Source_1.py
or
   python
   >>> import unittest; T=load("Source_1"); unittest.TextTestRunner(verbosity=1).run(T.suite())
"""
import pdb
import unittest
import random
import time

import lsst.daf.data as dafData
import lsst.pex.policy as dafPolicy
import lsst.daf.persistence as dafPers
import lsst.daf.tests as dafTests
import lsst.afw.detection as afwDet

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class SourceTestCase(unittest.TestCase):
    """A test case for Source and SourceVec"""

    def setUp(self):
        self.dsv1 = afwDet.SourceVec(16)
        self.dsv2 = afwDet.SourceVec()

        for m in xrange(16):
            self.dsv1[m].setId(m + 1)
            ds = afwDet.Source()
            ds.setId(m)
            ds.setRa(m*20)
            self.dsv2.push_back(ds)

    def tearDown(self):
        del self.dsv1
        del self.dsv2

    def testIterable(self):
        """Check that we can iterate over a SourceVec"""
        j = 1
        for i in self.dsv1:
            assert i.getId() == j
            j += 1

    def testCopyAndCompare(self):
        dsv1Copy = afwDet.SourceVec(self.dsv1)
        dsv2Copy = afwDet.SourceVec(self.dsv2)
        assert dsv1Copy == self.dsv1
        assert dsv2Copy == self.dsv2
        dsv1Copy.swap(dsv2Copy)
        assert dsv1Copy == self.dsv2
        assert dsv2Copy == self.dsv1
        dsv1Copy.swap(dsv2Copy)
        if dsv1Copy.size() == 0:
            dsv1Copy.push_back(afwDet.Source())
        else:
            dsv1Copy.pop_back()
        ds = afwDet.Source()
        ds.setId(123476519374511136)
        dsv2Copy.push_back(ds)
        assert dsv1Copy != self.dsv1
        assert dsv2Copy != self.dsv2

    def testInsertErase(self):
        dsv1Copy = afwDet.SourceVec(self.dsv1)
        dsv1Copy.insert(dsv1Copy.begin() + 8, afwDet.Source())
        dsv1Copy.insert(dsv1Copy.begin() + 9, 3, afwDet.Source())
        dsv1Copy.erase(dsv1Copy.begin() + 8)
        dsv1Copy.erase(dsv1Copy.begin() + 8, dsv1Copy.begin() + 11)
        assert dsv1Copy == self.dsv1

    def testSlice(self):
        slice = self.dsv1[0:3]
        j = 1
        for i in slice:
            print i
            assert i.getId() == j
            j += 1

    def testPersistence(self):
        if dafPers.DbAuth.available():
            pol  = dafPolicy.PolicyPtr()
            pers = dafPers.Persistence.getPersistence(pol)
            loc  =  dafPers.LogicalLocation("mysql://lsst10.ncsa.uiuc.edu:3306/test")
            dp = dafData.SupportFactory.createPropertyNode("root")
            dp.addProperty(dafData.DataProperty("visitId", int(time.clock())*16384 + random.randint(0,16383)))
            dp.addProperty(dafData.DataProperty("sliceId", 0))
            dp.addProperty(dafData.DataProperty("numSlices", 1))
            dp.addProperty(dafData.DataProperty("itemName", "Source"))
            stl = dafPers.StorageList()
            stl.push_back(pers.getPersistStorage("DbStorage", loc))
            pers.persist(self.dsv1, stl, dp)
            stl = dafPers.StorageList()
            stl.push_back(pers.getRetrieveStorage("DbStorage", loc))
            persistable = pers.unsafeRetrieve("SourceVector", stl, dp)
            res = afwDet.SourceVec.swigConvert(persistable)
            afwDet.dropAllVisitSliceTables(loc, pol, dp)
            assert(res == self.dsv1)
        else:
            print "skipping database tests"


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    dafTests.init()

    suites = []
    suites += unittest.makeSuite(SourceTestCase)
    suites += unittest.makeSuite(dafTests.MemoryTestCase)
    return unittest.TestSuite(suites)

if __name__ == "__main__":
    dafTests.run(suite())

