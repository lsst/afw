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

import lsst.daf.base as dafBase
import lsst.pex.policy as dafPolicy
import lsst.daf.persistence as dafPers
import lsst.utils.tests as utilsTests
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
            self.dsv2.append(ds)

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
            dsv1Copy.append(afwDet.Source())
        else:
            dsv1Copy.pop()
        ds = afwDet.Source()
        ds.setId(123476519374511136)
        dsv2Copy.append(ds)
        assert dsv1Copy != self.dsv1
        assert dsv2Copy != self.dsv2

    def testInsertErase(self):
        dsv1Copy = afwDet.SourceVec(self.dsv1)
	s = afwDet.Source()
        dsv1Copy.insert(8, s)
        dsv1Copy.insert(8, s)
        dsv1Copy.insert(8, s)
        dsv1Copy.insert(8, s)
        del dsv1Copy[8]
        del dsv1Copy[8:11]
        assert dsv1Copy == self.dsv1

    def testSlice(self):
        s = self.dsv1[0:3]
        j = 1
        for i in s:
            print i
            assert i.getId() == j
            j += 1

    def testPersistence(self):
        if dafPers.DbAuth.available():
            pol  = dafPolicy.Policy()
            pers = dafPers.Persistence.getPersistence(pol)
            loc  =  dafPers.LogicalLocation("mysql://lsst10.ncsa.uiuc.edu:3306/test")
            dp = dafBase.DataProperty.createPropertyNode("root")
            dp.addProperty(dafBase.DataProperty("visitId", int(time.clock())*16384 + random.randint(0,16383)))
            dp.addProperty(dafBase.DataProperty("sliceId", 0))
            dp.addProperty(dafBase.DataProperty("numSlices", 1))
            dp.addProperty(dafBase.DataProperty("itemName", "Source"))
            stl = dafPers.StorageList()
            stl.append(pers.getPersistStorage("DbStorage", loc))
            pers.persist(self.dsv1, stl, dp)
            stl = dafPers.StorageList()
            stl.append(pers.getRetrieveStorage("DbStorage", loc))
            persistable = pers.unsafeRetrieve("SourceVector", stl, dp)
            res = afwDet.SourceVec.swigConvert(persistable)
            afwDet.dropAllVisitSliceTables(loc, pol, dp)
            assert(res == self.dsv1)
        else:
            print "skipping database tests"


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(SourceTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

if __name__ == "__main__":
    utilsTests.run(suite())

