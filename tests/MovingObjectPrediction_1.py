"""
Tests for C++ MovingObjectPrediction and MovingObjectPredictionVector
Python wrappers (including persistence)

Run with:
   python MovingObjectPrediction_1.py
or
   python
   >>> import unittest; T=load("MovingObjectPrediction_1"); unittest.TextTestRunner(verbosity=1).run(T.suite())
"""

import pdb
import unittest
import time
import random
import lsst.mwi.data as data
import lsst.mwi.policy as policy
import lsst.mwi.persistence as persistence
import lsst.mwi.tests as tests
import lsst.afw.Core.fwCatalog as cat

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class MopsPredTestCase(unittest.TestCase):
    """A test case for MopsPred and MopsPredVec"""

    def setUp(self):
        self.mpv1 = cat.MopsPredVec(16)
        self.mpv2 = cat.MopsPredVec()

        for m in xrange(16):
            self.mpv1[m].setId(m)
            ds = cat.MopsPred()
            ds.setId(m)
            ds.setRa(m*20)
            self.mpv2.push_back(ds)

    def tearDown(self):
        del self.mpv1
        del self.mpv2

    def testIterable(self):
        """Check that we can iterate over a MopsPredVec"""
        j = 0
        for i in self.mpv1:
            assert i.getId() == j
            j += 1

    def testCopyAndCompare(self):
        mpv1Copy = cat.MopsPredVec(self.mpv1)
        mpv2Copy = cat.MopsPredVec(self.mpv2)
        assert mpv1Copy == self.mpv1
        assert mpv2Copy == self.mpv2
        mpv1Copy.swap(mpv2Copy)
        assert mpv1Copy == self.mpv2
        assert mpv2Copy == self.mpv1
        mpv1Copy.swap(mpv2Copy)
        if mpv1Copy.size() == 0:
            mpv1Copy.push_back(cat.MopsPred())
        else:
            mpv1Copy.pop_back()
        ds = cat.MopsPred()
        ds.setId(123476519374511136)
        mpv2Copy.push_back(ds)
        assert mpv1Copy != self.mpv1
        assert mpv2Copy != self.mpv2

    def testInsertErase(self):
        mpv1Copy = cat.MopsPredVec(self.mpv1)
        mpv1Copy.insert(mpv1Copy.begin() + 8, cat.MopsPred())
        mpv1Copy.insert(mpv1Copy.begin() + 9, 3, cat.MopsPred())
        mpv1Copy.erase(mpv1Copy.begin() + 8)
        mpv1Copy.erase(mpv1Copy.begin() + 8, mpv1Copy.begin() + 11)
        assert mpv1Copy == self.mpv1

    def testSlice(self):
        slice = self.mpv1[0:3]
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
            dp.addProperty(data.DataProperty("itemName", "MovingObjectPrediction"))
            stl = persistence.StorageList()
            stl.push_back(pers.getPersistStorage("DbStorage", loc))
            pers.persist(self.mpv1, stl, dp)
            stl = persistence.StorageList()
            stl.push_back(pers.getRetrieveStorage("DbStorage", loc))
            persistable = pers.unsafeRetrieve("MovingObjectPredictionVector", stl, dp)
            res = cat.MopsPredVec.swigConvert(persistable)
            cat.dropAllVisitSliceTables(loc, pol, dp)
            assert(res == self.mpv1)
        else:
            print "skipping database tests"

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    tests.init()

    suites = []
    suites += unittest.makeSuite(MopsPredTestCase)
    suites += unittest.makeSuite(tests.MemoryTestCase)
    return unittest.TestSuite(suites)

if __name__ == "__main__":
    tests.run(suite())

