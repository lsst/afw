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
        container1 = afwDet.SourceContainer(16)

        container2 = afwDet.SourceContainer()
		
        for m in xrange(16):
            container1[m].setId(m+1)           
            ds = afwDet.Source()
            ds.setId(m)
            ds.setRa(m*20)
            container2.append(ds)

        self.dsv1 = afwDet.SourceVec(container1)
        self.dsv2 = afwDet.SourceVec(container2)

    def tearDown(self):
        del self.dsv1
        del self.dsv2

    def testIterable(self):
        """Check that we can iterate over a SourceVec"""
        j = 1
        container = self.dsv1.getSources()
        for s in container[:]:
            assert s.getId() == j
            j += 1

    def testCopyAndCompare(self):
        dsv1Copy = self.dsv1.getSources()
        dsv2Copy = self.dsv2.getSources()
        
        assert dsv1Copy.size() == self.dsv1.getSources().size()
        for i in xrange(dsv1Copy.size()):
            assert dsv1Copy[i] == self.dsv1.getSources()[i]        
        assert dsv2Copy.size() == self.dsv2.getSources().size()
        for i in xrange(dsv2Copy.size()):
            assert dsv2Copy[i] == self.dsv2.getSources()[i]

        dsv1Copy.swap(dsv2Copy)
        assert dsv2Copy.size() == self.dsv1.getSources().size()
        for i in xrange(dsv2Copy.size()):
            assert dsv2Copy[i] == self.dsv1.getSources()[i]           
        assert dsv1Copy.size() == self.dsv2.getSources().size()
        for i in xrange(dsv1Copy.size()):
            assert dsv1Copy[i] == self.dsv2.getSources()[i]
            
        dsv1Copy.swap(dsv2Copy)
        
        if dsv1Copy.size() == 0:
            dsv1Copy.append(afwDet.Source())
        else:
            dsv1Copy.pop()
        ds = afwDet.Source()
        dsv2Copy.append(ds)
        
        assert dsv1Copy.size() != self.dsv1.getSources().size()
        assert dsv2Copy.size() != self.dsv2.getSources().size()

    def testInsertErase(self):
    	dsv1Copy = self.dsv1.getSources()[:8]
    	back = self.dsv1.getSources()[8:]

        s = afwDet.Source()
        for i in xrange(4):
            dsv1Copy.append(s)
		
        for b in back[:]:
            dsv1Copy.append(b)
					
        del dsv1Copy[8]
        del dsv1Copy[8:11]
        assert dsv1Copy.size() == self.dsv1.getSources().size()
        for i in xrange(dsv1Copy.size()):
            assert dsv1Copy[i] == self.dsv1.getSources()[i]       

    def testSlice(self):
        containerSlice = self.dsv1.getSources()[0:3]
        j = 1
        for i in containerSlice[:]:
            assert i.getId() == j
            j += 1

    def testPersistence(self):
        if dafPers.DbAuth.available():
            pol  = dafPolicy.Policy()
            pers = dafPers.Persistence.getPersistence(pol)
            loc  =  dafPers.LogicalLocation("mysql://lsst10.ncsa.uiuc.edu:3306/test")
            dp = dafBase.PropertySet()
            dp.addInt("visitId", int(time.clock())*16384 + random.randint(0,16383))
            dp.addInt("sliceId", 0)
            dp.addInt("numSlices", 1)
            dp.addString("itemName", "Source")
            stl = dafPers.StorageList()
            stl.append(pers.getPersistStorage("DbStorage", loc))
            pers.persist(self.dsv1, stl, dp)
            stl = dafPers.StorageList()
            stl.append(pers.getRetrieveStorage("DbStorage", loc))
            persistable = pers.unsafeRetrieve("PersistableSourceVector", stl, dp)
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

