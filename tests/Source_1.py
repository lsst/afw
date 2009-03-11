#!/usr/bin/env python
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
    """A test case for Source and PersistableSourceVector"""

    def setUp(self):
        self.container1 = afwDet.SourceSet(16)
        self.container2 = afwDet.SourceSet()
        
        for m in xrange(16):
            ds = afwDet.Source()
            ds.setId(m + 1)
            self.container1[m] = ds
            
            ds = afwDet.Source()
            ds.setId(m)
            ds.setRa(m*20)
            self.container2.push_back(ds)

        self.dsv1 = afwDet.PersistableSourceVector(self.container1)
        self.dsv2 = afwDet.PersistableSourceVector(self.container2)

    def tearDown(self):
        del self.dsv1
        del self.dsv2

    def testIterable(self):
        """Check that we can iterate over a SourceSet"""
        j = 1
        container = self.container1[:]
        for s in container:
            assert s.getId() == j
            j += 1

    def testCopyAndCompare(self):
        dsv1Copy = self.dsv1.getSources()
        dsv2Copy = self.dsv2.getSources()
        
        assert dsv1Copy.size() == self.container1.size()
        for i in xrange(dsv1Copy.size()):
            assert dsv1Copy[i] == self.container1[i]        
        assert dsv2Copy.size() == self.container2.size()
        for i in xrange(dsv2Copy.size()):
            assert dsv2Copy[i] == self.container2[i]

        dsv1Copy.swap(dsv2Copy)
        assert dsv2Copy.size() == self.container1.size()
        for i in xrange(dsv2Copy.size()):
            assert dsv2Copy[i] == self.container1[i]           
        assert dsv1Copy.size() == self.container2.size()
        for i in xrange(dsv1Copy.size()):
            assert dsv1Copy[i] == self.container2[i]
            
        dsv1Copy.swap(dsv2Copy)
        
        if dsv1Copy.size() == 0:
            ds = afwDet.Source()
            dsv1Copy.append(ds)
        else:
            dsv1Copy.pop()
        ds = afwDet.Source()
        dsv2Copy.append(ds)
        
        assert dsv1Copy.size() != self.container1.size()
        assert dsv2Copy.size() != self.container2.size()

    def testInsertErase(self):
        container =  self.dsv1.getSources()
        
        front = container[:8]
        back = container[8:]

        copy = afwDet.SourceSet()
        
        for i in xrange(front.size()):
            copy.append(front[i])
            
        ds = afwDet.Source()
        for i in xrange(4):
            copy.append(ds)
        
        for i in xrange(back.size()):
            copy.append(back[i])
                    
        del copy[8]
        del copy[8:11]
        assert copy.size() == self.container1.size()
        for i in xrange(copy.size()):
            assert copy[i] == self.container1[i]       

    def testSlice(self):
        containerSlice = self.dsv1.getSources()[0:3]
        
        j = 1
        for s in containerSlice:
            assert s.getId() == j
            j += 1

    def testPersistence(self):
        if dafPers.DbAuth.available("lsst10.ncsa.uiuc.edu", "3306"):
            pol  = dafPolicy.Policy()
            pol.set("Formatter.PersistableSourceVector.Source.templateTableName", "Source")
            pol.set("Formatter.PersistableSourceVector.Source.perVisitTableNamePattern", "_tmp_visit%1%_Source")
            pers = dafPers.Persistence.getPersistence(pol)
            loc  = dafPers.LogicalLocation("mysql://lsst10.ncsa.uiuc.edu:3306/test_source")
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
            res = afwDet.PersistableSourceVector.swigConvert(persistable)
            afwDet.dropAllVisitSliceTables(loc, pol.getPolicy("Formatter.PersistableSourceVector"), dp)
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

