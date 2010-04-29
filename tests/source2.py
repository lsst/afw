#!/usr/bin/env python
"""
Tests for C++ DiaSource and PersistableDiaSourceVector Python wrappers (including persistence)

Run with:
   python Source_2.py
or
   python
   >>> import unittest; T=load("Source_2"); unittest.TextTestRunner(verbosity=1).run(T.suite())
"""

import unittest
import random
import tempfile
import time

import lsst.daf.base as dafBase
import lsst.pex.policy as dafPolicy
import lsst.daf.persistence as dafPers
import lsst.utils.tests as utilsTests
import lsst.afw.detection as afwDet

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class DiaSourceTestCase(unittest.TestCase):
    """A test case for DiaSource and PersistableDiaSourceVector"""

    def setUp(self):
        self.container1 = afwDet.DiaSourceSet(16)
        self.container2 = afwDet.DiaSourceSet()
        
        for m in xrange(16):
            ds = afwDet.DiaSource()
            ds.setId(m + 1)
            self.container1[m] = ds
            
            ds = afwDet.DiaSource()
            ds.setId(m)
            ds.setRa(m*20)
            self.container2.push_back(ds)

        self.dsv1 = afwDet.PersistableDiaSourceVector(self.container1)
        self.dsv2 = afwDet.PersistableDiaSourceVector(self.container2)

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

        ds = afwDet.DiaSource()        
        if dsv1Copy.size() == 0:
            dsv1Copy.append(ds)
        else:
            dsv1Copy.pop()
        dsv2Copy.append(ds)
        
        assert dsv1Copy.size() != self.container1.size()
        assert dsv2Copy.size() != self.container2.size()

    def testInsertErase(self):
        container = self.dsv1.getSources()
        
        front = container[:8]
        back = container[8:]

        copy = afwDet.DiaSourceSet()
        
        for i in xrange(front.size()):
            copy.append(front[i])
            
        ds = afwDet.DiaSource()
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
            pol.set("Formatter.PersistableDiaSourceVector.DiaSource.templateTableName", "DIASource")
            pol.set("Formatter.PersistableDiaSourceVector.DiaSource.tableNamePattern",
                    "_tmp_v%(visitId)_DiaSource")
            pers = dafPers.Persistence.getPersistence(pol)
            loc  = dafPers.LogicalLocation("mysql://lsst10.ncsa.uiuc.edu:3306/test_diasource")
            dp = dafBase.PropertySet()
            dp.setInt("visitId", int(time.clock())*16384 + random.randint(0, 16383))
            dp.setInt("sliceId", 0)
            dp.setInt("numSlices", 1)
            dp.setLongLong("ampExposureId", 10)
            dp.setString("itemName", "DiaSource")
            stl = dafPers.StorageList()
            stl.append(pers.getPersistStorage("DbStorage", loc))
            pers.persist(self.dsv1, stl, dp)
            stl = dafPers.StorageList()
            stl.append(pers.getRetrieveStorage("DbStorage", loc))
            persistable = pers.unsafeRetrieve("PersistableDiaSourceVector", stl, dp)
            res = afwDet.PersistableDiaSourceVector.swigConvert(persistable)
            afwDet.dropAllSliceTables(loc, pol.getPolicy("Formatter.PersistableDiaSourceVector"), dp)
            assert(res == self.dsv1)
        else:
            print "skipping database tests"

    def testNaNPersistence(self):
        dss = afwDet.DiaSourceSet()
        ds = afwDet.DiaSource()
        nan = float('nan')
        ds.setRa(nan)
        ds.setDec(nan)
        ds.setRaErrForDetection(nan)
        ds.setRaErrForWcs(nan)
        ds.setDecErrForDetection(nan)
        ds.setDecErrForWcs(nan)
        ds.setXAstrom(nan)
        ds.setXAstromErr(nan)
        ds.setYAstrom(nan)
        ds.setYAstromErr(nan)
        ds.setTaiMidPoint(nan)
        ds.setTaiRange(nan)
        ds.setPsfFlux(nan)
        ds.setPsfFluxErr(nan)
        ds.setApFlux(nan)
        ds.setApFluxErr(nan)
        ds.setModelFlux(nan)
        ds.setModelFluxErr(nan)
        ds.setInstFlux(nan)
        ds.setInstFluxErr(nan)
        ds.setApDia(nan)
        ds.setIxx(nan)
        ds.setIxxErr(nan)
        ds.setIyy(nan)
        ds.setIyyErr(nan)
        ds.setIxy(nan)
        ds.setIxyErr(nan)
        ds.setSnr(nan)
        ds.setChi2(nan)
        dss.append(ds)
        pdsv = afwDet.PersistableDiaSourceVector(dss)
        pol = dafPolicy.Policy()
        pers = dafPers.Persistence.getPersistence(pol)
        dp = dafBase.PropertySet()
        dp.setInt("visitId", int(time.clock())*16384 + random.randint(0, 16383))
        dp.setInt("sliceId", 0)
        dp.setInt("numSlices", 1)
        dp.setLongLong("ampExposureId", 10)
        dp.setString("itemName", "DiaSource")
        stl = dafPers.StorageList()
        f = tempfile.NamedTemporaryFile()
        try:
            loc  = dafPers.LogicalLocation(f.name)
            stl.append(pers.getPersistStorage("BoostStorage", loc))
            pers.persist(pdsv, stl, dp)
            stl = dafPers.StorageList()
            stl.append(pers.getRetrieveStorage("BoostStorage", loc))
            persistable = pers.unsafeRetrieve("PersistableDiaSourceVector", stl, dp)
            res = afwDet.PersistableDiaSourceVector.swigConvert(persistable)
            self.assertTrue(res == pdsv)
        except:
            f.close()
            raise

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(DiaSourceTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

if __name__ == "__main__":
    utilsTests.run(suite())

