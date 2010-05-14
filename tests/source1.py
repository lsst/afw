#!/usr/bin/env python
"""
Tests for C++ Source and SourceVector Python wrappers (including persistence)

Run with:
   python Source_1.py
or
   python
   >>> import unittest; T=load("Source_1"); unittest.TextTestRunner(verbosity=1).run(T.suite())
"""

import unittest
import math
import random
import tempfile
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
            ds.setRa(math.radians(m*20))
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
            pol.set("Formatter.PersistableSourceVector.Source.tableNamePattern", "_tmp_v%(visitId)_Source")
            pers = dafPers.Persistence.getPersistence(pol)
            loc  = dafPers.LogicalLocation("mysql://lsst10.ncsa.uiuc.edu:3306/test_source_pt1")
            dp = dafBase.PropertySet()
            dp.setInt("visitId", int(time.clock())*16384 + random.randint(0, 16383))
            dp.setInt("sliceId", 0)
            dp.setInt("numSlices", 1)
            dp.setLongLong("ampExposureId", 10)
            dp.setString("itemName", "Source")
            stl = dafPers.StorageList()
            stl.append(pers.getPersistStorage("DbStorage", loc))
            pers.persist(self.dsv1, stl, dp)
            stl = dafPers.StorageList()
            stl.append(pers.getRetrieveStorage("DbStorage", loc))
            persistable = pers.unsafeRetrieve("PersistableSourceVector", stl, dp)
            res = afwDet.PersistableSourceVector.swigConvert(persistable)
            afwDet.dropAllSliceTables(loc, pol.getPolicy("Formatter.PersistableSourceVector"), dp)
            assert(res == self.dsv1)
        else:
            print "skipping database tests"

    def testSpecialValuesPersistence(self):
        ss = afwDet.SourceSet()
        s = afwDet.Source()
        for (vd, vf) in ((float('nan'), float('nan')),
                         (float('inf'), 0.0),
                         (float('-inf'), 0.0)):
            # we can't pass inf to methods taking floats - SWIG raises
            # an overflow error
            s.setRa(vd)
            s.setDec(vd)
            s.setRaErrForDetection(vf)
            s.setRaErrForWcs(vf)
            s.setDecErrForDetection(vf)
            s.setDecErrForWcs(vf)
            s.setXFlux(vd)
            s.setXFluxErr(vf)
            s.setYFlux(vd)
            s.setYFluxErr(vf)
            s.setRaFlux(vd)
            s.setRaFluxErr(vf)
            s.setDecFlux(vd)
            s.setDecFluxErr(vf)
            s.setXPeak(vd)
            s.setYPeak(vd)
            s.setRaPeak(vd)
            s.setDecPeak(vd)
            s.setXAstrom(vd)
            s.setXAstromErr(vf)
            s.setYAstrom(vd)
            s.setYAstromErr(vf)
            s.setRaAstrom(vd)
            s.setRaAstromErr(vf)
            s.setDecAstrom(vd)
            s.setDecAstromErr(vf)
            s.setTaiMidPoint(vd)
            s.setTaiRange(vd)
            s.setPsfFlux(vd)
            s.setPsfFluxErr(vf)
            s.setApFlux(vd)
            s.setApFluxErr(vf)
            s.setModelFlux(vd)
            s.setModelFluxErr(vf)
            s.setPetroFlux(vd)
            s.setPetroFluxErr(vf)
            s.setInstFlux(vd)
            s.setInstFluxErr(vf)
            s.setNonGrayCorrFlux(vd)
            s.setNonGrayCorrFluxErr(vf)
            s.setAtmCorrFlux(vd)
            s.setAtmCorrFluxErr(vf)
            s.setApDia(vf)
            s.setSnr(vf)
            s.setChi2(vf)
            s.setSky(vf)
            s.setSkyErr(vf)
            s.setRaObject(vd)
            s.setDecObject(vd)
            ss.append(s)
            psv = afwDet.PersistableSourceVector(ss) 
            pol = dafPolicy.Policy()
            pers = dafPers.Persistence.getPersistence(pol)
            dp = dafBase.PropertySet()
            dp.setInt("visitId", 0)
            dp.setInt("sliceId", 0)
            dp.setInt("numSlices", 1)
            dp.setLongLong("ampExposureId", 10)
            dp.setString("itemName", "Source")
            stl = dafPers.StorageList()
            f = tempfile.NamedTemporaryFile()
            try:
                loc  = dafPers.LogicalLocation(f.name)
                stl.append(pers.getPersistStorage("BoostStorage", loc))
                pers.persist(psv, stl, dp)
                stl = dafPers.StorageList()
                stl.append(pers.getRetrieveStorage("BoostStorage", loc))
                persistable = pers.unsafeRetrieve("PersistableSourceVector", stl, dp)
                res = afwDet.PersistableSourceVector.swigConvert(persistable)
                self.assertTrue(res == psv)
            except:
                f.close()
                raise

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

