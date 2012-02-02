#!/usr/bin/env python

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
import lsst.pex.policy as pexPolicy
import lsst.daf.persistence as dafPers
import lsst.utils.tests as utilsTests
import lsst.afw.detection as afwDet
import lsst.afw.geom as afwGeom

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
            ds.setRa(m*20 * afwGeom.degrees)
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
            pol  = pexPolicy.Policy()
            pol.set("Formatter.PersistableDiaSourceVector.DiaSource.templateTableName", "DIASource")
            pol.set("Formatter.PersistableDiaSourceVector.DiaSource.tableNamePattern",
                    "_tmp_v%(visitId)_DiaSource")
            pers = dafPers.Persistence.getPersistence(pol)
            loc  = dafPers.LogicalLocation("mysql://lsst10.ncsa.uiuc.edu:3306/test_diasource_v2")
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

    def testSpecialValuesPersistence(self):
        dss = afwDet.DiaSourceSet()
        ds = afwDet.DiaSource()
        for (vd, vf) in ((float('nan'), float('nan')),
                         (float('inf'), 0.0),
                         (float('-inf'), 0.0)):
            # we can't pass inf to methods taking floats - SWIG raises
            # an overflow error
            R = afwGeom.radians
            ds.setRa(vd * R)
            ds.setDec(vd * R)
            ds.setRaErrForDetection(vf * R)
            ds.setRaErrForWcs(vf * R)
            ds.setDecErrForDetection(vf * R)
            ds.setDecErrForWcs(vf * R)
            ds.setXAstrom(vd)
            ds.setXAstromErr(vf)
            ds.setYAstrom(vd)
            ds.setYAstromErr(vf)
            ds.setTaiMidPoint(vd)
            ds.setTaiRange(vd)
            ds.setPsfFlux(vd)
            ds.setPsfFluxErr(vf)
            ds.setApFlux(vd)
            ds.setApFluxErr(vf)
            ds.setModelFlux(vd)
            ds.setModelFluxErr(vf)
            ds.setInstFlux(vd)
            ds.setInstFluxErr(vf)
            ds.setApDia(vf)
            ds.setIxx(vf)
            ds.setIxxErr(vf)
            ds.setIyy(vf)
            ds.setIyyErr(vf)
            ds.setIxy(vf)
            ds.setIxyErr(vf)
            ds.setSnr(vf)
            ds.setChi2(vf)
            dss.append(ds)
            pdsv = afwDet.PersistableDiaSourceVector(dss)
            pol = pexPolicy.Policy()
            pers = dafPers.Persistence.getPersistence(pol)
            dp = dafBase.PropertySet()
            dp.setInt("visitId", 0)
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

