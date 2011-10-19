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
import os

import lsst.daf.base as dafBase
import lsst.pex.policy as dafPolicy
import lsst.pex.policy as pexPolicy
import lsst.daf.persistence as dafPers
import lsst.utils.tests as utilsTests
import lsst.afw.detection as afwDet
import lsst.afw.coord as afwCoord
import lsst.afw.geom as afwGeom

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
            ds.setRa(m*20 * afwGeom.degrees)
            self.container2.push_back(ds)

        self.dsv1 = afwDet.PersistableSourceVector(self.container1)
        self.dsv2 = afwDet.PersistableSourceVector(self.container2)

    def tearDown(self):
        del self.dsv1
        del self.dsv2

    def testRaDecUnits(self):
        # LSST decrees radians.
        # In the setUp() above, the RAs set could reasonably be interpreted as radians.

        sources = self.dsv2.getSources()

        # check that getRa() returns in radians
        # MAGIC 0.349... = math.radians(20.)
        self.assertAlmostEqual(sources[1].getRa().asRadians(), 0.3490658503988659)

        # These tests don't make so much sense now that we use Angle!

        # check that setRaDec() getRaDec() round-trips.
        ra,dec = 100., 50.
        # makeCoord takes degrees.
        c1 = afwCoord.makeCoord(afwCoord.ICRS, ra * afwGeom.degrees, dec * afwGeom.degrees)
        # (test that by using degrees explicitly)
        c2 = afwCoord.makeCoord(afwCoord.ICRS, afwGeom.Point2D(ra, dec),
                                afwGeom.degrees)
        self.assertAlmostEqual(c1.toIcrs().getRa().asDegrees(), c2.toIcrs().getRa().asDegrees())
        self.assertAlmostEqual(c1.toIcrs().getDec().asDegrees(), c2.toIcrs().getDec().asDegrees())

        src = afwDet.Source()
        src.setRaDec(c1)
        # get it back in ICRS by default
        c1b = src.getRaDec()
        self.assertAlmostEqual(c1.toIcrs().getDec().asDegrees(), c1b.toIcrs().getDec().asDegrees())
        self.assertAlmostEqual(c1.toIcrs().getRa().asDegrees(),  c1b.toIcrs().getRa().asDegrees())

        self.assertAlmostEqual(src.getRa().asDegrees(), ra)
        self.assertAlmostEqual(src.getDec().asDegrees(), dec)

        src.setRa(math.pi * afwGeom.radians)
        src.setDec(math.pi / 4. * afwGeom.radians)
        c1c = src.getRaDec()
        self.assertAlmostEqual(c1c.getLongitude().asDegrees(), 180.)
        self.assertAlmostEqual(c1c.getLatitude().asDegrees(), 45.)

    def testBoostFilePersistence(self):
        pytype = "lsst.afw.detection.PersistableSourceVector"
        ctype = "PersistableSourceVector"

        # This is copied from Mapper, ButlerFactory, et al.
        # Is it just me, or is this just wack?
        from lsst.daf.persistence import Persistence, LogicalLocation, StorageList
        perPol = pexPolicy.Policy()
        per = Persistence.getPersistence(perPol)
        additionalData = dafBase.PropertySet()
        storageName = 'BoostStorage'
        f,loc = tempfile.mkstemp(suffix='.boost')
        os.close(f)
        print 'Writing to temp file', loc
        logLoc = LogicalLocation(loc, additionalData)
        storageList = StorageList()
        storage = per.getPersistStorage(storageName, logLoc)
        storageList.append(storage)

        obj = self.dsv2
        if hasattr(obj, '__deref__'):
            # We have a smart pointer, so dereference it.
            obj = obj.__deref__()
        # persist
        per.persist(obj, storageList, additionalData)

        # import this pythonType dynamically 
        pythonTypeTokenList = pytype.split('.')
        importClassString = pythonTypeTokenList.pop()
        importClassString = importClassString.strip()
        importPackage = ".".join(pythonTypeTokenList)
        importType = __import__(importPackage, globals(), locals(), \
                                [importClassString], -1) 
        pythonType = getattr(importType, importClassString)
        # unpersist
        additionalData = dafBase.PropertySet()
        logLoc = LogicalLocation(loc, additionalData)
        storageList = StorageList()
        storage = per.getRetrieveStorage(storageName, logLoc)
        storageList.append(storage)
        itemData = per.unsafeRetrieve(ctype, storageList, additionalData)
        finalItem = pythonType.swigConvert(itemData)
        sources = finalItem

        #print 'unpersisted sources:', sources
        obj1 = obj.getSources()
        obj2 = sources.getSources()

        self.assertEqual(len(obj1), len(obj2))
        self.assertEqual(len(obj2), 16)
        s1 = obj2[1]
        # check that RA came out in radians
        self.assertAlmostEqual(s1.getRa().asRadians(), 0.3490658503988659)
        # check that we can get it out in degrees
        self.assertAlmostEqual(s1.getRaDec().toIcrs().getRa().asDegrees(), 20.)


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
            loc  = dafPers.LogicalLocation("mysql://lsst10.ncsa.uiuc.edu:3306/test_source_v2")
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
            R = afwGeom.radians
            s.setRa(vd * R)
            s.setDec(vd * R)
            s.setRaErrForDetection(vf * R)
            s.setRaErrForWcs(vf * R)
            s.setDecErrForDetection(vf * R)
            s.setDecErrForWcs(vf * R)
            s.setXFlux(vd)
            s.setXFluxErr(vf)
            s.setYFlux(vd)
            s.setYFluxErr(vf)
            s.setRaFlux(vd * R)
            s.setRaFluxErr(vf * R)
            s.setDecFlux(vd * R)
            s.setDecFluxErr(vf * R)
            s.setXPeak(vd)
            s.setYPeak(vd)
            s.setRaPeak(vd * R)
            s.setDecPeak(vd * R)
            s.setXAstrom(vd)
            s.setXAstromErr(vf)
            s.setYAstrom(vd)
            s.setYAstromErr(vf)
            s.setRaAstrom(vd * R)
            s.setRaAstromErr(vf * R)
            s.setDecAstrom(vd * R)
            s.setDecAstromErr(vf * R)
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
            s.setRaObject(vd * R)
            s.setDecObject(vd * R)
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

    def testLongId(self):
        """Test that we can set an ID from a python long; #1714"""

        if False:                       # #1714 is not fixed (Source ctor takes int)
            s = afwDet.Source(2355928297481L)

        s = afwDet.Source()
        s.setId(2355928297481L)         # ... but we can set the ID

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

