#!/usr/bin/env python

import os, sys
from math import *
import unittest
import eups
import random
import lsst.utils.tests as utilsTests
import lsst.daf.base as dafBase
import lsst.daf.persistence as dafPersist
import lsst.pex.exceptions as pexExceptions
import lsst.pex.logging as logging
import lsst.pex.policy as policy
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.detection as afwDet
import lsst.afw.math as afwMath
import lsst.afw.display.ds9 as ds9
import lsst.afw.display.utils as displayUtils
import lsst

try:
    type(verbose)
except NameError:
    display = False
    verbose = 0

#logging.Trace_setVerbosity("afw.psf", verbose)

def roundTripSourceMatch(storagetype, filename, matchlist):
    pol = policy.Policy()
    additionalData = dafBase.PropertySet()

    loc = dafPersist.LogicalLocation(filename)
    persistence = dafPersist.Persistence.getPersistence(pol)
    storageList = dafPersist.StorageList()
    storage = persistence.getPersistStorage(storagetype, loc)
    storageList.append(storage)
    persistence.persist(matchlist, storageList, additionalData)

    storageList2 = dafPersist.StorageList()
    storage2 = persistence.getRetrieveStorage(storagetype, loc)
    storageList2.append(storage2)
    matchlistptr = persistence.unsafeRetrieve("PersistableSourceMatchVector", storageList2, additionalData)
    matchlist2 = afwDet.PersistableSourceMatchVector.swigConvert(matchlistptr)

    return matchlist2

class matchlistTestCase(unittest.TestCase):
    def setUp(self):
        self.smv = afwDet.SourceMatchVector()
        Nmatch = 20
        #self.refids = [long(random.random() * 2**64) for i in range(Nmatch)]
        #self.refids = [int(random.random() * 2**64) for i in range(Nmatch)]
        self.refids = [int(random.random() * 2**31) for i in range(Nmatch)]
        print 'refids:', self.refids
        for m in range(Nmatch):
            sm = afwDet.SourceMatch()
            s1 = afwDet.Source()
            s1.setSourceId(self.refids[m])
            sm.first = s1
            s2 = afwDet.Source()
            s2.setSourceId(m)
            sm.second = s2
            sm.distance = 0
            self.smv.push_back(sm)

        print 'Sent:', self.smv

        self.psmv = afwDet.PersistableSourceMatchVector(self.smv)
        self.matchlist = roundTripSourceMatch('FitsStorage', 'tests/data/matchlist.fits',
                                              self.psmv)
        print 'Got PSMV:', self.matchlist
        self.smv2 = self.matchlist.getSourceMatches()
        print 'Got SMV:', self.smv2
        self.assertEqual(len(self.smv2), len(self.smv))
        for i,m in enumerate(self.smv2):
            self.assertEqual(type(m), lsst.afw.detection.detectionLib.SourceMatch)
            self.assertEqual(m.first.getId(), self.refids[i])
            self.assertEqual(m.second.getId(), i)
            #print '  type', type(m)
            print '  first', m.first
            print '  second', m.second
            # Strangely, "print m" fails with "getX()" not found
            # (from match.i : SourceMatch.__str__() )
            #print '  dir(first)', dir(m.first)
            #print '  dir(second)', dir(m.second)
            #print '  type(first)', type(m.first)
            #print '  dir(m)', dir(m)
            #print '  m', m

    def tearDown(self):
        pass

    def testStuff(self):
        #    # self.assertEqual(stats.getValue(afwMath.MAX), 0.0)
        pass

            
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= silly boilerplate -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    utilsTests.init()
    suites = []
    suites += unittest.makeSuite(matchlistTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(exit=False):
    """Run the utilsTests"""
    utilsTests.run(suite(), exit)

if __name__ == "__main__":
    run(True)
