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
import lsst.afw.table as afwTable
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
        pass
    def tearDown(self):
        pass

    def testRoundTrip(self):
        self.smv = afwDet.SourceMatchVector()
        Nmatch = 20
        #self.refids = [long(random.random() * 2**64) for i in range(Nmatch)]
        #self.refids = [int(random.random() * 2**64) for i in range(Nmatch)]
        self.refids = [int(random.random() * 2**31) for i in range(Nmatch)]
        print 'refids:', self.refids
        table = afwTable.SourceTable.make(afwTable.SourceTable.makeMinimalSchema())
        for m in range(Nmatch):
            sm = afwDet.SourceMatch()
            s1 = table.makeRecord()
            s1.setId(self.refids[m])
            sm.first = s1
            s2 = table.makeRecord()
            s2.setId(m)
            sm.second = s2
            sm.distance = 0
            self.smv.push_back(sm)

        print 'Sent:', self.smv

        self.psmv = afwDet.PersistableSourceMatchVector(self.smv)
        extra = dafBase.PropertyList()

        # as in meas_astrom : determineWcs.py
        #andata = os.environ.get('ASTROMETRY_NET_DATA_DIR')
        #if andata is None:
        #    extra.add('ANEUPS', 'none', 'ASTROMETRY_NET_DATA_DIR')
        #else:
        #    andata = os.path.basename(andata)
        #    extra.add('ANEUPS', andata, 'ASTROMETRY_NET_DATA_DIR')
        aneups = 'imsim-2010-11-09-0'
        ra,dec,rad = 2.389, 3.287, 0.158
        anindfn = '/home/dalang/lsst/astrometry_net_data/imsim-2010-11-09-0/index-101109003.fits'
        anindid = 101109003
        extra.add('ANEUPS', aneups)
        extra.add('RA', ra)
        extra.add('DEC', dec)
        extra.add('RAD', rad)
        extra.add('ANINDNM', anindfn)
        extra.add('ANINDID', anindid)

        # Sample:
        '''
        ANEUPS  = 'imsim-2010-11-09-0' / ASTROMETRY_NET_DATA_DIR
        RA      =     2.38923749033107 / field center in degrees
        DEC     =     3.28730056538314 / field center in degrees
        RADIUS  =    0.158420562703962 / field radius in degrees, approximate
        REFCAT  = 'none    '           / Reference catalog name
        REFCAMD5= 'none    '           / Reference catalog MD5 checksum
        ANINDID =            101109003 / Astrometry.net index id
        ANINDHP =                   -1 / Astrometry.net index HEALPix
        ANINDNM = '/home/dalang/lsst/astrometry_net_data/imsim-2010-11-09-0/index-1011&'
        CONTINUE  '09003.fits'         / Astrometry.net index name
        '''
        self.psmv.setSourceMatchMetadata(extra)

        self.matchlist = roundTripSourceMatch('FitsStorage', 'tests/data/matchlist.fits',
                                              self.psmv)
        print 'Got PSMV:', self.matchlist
        self.smv2 = self.matchlist.getSourceMatches()
        extra2 = self.matchlist.getSourceMatchMetadata()

        print 'Got SMV:', self.smv2
        print 'Got metadata:', extra2
        print '  ', extra2.toString()
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

        self.assertEqual(aneups, extra2.get('ANEUPS'))
        self.assertEqual(ra, extra2.get('RA'))
        self.assertEqual(dec, extra2.get('DEC'))
        self.assertEqual(rad, extra2.get('RAD'))
        self.assertEqual(anindfn, extra2.get('ANINDNM'))
        self.assertEqual(anindid, extra2.get('ANINDID'))

        del extra2
        del self.smv2
        del self.matchlist
        del self.psmv
        del extra
        del self.smv


            
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
