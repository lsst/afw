#!/usr/bin/env python2
from __future__ import absolute_import, division

#
# LSST Data Management System
# See the COPYRIGHT and LICENSE files in the top-level directory of this
# package for notices and licensing terms.
#

import os
import unittest
import lsst.utils
import lsst.afw.image as afwImage
import lsst.afw.geom as afwGeom
import lsst.utils.tests as utilsTests
from math import sqrt

class WCSTestRaWrap(unittest.TestCase):
    '''A test set for the RA=0 wrap-around'''
    def setUp(self):
        mydir = lsst.utils.getPackageDir('afw')
        self.assertTrue(mydir is not None)
        self.datadir = os.path.join(mydir, 'tests')
        
    def test1(self):
        # This fails due to #1386
        #wcsfn = os.path.join(self.datadir, 'imsim-v85518312-fu-R43-S12.wcs')
        wcsfn = os.path.join(self.datadir, 'imsim-v85518312-fu-R43-S12.wcs2')
        hdr = afwImage.readMetadata(wcsfn)
        wcs1 = afwImage.makeWcs(hdr)

        crval = wcs1.getSkyOrigin()
        cd = wcs1.getCDMatrix()
        print cd
        crval_p = afwGeom.Point2D(crval.getLongitude().asDegrees(), 
                                 crval.getLatitude().asDegrees())
        origin = wcs1.getPixelOrigin()
        print crval_p
        print origin
        wcs2 = afwImage.Wcs(crval_p, origin, cd)

        for wcs in [wcs1,wcs2]:
            print wcs
            print 'x, y, RA, Dec, pixscale("/pix), pixscale2'
            for x,y in [(0,0),(300,0),(350,0),(360,0),(370,0),(380,0),(400,0)]:
                radec = wcs.pixelToSky(x,y)
                ra  = radec.getLongitude().asDegrees()
                dec = radec.getLatitude ().asDegrees()
                pixscale = 3600. * sqrt(wcs.pixArea(afwGeom.Point2D(x,y)))
                ps2 = wcs.pixelScale().asArcseconds()
                print x,y,ra,dec,pixscale,ps2
                self.assertTrue(abs(pixscale - 0.2) < 1e-3)
                self.assertTrue(abs(ps2 - 0.2) < 1e-3)



# Ridiculous boilerplate
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    utilsTests.init()
    suites = []
    suites += unittest.makeSuite(WCSTestRaWrap)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit=False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)

