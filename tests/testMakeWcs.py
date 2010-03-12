#!/usr/bin/env python
import os
import math
import pdb                          # we may want to say pdb.set_trace()
import unittest

import eups
import lsst.afw.image as afwImage
import lsst.afw.geom as afwGeom
import lsst.utils.tests as utilsTests
import lsst.afw.display.ds9 as ds9
import lsst.pex.exceptions.exceptionsLib as exceptions
import lsst.daf.base as dafBase

import lsst

try:
    type(verbose)
except NameError:
    verbose = 0



class MakeWcsTestCase(unittest.TestCase):
    """Test that makeWcs correctly returns a Wcs or TanWcs object
       as appropriate based on the contents of a fits header
    """
    
    def setUp(self):
        #metadata taken from CFHT data
        #v695856-e0/v695856-e0-c000-a00.sci_img.fits

        self.metadata = dafBase.PropertySet()

        self.metadata.set("SIMPLE",                    "T") 
        self.metadata.set("BITPIX",                  -32) 
        self.metadata.set("NAXIS",                    2) 
        self.metadata.set("NAXIS1",                 1024) 
        self.metadata.set("NAXIS2",                 1153) 
        self.metadata.set("RADECSYS", 'FK5')
        self.metadata.set("EQUINOX",                2000.)


        self.metadata.setDouble("CRVAL1",     215.604025685476)
        self.metadata.setDouble("CRVAL2",     53.1595451514076)
        self.metadata.setDouble("CRPIX1",     1109.99981456774)
        self.metadata.setDouble("CRPIX2",     560.018167811613)
        self.metadata.set("CTYPE1", 'RA---SIN')
        self.metadata.set("CTYPE2", 'DEC--SIN')

        self.metadata.setDouble("CD1_1", 5.10808596133527E-05)
        self.metadata.setDouble("CD1_2", 1.85579539217196E-07)
        self.metadata.setDouble("CD2_2", -5.10281493481982E-05)
        self.metadata.setDouble("CD2_1", -8.27440751733828E-07)

    def tearDown(self):
        del self.metadata
        
    def testCreateBaseWcs(self):
        """Check that a non-TAN projection in the header creates a base Wcs object"""
        
        wcs = afwImage.makeWcs(self.metadata)
        strRepresentation = str(wcs)
        self.assertNotEqual( strRepresentation.find("image::Wcs"), -1, "non Wcs object returned")

    def testCreateTanWcs(self):
        """Check that a non-TAN projection in the header creates a base Wcs object"""
        
        self.metadata.set("CTYPE1", "RA---TAN")
        self.metadata.set("CTYPE2", "DEC--TAN")
        
        wcs0 = afwImage.makeWcs(self.metadata)
        wcs = afwImage.cast_TanWcs(afwImage.makeWcs(self.metadata))
        strRepresentation = str(wcs)
        self.assertNotEqual( strRepresentation.find("image::TanWcs"), -1, "non TanWcs object returned")

    def testCreateTanSipWcs(self):
        """Check that a non-TAN projection in the header creates a base Wcs object"""
        
        self.metadata.set("CTYPE1", "RA---TAN-SIP")
        self.metadata.set("CTYPE2", "DEC--TAN-SIP")
        
        wcs = afwImage.cast_TanWcs(afwImage.makeWcs(self.metadata))
        strRepresentation = str(wcs)
        print strRepresentation
        self.assertNotEqual( strRepresentation.find("image::TanWcs"), -1, "non TanWcs object returned")


#####

def suite():
    """Returns a suite containing all the test cases in this module."""
    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(MakeWcsTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit=False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)
