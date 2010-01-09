#!/usr/bin/env python
import unittest
import lsst.utils.tests as tests
import lsst.afw.image as afwImage
import lsst.daf.base as dafBase

class TanSipTestCases(unittest.TestCase):
    """Tests for the existence of the bug reported in #835
       (Wcs class doesn't gracefully handle the case of ctypes
       having -SIP appended to them).
    """
    
    def setUp(self):
        #metadata taken from CFHT data
        #v695856-e0/v695856-e0-c000-a00.sci_img.fits

        metadata = dafBase.PropertySet()

        metadata.set("SIMPLE",                    "T") 
        metadata.set("BITPIX",                  -32) 
        metadata.set("NAXIS",                    2) 
        metadata.set("NAXIS1",                 1024) 
        metadata.set("NAXIS2",                 1153) 
        metadata.set("RADECSYS", 'FK5')
        metadata.set("EQUINOX",                2000.)


        metadata.setDouble("CRVAL1",     215.604025685476)
        metadata.setDouble("CRVAL2",     53.1595451514076)
        metadata.setDouble("CRPIX1",     1109.99981456774)
        metadata.setDouble("CRPIX2",     560.018167811613)
        metadata.set("CTYPE1", 'RA---TAN-SIP')
        metadata.set("CTYPE2", 'DEC--TAN-SIP')

        metadata.setDouble("CD1_1", 5.10808596133527E-05)
        metadata.setDouble("CD1_2", 1.85579539217196E-07)
        metadata.setDouble("CD2_2", -5.10281493481982E-05)
        metadata.setDouble("CD2_1", -8.27440751733828E-07)




        self.wcs = afwImage.Wcs(metadata)

    def tearDown(self):
        del self.wcs

    def evalTanSip(self, ra, dec, x, y):

        # xy to sky; this is known ahead of time for this unit test
        # 1 pixel offset seems to be necessary to match the known answer
        sky = self.wcs.xyToRaDec(x - 1, y - 1)
        self.assertAlmostEqual(sky[0], ra,  5) # 5th digit in degrees ~ 0.035 arcsec ~ 1/10 pixel
        self.assertAlmostEqual(sky[1], dec, 5) # 

        # round trip it
        xy  = self.wcs.raDecToXY(sky)
        self.assertAlmostEqual(xy[0], x - 1, 5)
        self.assertAlmostEqual(xy[1], y - 1, 5)

    def testTanSip0(self):
        """The origin of the Wcs solution"""
        
        y   = 560.018167811613
        x   = 1109.99981456774
        
        ra  = 215.604025685476
        dec = 53.1595451514076
        self.evalTanSip(ra, dec, x, y)
        
    def testTanSip1(self):
        x   = 110
        y   = 90
        ra  = 215.51863778475067
        dec = 53.18432622376551
        self.evalTanSip(ra, dec, x, y)

    def testTanSip2(self):
        #Position 920, 200
        x   = 920
        y   = 200
        ra  = 215.51863778475067
        dec = +53.1780639
        print "Skipping testTanSip2() with x=%d, y=%d, ra=%f, dec=%f" % (x, y, ra, dec)
        #self.evalTanSip(ra, dec, x, y)
        
    def testTanSip3(self):
        x   = 427
        y   = 1131
        ra  = 215.5460541111905
        dec = 53.130960072287699
        self.evalTanSip(ra, dec, x, y)
        

#####
        
def suite():
    """Returns a suite containing all the test cases in this module."""
    tests.init()

    suites = []
    suites += unittest.makeSuite(TanSipTestCases)
    suites += unittest.makeSuite(tests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit=False):
    """Run the tests"""
    tests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)
