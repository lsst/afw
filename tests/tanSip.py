#!/usr/bin/env python
import unittest
import lsst.utils.tests as tests
import lsst.afw.image as afwImage
import lsst.daf.base as dafBase

class TanSipTestCases(unittest.TestCase):
    
    def setUp(self):
        # known TAN-SIP map
        metadata = dafBase.PropertySet()
        metadata.set('CTYPE1', 'RA---TAN-SIP')
        metadata.set('CRPIX1',  0.0)
        metadata.setDouble('CRVAL1',  22.1136691404219)
        metadata.set('CTYPE2', 'DEC--TAN-SIP')
        metadata.set('CRPIX2',  0.0)
        metadata.setDouble('CRVAL2', -0.1147084491242522)
        metadata.setDouble('CD1_1',  -5.557012868767685E-05)
        metadata.setDouble('CD1_2',   8.612902244741679E-09)
        metadata.setDouble('CD2_1',   9.142016644499551E-09)
        metadata.setDouble('CD2_2',   5.556972887000046E-05)
        metadata.set('RADESYS', 'FK5')
        metadata.set('EQUINOX',  2000.0)
        metadata.set('A_ORDER',  2)
        metadata.setDouble('A_0_2',  1.212767765974218E-09)
        metadata.setDouble('A_1_1', -1.513523887718658E-09)
        metadata.setDouble('A_2_0', -1.853226276965959E-09)
        metadata.set('B_ORDER',  2)
        metadata.setDouble('B_0_2',  1.396136742490582E-10)
        metadata.setDouble('B_1_1', -2.017605162108109E-09)
        metadata.setDouble('B_2_0', -1.718567550120541E-09)
        metadata.setDouble('CDELT1',  5.555555555555556E-05)
        metadata.setDouble('CDELT2',  5.555555555555556E-05)
        metadata.set('AP_ORDER',  2)
        metadata.setDouble('AP_0_2', -1.21284736702594E-09)
        metadata.setDouble('AP_1_1',  1.513327296944833E-09)
        metadata.setDouble('AP_2_0',  1.853061707063949E-09)
        metadata.set('BP_ORDER',  2)
        metadata.setDouble('BP_0_2', -1.396563965515465E-10)
        metadata.setDouble('BP_1_1',  2.017652363438723E-09)
        metadata.setDouble('BP_2_0',  1.71862166205852E-09)

        # Use this code to replace TAN-SIP with plain TAN to ensure that there
        # is a difference.
        if False:
            metadata.set('CTYPE1', 'RA---TAN')
            metadata.set('CTYPE2', 'DEC--TAN')
            metadata.remove('A_ORDER')
            metadata.remove('A_0_2')
            metadata.remove('A_1_1')
            metadata.remove('A_2_0')
            metadata.remove('B_ORDER')
            metadata.remove('B_0_2')
            metadata.remove('B_1_1')
            metadata.remove('B_2_0')
            metadata.remove('AP_ORDER')
            metadata.remove('AP_0_2')
            metadata.remove('AP_1_1')
            metadata.remove('AP_2_0')
            metadata.remove('BP_ORDER')
            metadata.remove('BP_0_2')
            metadata.remove('BP_1_1')
            metadata.remove('BP_2_0')

        self.wcs = afwImage.Wcs(metadata)

        # known evaluation of this particular TAN-SIP model
        #
        # at pixel coordinate 0,0
        self.ra_0_0  = 22.11367
        self.dec_0_0 = -0.11471
        # at pixel coordinate 100,0
        self.ra_100_0  = 22.10811
        self.dec_100_0 = -0.11471
        # at pixel coordinate 0,100
        self.ra_0_100  = 22.11367
        self.dec_0_100 = -0.10915
        # at pixel coordinate 100,100
        self.ra_100_100  = 22.10811
        self.dec_100_100 = -0.10915
        
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

    def testTanSip1(self):
        x   = 0
        y   = 0
        ra  = self.ra_0_0
        dec = self.dec_0_0
        self.evalTanSip(ra, dec, x, y)

    def testTanSip2(self):
        x   = 100
        y   = 0
        ra  = self.ra_100_0
        dec = self.dec_100_0
        self.evalTanSip(ra, dec, x, y)
        
    def testTanSip3(self):
        x   = 0
        y   = 100
        ra  = self.ra_0_100
        dec = self.dec_0_100
        self.evalTanSip(ra, dec, x, y)
        
    def testTanSip4(self):
        x   = 100
        y   = 100
        ra  = self.ra_100_100
        dec = self.dec_100_100
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
