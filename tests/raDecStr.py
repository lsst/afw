#!/usr/bin/env python
import os
import math
import pdb                          # we may want to say pdb.set_trace()
import unittest
import sys


import eups
import lsst.afw.image as afwImage
import lsst.afw.utils as utils
import lsst.utils.tests as utilsTests
import lsst.pex.exceptions.exceptionsLib as exceptions
import lsst

try:
    type(verbose)
except NameError:
    verbose = 0


class RaDecStrTestCase(unittest.TestCase):
    """A test case for routines converting ra and dec to and from strings"""
    
    posList=[ 
        #Trivial cases first
        (" 0:00:00.00", "+00:00:00.0", 000.000000, +0.0000000),
        (" 0:00:01.00", "+00:00:01.0", 000.004167, +0.0002778),
        (" 0:01:00.00", "+00:01:00.0", 000.250000, +0.0166667),
        (" 0:01:01.00", "+00:01:01.0", 000.254167, +0.0169444),
        (" 1:00:00.00", "+01:00:00.0", 015.000000, +1.0000000),
        (" 1:01:01.00", "+01:01:01.0", 015.254167, +1.0169444),
        (" 0:00:00.00", "-00:00:01.0", 000.000000, -0.0002778),
        (" 0:00:00.00", "-00:01:00.0", 000.000000, -0.0166667),
        (" 0:00:00.00", "-00:01:01.0", 000.000000, -0.0169444),
        (" 0:00:00.00", "-01:01:01.0", 000.000000, -1.0169444),
        
        #Some WDs
        (" 1:06:56.00", "-46:10:12.0",   016.733333, -46.1700000),
        (" 1:36:11.00", "-11:20:48.0",   024.045833, -11.3466667),
        (" 3:43:25.00", "-45:48:42.0",   055.854167, -45.8116667),
        (" 4:19:53.00", "+27:20:41.9",   064.970833, +27.3449722),
        (" 4:20:18.00", "+36:16:36.0",   065.075000, +36.2766667),
        (" 5:20:37.00", "+30:48:28.0",   080.154167, +30.8077778),
        (" 9:01:49.00", "+36:07:11.9",   135.454167, 36.11997222),
        (" 9:24:16.00", "+35:16:51.9",   141.066667, +35.28108333),
        (" 9:53:45.04", "+12:58:30.1",   148.437667, +12.9750278),
        (" 9:57:49.00", "+34:59:41.9",   149.454167, 34.994972222),
        ("12:01:46.00", "-03:45:38.9",   180.441667, -3.760805555),
        ("12:38:52.00", "-49:49:30.0",   189.716667, -49.8250000),
        ("13:09:58.00", "+35:09:29.9",   197.491667, +35.1583055),
        ("13:53:10.00", "+48:40:23.9",   208.291667, 48.67330555),
        ("13:52:12.00", "+65:22:00.0",   208.050000, +65.3666667),
        ("13:53:09.00", "+48:40:20.9",   208.287500, 48.67247222),
        ("14:03:57.00", "+00:00:00.0",   210.987500, +0.0000000),
        ("14:24:39.00", "+09:17:12.0",   216.162500, +9.2866667),
        ("14:32:18.00", "-81:20:05.9",   218.075000, -81.334972),
        ("16:01:23.00", "+36:48:34.9",  240.345833,   36.80969444),
        ("16:47:19.02", "+32:28:31.9",   251.829250, +32.4755278),
        ("16:48:25.00", "+59:03:36.0",   252.104167, +59.0600000),
        ("19:37:13.00", "+27:43:41.9",   294.304167, 27.72830555),
        ("19:52:29.00", "+25:09:18.0",   298.120833, +25.1550000),
        ("22:56:45.80", "+12:52:50.5",   344.190833, 12.88069444),
        ("23:28:47.62", "+05:14:54.2",   352.198417, +5.248388888)
    ]  

    #Irregular format strings, that should still parse
    irreg= [
        ("0 12 19", "14 16 23", 3.079166666, 14.27305555),
        ("0 45 26", "-6 09 08.12345", 11.358333333, -6.15225651)
    ]        

    def testStrToRa(self):
        for raStr, decStr, ra, dec in self.posList:
            result = utils.strToRa(raStr)
            #print "%s => %.7f =? %.7f" % (raStr, result, ra)
            self.assertAlmostEqual(result, ra, 6)

    def testStrToDec(self):
        for raStr, decStr, ra, dec in self.posList:
            result = utils.strToDec(decStr)
            #print "%s => %.7f =? %.7f" % (decStr, result, dec)
            self.assertAlmostEqual(result, dec, 6)

    def testRaToStr(self):
        for raStr, decStr, ra, dec in self.posList:
            result = utils.raToStr(ra)
            #print "%.7f => %s =? %s" % (ra, result, raStr)
            self.assertEqual(result, raStr)

    def testDecToStr(self):
        for raStr, decStr, ra, dec in self.posList:
            result = utils.decToStr(dec)
            #print "%.7f => %s =? %s" % (dec, result, decStr)
            self.assertEqual(result, decStr)
    
    def testRaDecToStr(self):
        for raStr, decStr, ra, dec in self.posList:
            correctStr = "%s %s" %(raStr, decStr)
            
            calc = utils.raDecToStr(ra, dec)
            #print "%.7f => %s =? %s" % (dec, result, decStr)
            self.assertEqual(correctStr, calc)
            
            #This function isn't working yet
            p = afwImage.PointD(ra, dec)
            help(p)
            print type(p)
            
            calc = utils.raDecToStr(p)
            self.assertEqual(correctStr, calc)

    def testIrregularInput(self):
        """Ensure that some irregularly formated inputs are correctly parsed"""
        for raStr, decStr, ra, dec in self.irreg:
            raRes = utils.strToRa(raStr)
            decRes= utils.strToDec(decStr)

            self.assertAlmostEqual(raRes, ra, 6)
            self.assertAlmostEqual(decRes, dec, 6)
            
    def testStringSeparator(self):
        """Strings separated by something other than a colon or a space
        should be parsed correctly"""
        raStr = "12x01x46.00"
        decStr= "-03x45x38.9"
        ra= 180.441667
        dec= -3.760805555
        
        resRa = utils.strToRa(raStr, sep="x")
        resDec = utils.strToDec(decStr, sep="x")

        self.assertAlmostEqual(raRes, ra, 6)
        self.assertAlmostEqual(decRes, dec, 6)
        

    #Currently commented out because exceptions are causing assertion failures
    #def testFail(self):
        #"""Test that functions fail on bad input"""
        #
        #str="three text strings"
        #self.assertRaises(exceptions.RuntimeErrorException, utils.strToRa(str))
        #self.assertRaises(exceptions.RuntimeErrorException, utils.strToDec(str))            
               
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-



def suite():
    """Returns a suite containing all the test cases in this module."""
    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(RaDecStrTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)

    return unittest.TestSuite(suites)

def run(exit=False):
    """Run the tests"""
    utilsTests.run(suite(), exit)

if __name__ == "__main__":
    run(True)
