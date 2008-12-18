#!/usr/bin/env python
"""
Test warpExposure

Author: Nicole M. Silvestri, University of Washington
Contact: nms@astro.washington.edu
Created on: Thu Sep 20, 2007
"""

import os
import math
import pdb # we may want to say pdb.set_trace()
import unittest

import numpy

import eups
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.utils.tests as utilsTests
import lsst.pex.logging as logging

try:
    type(verbose)
except NameError:
    verbose = 0
    logging.Trace_setVerbosity("lsst.afw.math", verbose)

dataDir = eups.productDir("afwdata")
if not dataDir:
    raise RuntimeError("Must set up afwdata to run these tests")

InputOrigExposureName = "871034p_1_MI"
InputSciExposureName = "871034p_1_MI"
SwarpedExposureName = "small_MISwarp"
RemappedExposureName = "871034p_1_MIRemapped"

inFilePathOrig = os.path.join(dataDir, InputOrigExposureName)
inFilePathSci = os.path.join(dataDir, InputSciExposureName)
inFilePathSwarp = os.path.join(dataDir, SwarpedExposureName)
outFilePathRemap = os.path.join(dataDir, RemappedExposureName)
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class wcsMatchTestCase(unittest.TestCase):
    """
    A test case for warpExposure
    """

    def setUp(self):
        # setup original (template) MaskedImage and WCS
        self.origExposure = afwImage.ExposureD(inFilePathOrig)

        # setup the blank remapped Exposure;
        # make it  1/2 kernel width larger than the origExposure
        # and with WCS from 
        origMaskedImage = self.origExposure.getMaskedImage()
        origWidth = origMaskedImage.getWidth()
        origHeight = origMaskedImage.getHeight()
        remapMaskedImage = afwImage.MaskedImageD(origWidth + 1, origHeight + 1)
        sciExposure = afwImage.ExposureD(inFilePathSci)
        sciWcs = sciExposure.getWcs()
        self.remapExposure = afwImage.ExposureD(remapMaskedImage, sciWcs)
        self.remapWcs = self.remapExposure.getWcs()

        # input parameters to generate the analytic remapping kernel
        self.kernelType = "lanczos"
        self.kernelWidth = 2
        self.kernelHeight = 2
        self.threshold = 1

    def tearDown(self):
        del self.origExposure 
        del self.remapExposure
        del self.kernelType
        del self.kernelWidth
        del self.kernelHeight
        del self.threshold

    def testWcsMatchVoidEasy(self):
        """
        Test that warpExposure simply returns a remapped Exposure.
        
        In this case, since the input (original and science) MaskedImages are the same, the remaped Exposure
        should have the same WCS as the original Exposure.

        Check that the same pixel position on the original MaskedImage and the remapped MaskedImage yield
        the same RA/Decl on the sky. 
        """
        numGoodPix = afwMath.warpExposure(self.remapExposure, self.origExposure, self.kernelType,
            self.kernelWidth, self.kernelHeight)

        # try the origin
        origWcs = self.origExposure.getWcs()
        colRow = afwImage.Coord2D(0,0)
        origRaDec = origWcs.colRowToRaDec(colRow)
        remapWcs = self.remapExposure.getWcs()
        remapRaDec = remapWcs.colRowToRaDec(colRow)

        self.assertAlmostEqual(origRaDec.x(), remapRaDec.x())
        self.assertAlmostEqual(origRaDec.y(), remapRaDec.y())

        # try a random location
        colRow1 = afwImage.Coord2D(0,21)
        origRaDec1 = origWcs.colRowToRaDec(colRow1)
        remapRaDec1 = remapWcs.colRowToRaDec(colRow1)

        self.assertAlmostEqual(origRaDec1.x(), remapRaDec1.x())
        self.assertAlmostEqual(origRaDec1.y(), remapRaDec1.y())


    def xtestWcsMatchVoidSwarp(self):
        """
        Test that warpExposure returns a remapped Exposure that is equivalent to an Exposure that has been
        remapped using the SWARP routine frm Terapix.  In this case, the remaped Exposure should have
        the same WCS as the SWARPed Exposure.

        Check that the same pixel position on the SWARPed MaskedImage and the remapped MaskedImage
        yield the same RA/Decl on the sky.  
        """
        numGoodPix = afwMath.warpExposure(self.remapExposure, self.origExposure, self.kernelType,
            self.kernelWidth, self.kernelHeight)

        swarpExposure = afwImage.ExposureD(inFilePathSwarp)
        swarpWcs = swarpExposure.getWcs()

        remapWcs = self.remapExposure.getWcs()
        remapRaDec = remapWcs.colRowToRaDec(colRow)

        # try the origin
        colRow = afwImage.Coord2D(0,0)
        swarpRaDec = swarpWcs.colRowToRaDec(colRow)
        remapRaDec = remapWcs.colRowToRaDec(colRow)

        self.assertAlmostEqual(swarpRaDec.x(), remapRaDec.x())
        self.assertAlmostEqual(swarpRaDec.y(), remapRaDec.y())

        # try a random location
        colRow1 = afwImage.Coord2D(12,21)
        swarpRaDec1 = swarpWcs.colRowToRaDec(colRow1)
        remapRaDec1 = remapWcs.colRowToRaDec(colRow1)

        self.assertAlmostEqual(swarpRaDec1.x(), remapRaDec1.x())
        self.assertAlmostEqual(swarpRaDec1.y(), remapRaDec1.y())

                    
    def xtestWcsMatchOverloadEasy(self):
        """        
        Test that warpExposure overloaded simply returns a remapped Exposure.  In this case, since the input
        (original and science) MaskedImages are the same, the remaped Exposure should have the same WCS
        as the original Exposure.

        Check that the same pixel position on the original MaskedImage and the remapped MaskedImage yield
        the same RA/Decl on the sky. 
        """
        numGoodPix = afwMath.warpExposure(self.remapExposure, self.origExposure, self.kernelType,
            self.kernelWidth, self.kernelHeight)

        # try the origin
        origWcs = self.origExposure.getWcs()
        colRow = afwImage.Coord2D(0,0)
        origRaDec = origWcs.colRowToRaDec(colRow)
        newRemapWcs = self.remapExposure.getWcs()
        newRemapRaDec = newRemapWcs.colRowToRaDec(colRow)

        self.assertAlmostEqual(origRaDec.x(), newRemapRaDec.x())
        self.assertAlmostEqual(origRaDec.y(), newRemapRaDec.y())

        # try a random location
        colRow1 = afwImage.Coord2D(21,12)
        origRaDec1 = origWcs.colRowToRaDec(colRow1)
        newRemapRaDec1 = newRemapWcs.colRowToRaDec(colRow1)

        self.assertAlmostEqual(origRaDec1.x(), newRemapRaDec1.x())
        self.assertAlmostEqual(origRaDec1.y(), newRemapRaDec1.y())


    def xtestWcsMatchOverloadSwarp(self):
        """
        Test that warpExposure overload returns a remapped Exposure that is equivalent to an Exposure
        that has been remapped using the SWARP routine frm Terapix. In this case, the remaped Exposure
        should have the same WCS as the SWARPed Exposure.

        Check that the same pixel position on the SWARPed MaskedImage and the remapped MaskedImage yield
        the same RA/Decl on the sky.      
        """
        numGoodPix = afwMath.warpExposure(slew.remapExposure, self.origExposure, self.kernelType,
            self.kernelWidth, self.kernelHeight)

        swarpExposure = afwImage.ExposureD(inFilePathSwarp)
        swarpWcs = swarpExposure.getWcs()

        newRemapWcs = self.remapExposure.getWcs()
        newRemapRaDec = newRemapWcs.colRowToRaDec(colRow)

        # try the origin
        colRow = afwImage.Coord2D(0,0)
        swarpRaDec = swarpWcs.colRowToRaDec(colRow)
        newRemapRaDec = newRemapWcs.colRowToRaDec(colRow)

        self.assertAlmostEqual(swarpRaDec.x(), newRemapRaDec.x())
        self.assertAlmostEqual(swarpRaDec.y(), newRemapRaDec.y())

        # try a random location
        colRow1 = afwImage.Coord2D(12,21)
        swarpRaDec1 = swarpWcs.colRowToRaDec(colRow1)
        newRemapRaDec1 = newRemapWcs.colRowToRaDec(colRow1)

        self.assertAlmostEqual(swarpRaDec1.x(), newRemapRaDec1.x())
        self.assertAlmostEqual(swarpRaDec1.y(), newRemapRaDec1.y())
        
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """
    Returns a suite containing all the test cases in this module.
    """
    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(wcsMatchTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)

    return unittest.TestSuite(suites)

def run(exit=False):
    """Run the tests"""
    utilsTests.run(suite(), exit)

if __name__ == "__main__":
    run(True)
