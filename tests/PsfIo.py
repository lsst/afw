#!/usr/bin/env python
"""
Tests for bad pixel interpolation code

Run with:
   python PsfIo.py
or
   python
   >>> import PsfIo; PsfIo.run()
"""

import os, sys
from math import *
import unittest
import eups
import lsst.utils.tests as utilsTests
import lsst.daf.base as dafBase
import lsst.daf.persistence as dafPersist
import lsst.pex.exceptions as pexExceptions
import lsst.pex.logging as logging
import lsst.pex.policy as policy
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.detection as afwDetect
import lsst.afw.math as afwMath
import lsst.afw.display.ds9 as ds9
import lsst.afw.display.utils as displayUtils

try:
    type(verbose)
except NameError:
    display = False
    verbose = 0

logging.Trace_setVerbosity("afw.psf", verbose)

psfFileNum = 1
def roundTripPsf(key, psf):
    global psfFileNum
    pol = policy.Policy()
    additionalData = dafBase.PropertySet()

    if psfFileNum % 2 == 1:
        storageType = "Boost"
    else:
        storageType = "Xml"
    loc = dafPersist.LogicalLocation(
            "tests/data/psf%d-%d.%s" % (psfFileNum, key, storageType))
    psfFileNum += 1
    persistence = dafPersist.Persistence.getPersistence(pol)

    storageList = dafPersist.StorageList()
    storage = persistence.getPersistStorage("%sStorage" % (storageType), loc)
    storageList.append(storage)
    persistence.persist(psf, storageList, additionalData)

    storageList2 = dafPersist.StorageList()
    storage2 = persistence.getRetrieveStorage("%sStorage" % (storageType), loc)
    storageList2.append(storage2)
    psfptr = persistence.unsafeRetrieve("Psf", storageList2, additionalData)
    psf2 = afwDetect.Psf.swigConvert(psfptr)

    return psf2

class dgPsfTestCase(unittest.TestCase):
    """A test case for dgPSFs"""
    def setUp(self):
        self.ksize = 25                      # size of desired kernel
        FWHM = 5
        self.sigma1 = FWHM/(2*sqrt(2*log(2)))
        self.sigma2 = 2*self.sigma1
        self.b = 0.1
        self.psf = roundTripPsf(1,
                                afwDetect.createPsf("DoubleGaussian",
                                                    self.ksize, self.ksize, self.sigma1, self.sigma2, self.b))

    def tearDown(self):
        del self.psf

    def testKernel(self):
        """Test the creation of the Psf's kernel"""

        kIm = self.psf.computeImage()

        if False:
            ds9.mtv(kIm)        

        self.assertTrue(kIm.getWidth() == self.ksize)
        #
        # Check that the image is as expected
        #
        dgPsf = afwDetect.createPsf("DoubleGaussian",
                                    self.ksize, self.ksize, self.sigma1, self.sigma2, self.b)
        dgIm = dgPsf.computeImage()
        #
        # Check that they're the same
        #
        diff = type(kIm)(kIm, True); diff -= dgIm
        stats = afwMath.makeStatistics(diff, afwMath.MAX | afwMath.MIN)
        self.assertEqual(stats.getValue(afwMath.MAX), 0.0)
        self.assertEqual(stats.getValue(afwMath.MIN), 0.0)

    def testComputeImage(self):
        """Test returning a realisation of the Psf"""

        xcen = self.psf.getKernel().getWidth()//2
        ycen = self.psf.getKernel().getHeight()//2

        stamps = []
        trueCenters = []
        for x, y in ([10, 10], [9.4999, 10.4999], [10.5001, 10.5001]):
            fx, fy = x - int(x), y - int(y)
            if fx >= 0.5:
                fx -= 1.0
            if fy >= 0.5:
                fy -= 1.0

            im = self.psf.computeImage(afwGeom.makePointD(x, y)).convertFloat()

            stamps.append(im.Factory(im, True))
            trueCenters.append([xcen + fx, ycen + fy])
            
        if display:
            mos = displayUtils.Mosaic()     # control mosaics
            ds9.mtv(mos.makeMosaic(stamps))

            for i in range(len(trueCenters)):
                bbox = mos.getBBox(i)

                ds9.dot("+",
                        bbox.getX0() + xcen, bbox.getY0() + ycen, ctype=ds9.RED, size=1)
                ds9.dot("+",
                        bbox.getX0() + trueCenters[i][0], bbox.getY0() + trueCenters[i][1])

                ds9.dot("%.2f, %.2f" % (trueCenters[i][0], trueCenters[i][1]),
                        bbox.getX0() + xcen, bbox.getY0() + 2)
            
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(dgPsfTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(exit=False):
    """Run the utilsTests"""
    utilsTests.run(suite(), exit)

if __name__ == "__main__":
    run(True)
