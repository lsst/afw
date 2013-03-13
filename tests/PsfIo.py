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
import numpy
import unittest
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
    psf2 = afwDetect.DoubleGaussianPsf.swigConvert(psfptr)

    return psf2

class DoubleGaussianPsfTestCase(unittest.TestCase):
    """A test case for DoubleGaussianPsf"""

    def assertClose(self, a, b):
        self.assert_(numpy.allclose(a, b), "%s != %s" % (a, b))

    def comparePsfs(self, psf1, psf2):
        self.assert_(isinstance(psf1, afwDetect.DoubleGaussianPsf))
        self.assert_(isinstance(psf2, afwDetect.DoubleGaussianPsf))
        self.assertEqual(psf1.getKernel().getWidth(), psf2.getKernel().getWidth())
        self.assertEqual(psf1.getKernel().getHeight(), psf2.getKernel().getHeight())
        self.assertEqual(psf1.getSigma1(), psf2.getSigma1())
        self.assertEqual(psf1.getSigma2(), psf2.getSigma2())
        self.assertEqual(psf1.getB(), psf2.getB())
        
    def setUp(self):
        self.ksize = 25                      # size of desired kernel
        FWHM = 5
        self.sigma1 = FWHM/(2*numpy.sqrt(2*numpy.log(2)))
        self.sigma2 = 2*self.sigma1
        self.b = 0.1

    def testBoostPersistence(self):
        psf1 = afwDetect.DoubleGaussianPsf(self.ksize, self.ksize, self.sigma1, self.sigma2, self.b)
        psf2 = roundTripPsf(1, psf1)
        psf3 = roundTripPsf(1, psf1)
        self.comparePsfs(psf1, psf2)
        self.comparePsfs(psf1, psf3)

    def testFitsPersistence(self):
        psf1 = afwDetect.DoubleGaussianPsf(self.ksize, self.ksize, self.sigma1, self.sigma2, self.b)
        filename = "tests/data/psf1-1.fits"
        psf1.writeFits("tests/data/psf1-1.fits")
        psf2 = afwDetect.DoubleGaussianPsf.readFits("tests/data/psf1-1.fits")
        self.comparePsfs(psf1, psf2)

    def testArchiveImports(self):
        # This file was saved with a Psf defined in testTableArchivesLib, so we'll only be able
        # to load it if the module-importer mechanism works.
        filename = "tests/data/archiveImportTest.fits"
        exposure = afwImage.ExposureF(filename)
        self.assert_(exposure.getPsf() is not None)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(DoubleGaussianPsfTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(exit=False):
    """Run the utilsTests"""
    utilsTests.run(suite(), exit)

if __name__ == "__main__":
    run(True)
