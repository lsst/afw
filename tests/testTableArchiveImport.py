#!/usr/bin/env python2
from __future__ import absolute_import, division

#
# LSST Data Management System
# See the COPYRIGHT and LICENSE files in the top-level directory of this
# package for notices and licensing terms.
#

import unittest

import lsst.utils.tests as utilsTests
import lsst.afw.image as afwImage

class ArchiveImportTestCase(unittest.TestCase):

    def testArchiveImports(self):
        # This file was saved with a Psf defined in testTableArchivesLib, so we'll only be able
        # to load it if the module-importer mechanism works.
        filename = "tests/data/archiveImportTest.fits"
        exposure = afwImage.ExposureF(filename)
        self.assert_(exposure.getPsf() is not None)

def suite():
    """Returns a suite containing all the test cases in this module."""
    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(ArchiveImportTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(exit=False):
    """Run the utilsTests"""
    utilsTests.run(suite(), exit)

if __name__ == "__main__":
    run(True)
