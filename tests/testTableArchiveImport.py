#!/usr/bin/env python2
from __future__ import absolute_import, division

# 
# LSST Data Management System
# Copyright 2008, 2009, 2010 LSST Corporation.
# 
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the LSST License Statement and 
# the GNU General Public License along with this program.  If not, 
# see <http://www.lsstcorp.org/LegalNotices/>.
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
