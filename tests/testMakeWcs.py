#pybind11##!/usr/bin/env python
#pybind11#from __future__ import absolute_import, division
#pybind11#from __future__ import print_function
#pybind11#from builtins import str
#pybind11#
#pybind11##
#pybind11## LSST Data Management System
#pybind11## Copyright 2008, 2009, 2010 LSST Corporation.
#pybind11##
#pybind11## This product includes software developed by the
#pybind11## LSST Project (http://www.lsst.org/).
#pybind11##
#pybind11## This program is free software: you can redistribute it and/or modify
#pybind11## it under the terms of the GNU General Public License as published by
#pybind11## the Free Software Foundation, either version 3 of the License, or
#pybind11## (at your option) any later version.
#pybind11##
#pybind11## This program is distributed in the hope that it will be useful,
#pybind11## but WITHOUT ANY WARRANTY; without even the implied warranty of
#pybind11## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#pybind11## GNU General Public License for more details.
#pybind11##
#pybind11## You should have received a copy of the LSST License Statement and
#pybind11## the GNU General Public License along with this program.  If not,
#pybind11## see <http://www.lsstcorp.org/LegalNotices/>.
#pybind11##
#pybind11#
#pybind11#import unittest
#pybind11#import lsst.utils.tests
#pybind11#
#pybind11#import lsst.afw.image as afwImage
#pybind11#import lsst.afw.geom as afwGeom
#pybind11#import lsst.afw.coord as afwCoord
#pybind11#import lsst.daf.base as dafBase
#pybind11#
#pybind11#import lsst
#pybind11#
#pybind11#try:
#pybind11#    type(verbose)
#pybind11#except NameError:
#pybind11#    verbose = 0
#pybind11#
#pybind11#
#pybind11#class MakeWcsTestCase(unittest.TestCase):
#pybind11#    """Test that makeWcs correctly returns a Wcs or TanWcs object
#pybind11#       as appropriate based on the contents of a fits header
#pybind11#    """
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        # metadata taken from CFHT data
#pybind11#        # v695856-e0/v695856-e0-c000-a00.sci_img.fits
#pybind11#
#pybind11#        self.metadata = dafBase.PropertySet()
#pybind11#
#pybind11#        self.metadata.set("SIMPLE", "T")
#pybind11#        self.metadata.set("BITPIX", -32)
#pybind11#        self.metadata.set("NAXIS", 2)
#pybind11#        self.metadata.set("NAXIS1", 1024)
#pybind11#        self.metadata.set("NAXIS2", 1153)
#pybind11#        self.metadata.set("RADECSYS", 'FK5')
#pybind11#        self.metadata.set("EQUINOX", 2000.)
#pybind11#
#pybind11#        self.metadata.setDouble("CRVAL1", 215.604025685476)
#pybind11#        self.metadata.setDouble("CRVAL2", 53.1595451514076)
#pybind11#        self.metadata.setDouble("CRPIX1", 1109.99981456774)
#pybind11#        self.metadata.setDouble("CRPIX2", 560.018167811613)
#pybind11#        self.metadata.set("CTYPE1", 'RA---SIN')
#pybind11#        self.metadata.set("CTYPE2", 'DEC--SIN')
#pybind11#
#pybind11#        self.metadata.setDouble("CD1_1", 5.10808596133527E-05)
#pybind11#        self.metadata.setDouble("CD1_2", 1.85579539217196E-07)
#pybind11#        self.metadata.setDouble("CD2_2", -5.10281493481982E-05)
#pybind11#        self.metadata.setDouble("CD2_1", -8.27440751733828E-07)
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        del self.metadata
#pybind11#
#pybind11#    def testCreateBaseWcs(self):
#pybind11#        """Check that a non-TAN projection in the header creates a base Wcs object"""
#pybind11#
#pybind11#        wcs = afwImage.makeWcs(self.metadata)
#pybind11#        strRepresentation = str(wcs)
#pybind11#        self.assertNotEqual(strRepresentation.find("image::Wcs"), -1, "non Wcs object returned")
#pybind11#
#pybind11#    def testNoCreateTanWcs(self):
#pybind11#        """Test than an exception is thrown if we try to upcast to a TanWcs inappropriately"""
#pybind11#        wcs = afwImage.makeWcs(self.metadata)
#pybind11#
#pybind11#        with self.assertRaises(lsst.pex.exceptions.Exception):
#pybind11#            afwImage.cast_TanWcs(wcs)
#pybind11#
#pybind11#    def testCreateTanWcs(self):
#pybind11#        """Check that a non-TAN projection in the header creates a base Wcs object"""
#pybind11#
#pybind11#        self.metadata.set("CTYPE1", "RA---TAN")
#pybind11#        self.metadata.set("CTYPE2", "DEC--TAN")
#pybind11#
#pybind11#        afwImage.makeWcs(self.metadata)
#pybind11#        wcs = afwImage.cast_TanWcs(afwImage.makeWcs(self.metadata))
#pybind11#        strRepresentation = str(wcs)
#pybind11#        self.assertNotEqual(strRepresentation.find("image::TanWcs"), -1, "non TanWcs object returned")
#pybind11#
#pybind11#    def testCreateTanSipWcs(self):
#pybind11#
#pybind11#        self.metadata.set("CTYPE1", "RA---TAN")
#pybind11#        self.metadata.set("CTYPE2", "DEC--TAN")
#pybind11#
#pybind11#        wcs = afwImage.cast_TanWcs(afwImage.makeWcs(self.metadata))
#pybind11#        strRepresentation = str(wcs)
#pybind11#        self.assertNotEqual(strRepresentation.find("image::TanWcs"), -1, "non TanWcs object returned")
#pybind11#
#pybind11#    def testPythonLevelMakeWcs(self):
#pybind11#        """Verify that we can make a Wcs by providing the CD matrix elements in python."""
#pybind11#
#pybind11#        m = self.metadata
#pybind11#        crval = afwCoord.makeCoord(afwCoord.ICRS, m.getDouble("CRVAL1") *
#pybind11#                                   afwGeom.degrees, m.getDouble("CRVAL2") * afwGeom.degrees)
#pybind11#        crpix = afwGeom.Point2D(m.getDouble("CRPIX1"), m.getDouble("CRPIX2"))
#pybind11#        cd11, cd12 = m.getDouble("CD1_1"), m.getDouble("CD1_2")
#pybind11#        cd21, cd22 = m.getDouble("CD2_1"), m.getDouble("CD2_2")
#pybind11#        print('CRVAL:', crval)
#pybind11#
#pybind11#        # this is defined at the c++ level in src/image/makeWcs.cc
#pybind11#        wcsMade = afwImage.makeWcs(crval, crpix, cd11, cd12, cd21, cd22)
#pybind11#
#pybind11#        # trivial test ... verify that we get back what we put in.
#pybind11#        for wcs in [wcsMade]:
#pybind11#            crvalTest = wcs.getSkyOrigin().getPosition(afwGeom.degrees)
#pybind11#            crpixTest = wcs.getPixelOrigin()
#pybind11#            CD = wcs.getCDMatrix()
#pybind11#
#pybind11#            self.assertAlmostEqual(crvalTest[0], crval.getLongitude().asDegrees())
#pybind11#            self.assertAlmostEqual(crvalTest[1], crval.getLatitude().asDegrees())
#pybind11#            self.assertAlmostEqual(crpixTest[0], crpix[0])
#pybind11#            self.assertAlmostEqual(crpixTest[1], crpix[1])
#pybind11#            self.assertAlmostEqual(CD[0, 0], cd11)
#pybind11#            self.assertAlmostEqual(CD[0, 1], cd12)
#pybind11#            self.assertAlmostEqual(CD[1, 0], cd21)
#pybind11#            self.assertAlmostEqual(CD[1, 1], cd22)
#pybind11#
#pybind11#    def testReadDESHeader(self):
#pybind11#        """Verify that we can read a DES header"""
#pybind11#        self.metadata.set("RADESYS", "FK5    ")  # note trailing white space
#pybind11#        self.metadata.set("CTYPE1", 'RA---TPV')
#pybind11#        self.metadata.set("CTYPE2", 'DEC--TPV')
#pybind11#
#pybind11#        afwImage.makeWcs(self.metadata)
#pybind11#
#pybind11#
#pybind11#class MemoryTester(lsst.utils.tests.MemoryTestCase):
#pybind11#    pass
#pybind11#
#pybind11#
#pybind11#def setup_module(module):
#pybind11#    lsst.utils.tests.init()
#pybind11#
#pybind11#
#pybind11#if __name__ == "__main__":
#pybind11#    lsst.utils.tests.init()
#pybind11#    unittest.main()
