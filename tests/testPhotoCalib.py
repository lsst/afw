#
# LSST Data Management System
# Copyright 2008-2016 LSST Corporation.
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
from __future__ import absolute_import, division, print_function

import unittest

import os.path
import numpy as np

import lsst.utils.tests
import lsst.afw.geom
import lsst.afw.image
import lsst.afw.image.utils
import lsst.afw.math
import lsst.daf.base
import lsst.pex.exceptions


def computeMaggiesErr(instFluxErr, instFlux, instFlux0Err, instFlux0, flux):
    """Return the error on the flux (Maggies)."""
    return flux*np.sqrt((instFluxErr/instFlux)**2 + (instFlux0Err/instFlux0)**2)


def computeMagnitudeErr(instFluxErr, instFlux, instFlux0Err, instFlux0, flux):
    """Return the error on the magnitude."""
    return 2.5/np.log(10)*computeMaggiesErr(instFluxErr, instFlux, instFlux0Err, instFlux0, flux) / flux


class PhotoCalibTestCase(lsst.utils.tests.TestCase):

    def setUp(self):
        self.testDir = os.path.dirname(__file__)
        self.persistenceFile = "testPhotoCalib_testPersistence.fits"

        self.point0 = lsst.afw.geom.Point2D(0, 0)
        self.pointXShift = lsst.afw.geom.Point2D(-10, 0)
        self.pointYShift = lsst.afw.geom.Point2D(0, -10)
        self.bbox = lsst.afw.geom.Box2I(lsst.afw.geom.Point2I(-100, -100), lsst.afw.geom.Point2I(100, 100))

        self.instFlux0 = 1000.
        self.instFlux0Err = 10.
        self.instFlux = 1000.
        self.instFluxErr = 10.

        self.schema = lsst.afw.table.SourceTable.makeMinimalSchema()
        self.instFluxKeyName = "SomeFlux"
        lsst.afw.table.Point2DKey.addFields(self.schema, "centroid", "centroid", "pixels")
        self.instFluxKey = self.schema.addField(
            self.instFluxKeyName+"_instFlux", type="D", doc="post-ISR instFlux")
        self.instFluxErrKey = self.schema.addField(self.instFluxKeyName+"_instFluxErr", type="D",
                                                   doc="post-ISR instFlux stddev")
        self.maggiesKey = self.schema.addField(self.instFluxKeyName+"_flux", type="D", doc="maggies")
        self.maggiesErrKey = self.schema.addField(self.instFluxKeyName+"_fluxErr", type="D",
                                                  doc="maggies stddev")
        self.magnitudeKey = self.schema.addField(self.instFluxKeyName+"_mag", type="D", doc="magnitude")
        self.magnitudeErrKey = self.schema.addField(self.instFluxKeyName+"_magErr", type="D",
                                                    doc="magnitude stddev")
        self.table = lsst.afw.table.SourceTable.make(self.schema)
        self.table.defineCentroid('centroid')
        self.catalog = lsst.afw.table.SourceCatalog(self.table)
        record = self.catalog.addNew()
        record.set('id', 1)
        record.set('centroid_x', self.point0[0])
        record.set('centroid_y', self.point0[1])
        record.set(self.instFluxKeyName+'_instFlux', self.instFlux)
        record.set(self.instFluxKeyName+'_instFluxErr', self.instFluxErr)
        record = self.catalog.addNew()
        record.set('id', 2)
        record.set('centroid_x', self.pointYShift[0])
        record.set('centroid_y', self.pointYShift[1])
        record.set(self.instFluxKeyName+'_instFlux', self.instFlux*1e-9)
        record.set(self.instFluxKeyName+'_instFluxErr', self.instFluxErr)

        self.constantZeroPoint = lsst.afw.math.ChebyshevBoundedField(self.bbox, np.array([[self.instFlux0]]))
        self.linearXZeroPoint = lsst.afw.math.ChebyshevBoundedField(self.bbox,
                                                                    np.array([[self.instFlux0,
                                                                             self.instFlux0]]))

    def tearDown(self):
        del self.schema
        del self.table
        del self.catalog

    def _testPhotoCalibCenter(self, photoCalib, instFlux0Err):
        """
        Test conversions of instFlux for the mean and (0,0) value of a photoCalib.
        Assumes those are the same, e.g. that the non-constant terms are all
        odd, and that the mean of the calib is self.instFlux0.
        """
        # test that the constructor set the instFluxMag0 and err correctly
        self.assertEqual(self.instFlux0, photoCalib.getInstFluxMag0())
        self.assertEqual(instFlux0Err, photoCalib.getInstFluxMag0Err())

        # useful reference points: 1 nanomaggy == magnitude 22.5, 1 maggy = magnitude 0
        self.assertEqual(1, photoCalib.instFluxToMaggies(self.instFlux))
        self.assertEqual(0, photoCalib.instFluxToMagnitude(self.instFlux))

        self.assertFloatsAlmostEqual(1e-9, photoCalib.instFluxToMaggies(self.instFlux*1e-9))
        self.assertFloatsAlmostEqual(22.5, photoCalib.instFluxToMagnitude(self.instFlux*1e-9))
        # test that (0,0) gives the same result as above
        self.assertFloatsAlmostEqual(1e-9, photoCalib.instFluxToMaggies(self.instFlux*1e-9, self.point0))
        self.assertFloatsAlmostEqual(22.5, photoCalib.instFluxToMagnitude(self.instFlux*1e-9, self.point0))

        # test that we get a correct maggies err for the base instFlux
        errFlux = computeMaggiesErr(self.instFluxErr, self.instFlux, instFlux0Err, self.instFlux0, 1)
        result = photoCalib.instFluxToMaggies(self.instFlux, self.instFluxErr)
        self.assertEqual(1, result.value)
        self.assertFloatsAlmostEqual(errFlux, result.err)
        result = photoCalib.instFluxToMaggies(self.instFlux, self.instFluxErr, self.point0)
        self.assertFloatsAlmostEqual(1, result.value)
        self.assertFloatsAlmostEqual(errFlux, result.err)

        # test that we get a correct magnitude err for the base instFlux
        errMag = computeMagnitudeErr(self.instFluxErr, self.instFlux, instFlux0Err, self.instFlux0, 1)
        result = photoCalib.instFluxToMagnitude(self.instFlux, self.instFluxErr)
        self.assertEqual(0, result.value)
        self.assertFloatsAlmostEqual(errMag, result.err)
        result = photoCalib.instFluxToMagnitude(self.instFlux, self.instFluxErr, self.point0)
        self.assertFloatsAlmostEqual(0, result.value)
        self.assertFloatsAlmostEqual(errMag, result.err)

        # test that we get a correct maggies err for base instFlux*1e-9
        errFluxNano = computeMaggiesErr(self.instFluxErr, self.instFlux*1e-9,
                                        self.instFlux0Err, self.instFlux0, 1e-9)
        result = photoCalib.instFluxToMaggies(self.instFlux*1e-9, self.instFluxErr)
        self.assertFloatsAlmostEqual(1e-9, result.value)
        self.assertFloatsAlmostEqual(errFluxNano, result.err)
        result = photoCalib.instFluxToMaggies(self.instFlux*1e-9, self.instFluxErr, self.point0)
        self.assertFloatsAlmostEqual(1e-9, result.value)
        self.assertFloatsAlmostEqual(errFluxNano, result.err)

        # test that we get a correct magnitude err for base instFlux*1e-9
        errMagNano = computeMagnitudeErr(self.instFluxErr, self.instFlux*1e-9,
                                         instFlux0Err, self.instFlux0, 1e-9)
        result = photoCalib.instFluxToMagnitude(self.instFlux*1e-9, self.instFluxErr)
        self.assertFloatsAlmostEqual(22.5, result.value)
        self.assertFloatsAlmostEqual(errMagNano, result.err)
        result = photoCalib.instFluxToMagnitude(self.instFlux*1e-9, self.instFluxErr, self.point0)
        self.assertFloatsAlmostEqual(22.5, result.value)
        self.assertFloatsAlmostEqual(errMagNano, result.err)

        # test calculations on a single sourceRecord
        record = self.catalog[0]
        result = photoCalib.instFluxToMaggies(record, self.instFluxKeyName)
        self.assertEqual(1, result.value)
        self.assertFloatsAlmostEqual(errFlux, result.err)
        result = photoCalib.instFluxToMagnitude(record, self.instFluxKeyName)
        self.assertEqual(0, result.value)
        self.assertFloatsAlmostEqual(errMag, result.err)

        expectMaggies = np.array([[1, errFlux], [1e-9, errFluxNano]])
        expectMag = np.array([[0, errMag], [22.5, errMagNano]])
        self._testSourceCatalog(photoCalib, self.catalog, expectMaggies, expectMag)

        # test reverse conversion: magnitude to instFlux
        self.assertFloatsAlmostEqual(self.instFlux, photoCalib.magnitudeToInstFlux(0))
        self.assertFloatsAlmostEqual(self.instFlux*1e-9, photoCalib.magnitudeToInstFlux(22.5))

    def _testSourceCatalog(self, photoCalib, catalog, expectMaggies, expectMag):
        """Test passing in a sourceCatalog."""

        # test calculations on a sourceCatalog, returning the array
        result = photoCalib.instFluxToMaggies(catalog, self.instFluxKeyName)
        self.assertFloatsAlmostEqual(expectMaggies, result)
        result = photoCalib.instFluxToMagnitude(catalog, self.instFluxKeyName)
        self.assertFloatsAlmostEqual(expectMag, result)

        # test calculations on a sourceCatalog, setting values in the catalog
        # TODO: see RFC-322 and DM-10155
        RFC322_implemented = False
        if RFC322_implemented:
            photoCalib.instFluxToMaggiesAndMagnitude(self.catalog, self.instFluxKeyName, self.instFluxKeyName)
            self.assertFloatsAlmostEqual(self.catalog[self.instFluxKeyName+'flux'], expectMaggies[:, 0])
            self.assertFloatsAlmostEqual(self.catalog[self.instFluxKeyName+'fluxErr'], expectMaggies[:, 1])
            self.assertFloatsAlmostEqual(self.catalog[self.instFluxKeyName+'magnitude'], expectMag[:, 0])
            self.assertFloatsAlmostEqual(self.catalog[self.instFluxKeyName+'magnitudeErr'], expectMag[:, 1])

    def testNonVarying(self):
        """Tests a non-spatially-varying zeropoint."""
        photoCalib = lsst.afw.image.PhotoCalib(self.instFlux0)
        self._testPhotoCalibCenter(photoCalib, 0)

        self.assertEqual(1, photoCalib.instFluxToMaggies(self.instFlux, self.pointXShift))
        self.assertEqual(0, photoCalib.instFluxToMagnitude(self.instFlux, self.pointXShift))
        result = photoCalib.instFluxToMaggies(self.instFlux, self.instFluxErr)
        self.assertEqual(1, result.value)

        photoCalib = lsst.afw.image.PhotoCalib(self.instFlux0, self.instFlux0Err)
        self._testPhotoCalibCenter(photoCalib, self.instFlux0Err)

        # constant, with a bbox
        photoCalib = lsst.afw.image.PhotoCalib(self.instFlux0, bbox=self.bbox)
        self._testPhotoCalibCenter(photoCalib, 0)

    def testConstantBoundedField(self):
        """Test a spatially-constant bounded field."""
        photoCalib = lsst.afw.image.PhotoCalib(self.constantZeroPoint)
        self._testPhotoCalibCenter(photoCalib, 0)

        self.assertEqual(1, photoCalib.instFluxToMaggies(self.instFlux, self.pointYShift))
        self.assertEqual(0, photoCalib.instFluxToMagnitude(self.instFlux, self.pointYShift))
        self.assertFloatsAlmostEqual(1e-9, photoCalib.instFluxToMaggies(self.instFlux*1e-9, self.pointXShift))
        self.assertFloatsAlmostEqual(22.5, photoCalib.instFluxToMagnitude(
            self.instFlux*1e-9, self.pointXShift))

        photoCalib = lsst.afw.image.PhotoCalib(self.constantZeroPoint, self.instFlux0Err)
        self._testPhotoCalibCenter(photoCalib, self.instFlux0Err)

    def testLinearXBoundedField(self):
        photoCalib = lsst.afw.image.PhotoCalib(self.linearXZeroPoint)
        self._testPhotoCalibCenter(photoCalib, 0)

        self.assertEqual(1, photoCalib.instFluxToMaggies(self.instFlux, self.pointYShift))
        self.assertEqual(0, photoCalib.instFluxToMagnitude(self.instFlux, self.pointYShift))

        instFlux0 = (self.instFlux0 + self.pointXShift.getX()*self.instFlux0/(self.bbox.getWidth()/2.))
        expect = self.instFlux/instFlux0
        self.assertFloatsAlmostEqual(expect, photoCalib.instFluxToMaggies(self.instFlux, self.pointXShift))
        self.assertFloatsAlmostEqual(-2.5*np.log10(expect),
                                     photoCalib.instFluxToMagnitude(self.instFlux, self.pointXShift))

        self.assertFloatsAlmostEqual(expect*1e-9,
                                     photoCalib.instFluxToMaggies(self.instFlux*1e-9, self.pointXShift))
        self.assertFloatsAlmostEqual(-2.5*np.log10(expect*1e-9),
                                     photoCalib.instFluxToMagnitude(self.instFlux*1e-9, self.pointXShift))

        photoCalib = lsst.afw.image.PhotoCalib(self.linearXZeroPoint, self.instFlux0Err)
        self._testPhotoCalibCenter(photoCalib, self.instFlux0Err)

        # New catalog with a spatial component in the varying direction,
        # to ensure the calculations on a catalog properly handle non-constant BF.
        # NOTE: only the first quantity of the result (maggies or mags) should change.
        catalog = self.catalog.copy(deep=True)
        catalog[0].set('centroid_x', self.pointXShift[0])
        catalog[0].set('centroid_y', self.pointXShift[1])
        errFlux = computeMaggiesErr(self.instFluxErr, self.instFlux, self.instFlux0Err, instFlux0, expect)
        errMag = computeMagnitudeErr(self.instFluxErr, self.instFlux, self.instFlux0Err, instFlux0, expect)
        errFluxNano = computeMaggiesErr(self.instFluxErr, self.instFlux*1e-9,
                                        self.instFlux0Err, self.instFlux0, 1e-9)
        errMagNano = computeMagnitudeErr(self.instFluxErr, self.instFlux*1e-9,
                                         self.instFlux0Err, self.instFlux0, 1e-9)
        expectMaggies = np.array([[expect, errFlux], [1e-9, errFluxNano]])
        expectMag = np.array([[-2.5*np.log10(expect), errMag], [22.5, errMagNano]])
        self._testSourceCatalog(photoCalib, catalog, expectMaggies, expectMag)

    def testComputeScaledZeroPoint(self):
        photoCalib = lsst.afw.image.PhotoCalib(self.instFlux0, bbox=self.bbox)
        scaledCalib = lsst.afw.image.PhotoCalib(photoCalib.computeScaledZeroPoint())
        self.assertEqual(1, scaledCalib.instFluxToMaggies(self.instFlux)/photoCalib.getInstFluxMag0())
        self.assertEqual(photoCalib.instFluxToMaggies(self.instFlux),
                         scaledCalib.instFluxToMaggies(self.instFlux)/photoCalib.getInstFluxMag0())

        photoCalib = lsst.afw.image.PhotoCalib(self.constantZeroPoint)
        scaledCalib = lsst.afw.image.PhotoCalib(photoCalib.computeScaledZeroPoint())

        self.assertEqual(1, scaledCalib.instFluxToMaggies(self.instFlux/self.instFlux0))
        self.assertEqual(photoCalib.instFluxToMaggies(self.instFlux),
                         scaledCalib.instFluxToMaggies(self.instFlux)/photoCalib.getInstFluxMag0())

    @unittest.skip("Not yet implemented: see DM-10154")
    def testComputeScalingTo(self):
        photoCalib1 = lsst.afw.image.PhotoCalib(self.instFlux0, self.instFlux0Err, bbox=self.bbox)
        photoCalib2 = lsst.afw.image.PhotoCalib(self.instFlux0*500, self.instFlux0Err, bbox=self.bbox)
        scaling = photoCalib1.computeScalingTo(photoCalib2)(self.pointXShift)
        self.assertEqual(photoCalib1.instFluxToMaggies(self.instFlux, self.pointXShift)*scaling,
                         photoCalib2.instFluxToMaggies(self.instFlux, self.pointXShift))

        photoCalib3 = lsst.afw.image.PhotoCalib(self.constantZeroPoint, self.instFlux0Err)
        scaling = photoCalib1.computeScalingTo(photoCalib3)(self.pointXShift)
        self.assertEqual(photoCalib1.instFluxToMaggies(self.instFlux, self.pointXShift)*scaling,
                         photoCalib3.instFluxToMaggies(self.instFlux, self.pointXShift))
        scaling = photoCalib3.computeScalingTo(photoCalib1)(self.pointXShift)
        self.assertEqual(photoCalib3.instFluxToMaggies(self.instFlux, self.pointXShift)*scaling,
                         photoCalib1.instFluxToMaggies(self.instFlux, self.pointXShift))

        photoCalib4 = lsst.afw.image.PhotoCalib(self.linearXZeroPoint, self.instFlux0Err)
        scaling = photoCalib1.computeScalingTo(photoCalib4)(self.pointXShift)
        self.assertEqual(photoCalib1.instFluxToMaggies(self.instFlux, self.pointXShift)*scaling,
                         photoCalib4.instFluxToMaggies(self.instFlux, self.pointXShift))
        scaling = photoCalib4.computeScalingTo(photoCalib1)(self.pointXShift)
        self.assertEqual(photoCalib4.instFluxToMaggies(self.instFlux, self.pointXShift)*scaling,
                         photoCalib1.instFluxToMaggies(self.instFlux, self.pointXShift))

        # Don't allow division of BoundedFields with different bounding boxes
        photoCalibNoBBox = lsst.afw.image.PhotoCalib(self.instFlux0, self.instFlux0Err)
        with self.assertRaises(lsst.pex.exceptions.DomainError):
            scaling = photoCalibNoBBox.computeScalingTo(photoCalib1)
        with self.assertRaises(lsst.pex.exceptions.DomainError):
            scaling = photoCalibNoBBox.computeScalingTo(photoCalib4)
        with self.assertRaises(lsst.pex.exceptions.DomainError):
            scaling = photoCalib1.computeScalingTo(photoCalibNoBBox)

    def _testPersistence(self, photoCalib):
        photoCalib.writeFits(self.persistenceFile)
        result = lsst.afw.image.PhotoCalib.readFits(self.persistenceFile)
        self.assertEqual(result, photoCalib)
        os.remove(self.persistenceFile)

    def testPersistence(self):
        photoCalib = lsst.afw.image.PhotoCalib(self.instFlux0)
        self._testPersistence(photoCalib)

        photoCalib = lsst.afw.image.PhotoCalib(self.instFlux0, self.instFlux0Err)
        self._testPersistence(photoCalib)

        photoCalib = lsst.afw.image.PhotoCalib(self.instFlux0, self.instFlux0Err, self.bbox)
        self._testPersistence(photoCalib)

        photoCalib = lsst.afw.image.PhotoCalib(self.constantZeroPoint, self.instFlux0Err)
        self._testPersistence(photoCalib)

        photoCalib = lsst.afw.image.PhotoCalib(self.linearXZeroPoint, self.instFlux0Err)
        self._testPersistence(photoCalib)

    def testCalibEquality(self):
        photoCalib1 = lsst.afw.image.PhotoCalib(self.linearXZeroPoint, 0.5)
        photoCalib2 = lsst.afw.image.PhotoCalib(self.linearXZeroPoint, 0.5)
        photoCalib3 = lsst.afw.image.PhotoCalib(5, 0.5)
        photoCalib4 = lsst.afw.image.PhotoCalib(5, 0.5)
        photoCalib5 = lsst.afw.image.PhotoCalib(5)
        photoCalib6 = lsst.afw.image.PhotoCalib(self.linearXZeroPoint)
        photoCalib7 = lsst.afw.image.PhotoCalib(self.instFlux0, 0.5)
        photoCalib8 = lsst.afw.image.PhotoCalib(self.constantZeroPoint, 0.5)

        constantZeroPoint = lsst.afw.math.ChebyshevBoundedField(self.bbox, np.array([[self.instFlux0]]))
        photoCalib9 = lsst.afw.image.PhotoCalib(constantZeroPoint, 0.5)

        self.assertEqual(photoCalib1, photoCalib1)
        self.assertEqual(photoCalib1, photoCalib2)
        self.assertEqual(photoCalib3, photoCalib4)
        self.assertEqual(photoCalib8, photoCalib9)

        self.assertNotEqual(photoCalib1, photoCalib6)
        self.assertNotEqual(photoCalib1, photoCalib7)
        self.assertNotEqual(photoCalib1, photoCalib3)
        self.assertNotEqual(photoCalib3, photoCalib5)
        self.assertNotEqual(photoCalib1, photoCalib8)

        self.assertFalse(photoCalib1 != photoCalib2)  # using assertFalse to directly test != operator


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
