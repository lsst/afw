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

import unittest

import numpy as np

import lsst.utils.tests
import lsst.geom
import lsst.afw.image
import lsst.afw.image.testUtils
import lsst.afw.math
import lsst.daf.base
import lsst.pex.exceptions


def computeMaggiesErr(instFluxErr, instFlux, calibrationErr, calibration, flux):
    """Return the error on the flux (Maggies)."""
    return flux*np.hypot(instFluxErr/instFlux, calibrationErr/calibration)


def computeMagnitudeErr(instFluxErr, instFlux, calibrationErr, calibration, flux):
    """Return the error on the magnitude."""
    return 2.5/np.log(10)*computeMaggiesErr(instFluxErr, instFlux, calibrationErr, calibration, flux) / flux


def makeCalibratedMaskedImage(image, mask, variance, outImage, calibration, calibrationErr):
    """Return a MaskedImage using outImage, mask, and a computed variance image."""
    outErr = computeMaggiesErr(np.sqrt(variance),
                               image,
                               calibrationErr,
                               calibration,
                               outImage).astype(np.float32)  # variance plane must be 32bit
    return lsst.afw.image.makeMaskedImageFromArrays(outImage,
                                                    mask,
                                                    outErr**2)


class PhotoCalibTestCase(lsst.utils.tests.TestCase):

    def setUp(self):
        np.random.seed(100)

        self.point0 = lsst.geom.Point2D(0, 0)
        self.pointXShift = lsst.geom.Point2D(-10, 0)
        self.pointYShift = lsst.geom.Point2D(0, -10)
        self.bbox = lsst.geom.Box2I(lsst.geom.Point2I(-100, -100), lsst.geom.Point2I(100, 100))

        # calibration and instFlux designed to produce calibrated flux of 1.
        self.calibration = 1e-3
        self.calibrationErr = 1e-4
        self.instFlux = 1000.
        self.instFluxErr = 10.

        self.schema = lsst.afw.table.SourceTable.makeMinimalSchema()
        self.instFluxKeyName = "SomeFlux"
        lsst.afw.table.Point2DKey.addFields(self.schema, "centroid", "centroid", "pixels")
        self.instFluxKey = self.schema.addField(
            self.instFluxKeyName+"_instFlux", type="D", doc="post-ISR instFlux")
        self.instFluxErrKey = self.schema.addField(self.instFluxKeyName+"_instFluxErr", type="D",
                                                   doc="post-ISR instFlux stddev")
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

        self.constantCalibration = lsst.afw.math.ChebyshevBoundedField(self.bbox,
                                                                       np.array([[self.calibration]]))
        self.linearXCalibration = lsst.afw.math.ChebyshevBoundedField(self.bbox,
                                                                      np.array([[self.calibration,
                                                                               self.calibration]]))

    def tearDown(self):
        del self.schema
        del self.table
        del self.catalog

    def _testPhotoCalibCenter(self, photoCalib, calibrationErr):
        """
        Test conversions of instFlux for the mean and (0,0) value of a photoCalib.
        Assumes those are the same, e.g. that the non-constant terms are all
        odd, and that the mean of the calib is self.calibration.

        calibrationErr is provided as an option to allow testing of photoCalibs
        that have no error specified, and those that do.
        """
        # test that the constructor set the calibrationMean and err correctly
        self.assertEqual(self.calibration, photoCalib.getCalibrationMean())
        self.assertEqual(self.calibration, 1.0/photoCalib.getInstFluxMag0())
        self.assertEqual(calibrationErr, photoCalib.getCalibrationErr())

        # useful reference points: 1 nanomaggy == magnitude 22.5, 1 maggy = magnitude 0
        self.assertEqual(1, photoCalib.instFluxToMaggies(self.instFlux))
        self.assertEqual(0, photoCalib.instFluxToMagnitude(self.instFlux))

        self.assertFloatsAlmostEqual(1e-9, photoCalib.instFluxToMaggies(self.instFlux*1e-9))
        self.assertFloatsAlmostEqual(22.5, photoCalib.instFluxToMagnitude(self.instFlux*1e-9))
        # test that (0,0) gives the same result as above
        self.assertFloatsAlmostEqual(1e-9, photoCalib.instFluxToMaggies(self.instFlux*1e-9, self.point0))
        self.assertFloatsAlmostEqual(22.5, photoCalib.instFluxToMagnitude(self.instFlux*1e-9, self.point0))

        # test that we get a correct maggies err for the base instFlux
        errFlux = computeMaggiesErr(self.instFluxErr, self.instFlux, calibrationErr, self.calibration, 1)
        result = photoCalib.instFluxToMaggies(self.instFlux, self.instFluxErr)
        self.assertEqual(1, result.value)
        self.assertFloatsAlmostEqual(errFlux, result.err)
        result = photoCalib.instFluxToMaggies(self.instFlux, self.instFluxErr, self.point0)
        self.assertFloatsAlmostEqual(1, result.value)
        self.assertFloatsAlmostEqual(errFlux, result.err)

        # test that we get a correct magnitude err for the base instFlux
        errMag = computeMagnitudeErr(self.instFluxErr, self.instFlux, calibrationErr, self.calibration, 1)
        result = photoCalib.instFluxToMagnitude(self.instFlux, self.instFluxErr)
        self.assertEqual(0, result.value)
        self.assertFloatsAlmostEqual(errMag, result.err)
        result = photoCalib.instFluxToMagnitude(self.instFlux, self.instFluxErr, self.point0)
        self.assertFloatsAlmostEqual(0, result.value)
        self.assertFloatsAlmostEqual(errMag, result.err)

        # test that we get a correct maggies err for base instFlux*1e-9
        errFluxNano = computeMaggiesErr(self.instFluxErr, self.instFlux*1e-9,
                                        calibrationErr, self.calibration, 1e-9)
        result = photoCalib.instFluxToMaggies(self.instFlux*1e-9, self.instFluxErr)
        self.assertFloatsAlmostEqual(1e-9, result.value)
        self.assertFloatsAlmostEqual(errFluxNano, result.err)
        result = photoCalib.instFluxToMaggies(self.instFlux*1e-9, self.instFluxErr, self.point0)
        self.assertFloatsAlmostEqual(1e-9, result.value)
        self.assertFloatsAlmostEqual(errFluxNano, result.err)

        # test that we get a correct magnitude err for base instFlux*1e-9
        errMagNano = computeMagnitudeErr(self.instFluxErr, self.instFlux*1e-9,
                                         calibrationErr, self.calibration, 1e-9)
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

        # modify the catalog in-place.
        photoCalib.instFluxToMagnitude(catalog, self.instFluxKeyName, self.instFluxKeyName)
        self.assertFloatsAlmostEqual(catalog[self.instFluxKeyName+'_mag'], expectMag[:, 0])
        self.assertFloatsAlmostEqual(catalog[self.instFluxKeyName+'_magErr'], expectMag[:, 1])

        # TODO: have to save the values and restore them, until DM-10302 is implemented.
        origFlux = catalog[self.instFluxKeyName+'_instFlux'].copy()
        origFluxErr = catalog[self.instFluxKeyName+'_instFluxErr'].copy()
        photoCalib.instFluxToMaggies(catalog, self.instFluxKeyName, self.instFluxKeyName)
        self.assertFloatsAlmostEqual(catalog[self.instFluxKeyName+'_instFlux'], expectMaggies[:, 0])
        self.assertFloatsAlmostEqual(catalog[self.instFluxKeyName+'_instFluxErr'], expectMaggies[:, 1])
        # TODO: restore values, until DM-10302 is implemented.
        for record, f, fErr in zip(catalog, origFlux, origFluxErr):
            record.set(self.instFluxKeyName+'_instFlux', f)
            record.set(self.instFluxKeyName+'_instFluxErr', fErr)

    def testNonVarying(self):
        """Tests a non-spatially-varying Calibration."""
        photoCalib = lsst.afw.image.PhotoCalib(self.calibration)
        self._testPhotoCalibCenter(photoCalib, 0)

        self.assertEqual(1, photoCalib.instFluxToMaggies(self.instFlux, self.pointXShift))
        self.assertEqual(0, photoCalib.instFluxToMagnitude(self.instFlux, self.pointXShift))
        result = photoCalib.instFluxToMaggies(self.instFlux, self.instFluxErr)
        self.assertEqual(1, result.value)

        photoCalib = lsst.afw.image.PhotoCalib(self.calibration, self.calibrationErr)
        self._testPhotoCalibCenter(photoCalib, self.calibrationErr)

        # constant, with a bbox
        photoCalib = lsst.afw.image.PhotoCalib(self.calibration, bbox=self.bbox)
        self._testPhotoCalibCenter(photoCalib, 0)

    def testConstantBoundedField(self):
        """Test a spatially-constant bounded field."""
        photoCalib = lsst.afw.image.PhotoCalib(self.constantCalibration)
        self._testPhotoCalibCenter(photoCalib, 0)

        self.assertEqual(1, photoCalib.instFluxToMaggies(self.instFlux, self.pointYShift))
        self.assertEqual(0, photoCalib.instFluxToMagnitude(self.instFlux, self.pointYShift))
        self.assertFloatsAlmostEqual(1e-9, photoCalib.instFluxToMaggies(self.instFlux*1e-9, self.pointXShift))
        self.assertFloatsAlmostEqual(22.5, photoCalib.instFluxToMagnitude(
            self.instFlux*1e-9, self.pointXShift))

        photoCalib = lsst.afw.image.PhotoCalib(self.constantCalibration, self.calibrationErr)
        self._testPhotoCalibCenter(photoCalib, self.calibrationErr)

    def testLinearXBoundedField(self):
        photoCalib = lsst.afw.image.PhotoCalib(self.linearXCalibration)
        self._testPhotoCalibCenter(photoCalib, 0)

        self.assertEqual(1, photoCalib.instFluxToMaggies(self.instFlux, self.pointYShift))
        self.assertEqual(0, photoCalib.instFluxToMagnitude(self.instFlux, self.pointYShift))

        calibration = (self.calibration + self.pointXShift.getX()*self.calibration/(self.bbox.getWidth()/2.))
        expect = self.instFlux*calibration
        self.assertFloatsAlmostEqual(expect, photoCalib.instFluxToMaggies(self.instFlux, self.pointXShift))
        self.assertFloatsAlmostEqual(-2.5*np.log10(expect),
                                     photoCalib.instFluxToMagnitude(self.instFlux, self.pointXShift))

        self.assertFloatsAlmostEqual(expect*1e-9,
                                     photoCalib.instFluxToMaggies(self.instFlux*1e-9, self.pointXShift))
        self.assertFloatsAlmostEqual(-2.5*np.log10(expect*1e-9),
                                     photoCalib.instFluxToMagnitude(self.instFlux*1e-9, self.pointXShift))

        photoCalib = lsst.afw.image.PhotoCalib(self.linearXCalibration, self.calibrationErr)
        self._testPhotoCalibCenter(photoCalib, self.calibrationErr)

        # New catalog with a spatial component in the varying direction,
        # to ensure the calculations on a catalog properly handle non-constant BF.
        # NOTE: only the first quantity of the result (maggies or mags) should change.
        catalog = self.catalog.copy(deep=True)
        catalog[0].set('centroid_x', self.pointXShift[0])
        catalog[0].set('centroid_y', self.pointXShift[1])
        errFlux = computeMaggiesErr(self.instFluxErr, self.instFlux,
                                    self.calibrationErr, calibration, expect)
        errMag = computeMagnitudeErr(self.instFluxErr, self.instFlux,
                                     self.calibrationErr, calibration, expect)
        errFluxNano = computeMaggiesErr(self.instFluxErr, self.instFlux*1e-9,
                                        self.calibrationErr, self.calibration, 1e-9)
        errMagNano = computeMagnitudeErr(self.instFluxErr, self.instFlux*1e-9,
                                         self.calibrationErr, self.calibration, 1e-9)
        expectMaggies = np.array([[expect, errFlux], [1e-9, errFluxNano]])
        expectMag = np.array([[-2.5*np.log10(expect), errMag], [22.5, errMagNano]])
        self._testSourceCatalog(photoCalib, catalog, expectMaggies, expectMag)

    def testComputeScaledCalibration(self):
        photoCalib = lsst.afw.image.PhotoCalib(self.calibration, bbox=self.bbox)
        scaledCalib = lsst.afw.image.PhotoCalib(photoCalib.computeScaledCalibration())
        self.assertEqual(1, scaledCalib.instFluxToMaggies(self.instFlux)*photoCalib.getCalibrationMean())
        self.assertEqual(photoCalib.instFluxToMaggies(self.instFlux),
                         scaledCalib.instFluxToMaggies(self.instFlux)*photoCalib.getCalibrationMean())

        photoCalib = lsst.afw.image.PhotoCalib(self.constantCalibration)
        scaledCalib = lsst.afw.image.PhotoCalib(photoCalib.computeScaledCalibration())

        self.assertEqual(1, scaledCalib.instFluxToMaggies(self.instFlux*self.calibration))
        self.assertEqual(photoCalib.instFluxToMaggies(self.instFlux),
                         scaledCalib.instFluxToMaggies(self.instFlux)*photoCalib.getCalibrationMean())

    @unittest.skip("Not yet implemented: see DM-10154")
    def testComputeScalingTo(self):
        photoCalib1 = lsst.afw.image.PhotoCalib(self.calibration, self.calibrationErr, bbox=self.bbox)
        photoCalib2 = lsst.afw.image.PhotoCalib(self.calibration*500, self.calibrationErr, bbox=self.bbox)
        scaling = photoCalib1.computeScalingTo(photoCalib2)(self.pointXShift)
        self.assertEqual(photoCalib1.instFluxToMaggies(self.instFlux, self.pointXShift)*scaling,
                         photoCalib2.instFluxToMaggies(self.instFlux, self.pointXShift))

        photoCalib3 = lsst.afw.image.PhotoCalib(self.constantCalibration, self.calibrationErr)
        scaling = photoCalib1.computeScalingTo(photoCalib3)(self.pointXShift)
        self.assertEqual(photoCalib1.instFluxToMaggies(self.instFlux, self.pointXShift)*scaling,
                         photoCalib3.instFluxToMaggies(self.instFlux, self.pointXShift))
        scaling = photoCalib3.computeScalingTo(photoCalib1)(self.pointXShift)
        self.assertEqual(photoCalib3.instFluxToMaggies(self.instFlux, self.pointXShift)*scaling,
                         photoCalib1.instFluxToMaggies(self.instFlux, self.pointXShift))

        photoCalib4 = lsst.afw.image.PhotoCalib(self.linearXCalibration, self.calibrationErr)
        scaling = photoCalib1.computeScalingTo(photoCalib4)(self.pointXShift)
        self.assertEqual(photoCalib1.instFluxToMaggies(self.instFlux, self.pointXShift)*scaling,
                         photoCalib4.instFluxToMaggies(self.instFlux, self.pointXShift))
        scaling = photoCalib4.computeScalingTo(photoCalib1)(self.pointXShift)
        self.assertEqual(photoCalib4.instFluxToMaggies(self.instFlux, self.pointXShift)*scaling,
                         photoCalib1.instFluxToMaggies(self.instFlux, self.pointXShift))

        # Don't allow division of BoundedFields with different bounding boxes
        photoCalibNoBBox = lsst.afw.image.PhotoCalib(self.calibration, self.calibrationErr)
        with self.assertRaises(lsst.pex.exceptions.DomainError):
            scaling = photoCalibNoBBox.computeScalingTo(photoCalib1)
        with self.assertRaises(lsst.pex.exceptions.DomainError):
            scaling = photoCalibNoBBox.computeScalingTo(photoCalib4)
        with self.assertRaises(lsst.pex.exceptions.DomainError):
            scaling = photoCalib1.computeScalingTo(photoCalibNoBBox)

    def _testPersistence(self, photoCalib):
        with lsst.utils.tests.getTempFilePath(".fits") as filename:
            photoCalib.writeFits(filename)
            result = lsst.afw.image.PhotoCalib.readFits(filename)
            self.assertEqual(result, photoCalib)

    def testPersistence(self):
        photoCalib = lsst.afw.image.PhotoCalib(self.calibration)
        self._testPersistence(photoCalib)

        photoCalib = lsst.afw.image.PhotoCalib(self.calibration, self.calibrationErr)
        self._testPersistence(photoCalib)

        photoCalib = lsst.afw.image.PhotoCalib(self.calibration, self.calibrationErr, self.bbox)
        self._testPersistence(photoCalib)

        photoCalib = lsst.afw.image.PhotoCalib(self.constantCalibration, self.calibrationErr)
        self._testPersistence(photoCalib)

        photoCalib = lsst.afw.image.PhotoCalib(self.linearXCalibration, self.calibrationErr)
        self._testPersistence(photoCalib)

    def testCalibEquality(self):
        photoCalib1 = lsst.afw.image.PhotoCalib(self.linearXCalibration, 0.5)
        photoCalib2 = lsst.afw.image.PhotoCalib(self.linearXCalibration, 0.5)
        photoCalib3 = lsst.afw.image.PhotoCalib(5, 0.5)
        photoCalib4 = lsst.afw.image.PhotoCalib(5, 0.5)
        photoCalib5 = lsst.afw.image.PhotoCalib(5)
        photoCalib6 = lsst.afw.image.PhotoCalib(self.linearXCalibration)
        photoCalib7 = lsst.afw.image.PhotoCalib(self.calibration, 0.5)
        photoCalib8 = lsst.afw.image.PhotoCalib(self.constantCalibration, 0.5)

        constantCalibration = lsst.afw.math.ChebyshevBoundedField(self.bbox, np.array([[self.calibration]]))
        photoCalib9 = lsst.afw.image.PhotoCalib(constantCalibration, 0.5)

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

    def setupImage(self):
        dim = (5, 6)
        npDim = (dim[1], dim[0])  # numpy and afw have a different x/y order
        sigma = 10.0
        image = np.random.normal(loc=1000.0, scale=sigma, size=npDim).astype(np.float32)
        mask = np.zeros(npDim, dtype=np.int32)
        variance = (np.random.normal(loc=0.0, scale=sigma, size=npDim).astype(np.float32))**2
        maskedImage = lsst.afw.image.basicUtils.makeMaskedImageFromArrays(image, mask, variance)
        maskedImage.mask[0, 0] = True  # set one mask bit to check propagation of mask bits.

        return npDim, maskedImage, image, mask, variance

    def testCalibrateImageConstant(self):
        """Test a spatially-constant calibration."""
        npDim, maskedImage, image, mask, variance = self.setupImage()
        outImage = maskedImage.image.getArray()*self.calibration
        expect = makeCalibratedMaskedImage(image, mask, variance, outImage,
                                           self.calibration, self.calibrationErr)
        photoCalib = lsst.afw.image.PhotoCalib(self.calibration, self.calibrationErr)
        result = photoCalib.calibrateImage(maskedImage)
        self.assertMaskedImagesAlmostEqual(expect, result)

    def testCalibrateImageNonConstant(self):
        """Test a spatially-varying calibration."""
        npDim, maskedImage, image, mask, variance = self.setupImage()
        xIndex, yIndex = np.indices(npDim, dtype=np.float64)
        # y then x, as afw order and np order are flipped
        calibration = self.linearXCalibration.evaluate(yIndex.flatten(), xIndex.flatten()).reshape(npDim)
        outImage = maskedImage.image.getArray()*calibration  # element-wise product, not matrix product
        expect = makeCalibratedMaskedImage(image, mask, variance, outImage, calibration, self.calibrationErr)

        photoCalib = lsst.afw.image.PhotoCalib(self.linearXCalibration, self.calibrationErr)
        result = photoCalib.calibrateImage(maskedImage)
        self.assertMaskedImagesAlmostEqual(expect, result)

    def testCalibrateImageNonConstantSubimage(self):
        """Test a non-constant calibration on a sub-image, to ensure we're
        handling xy0 correctly.
        """
        npDim, maskedImage, image, mask, variance = self.setupImage()
        xIndex, yIndex = np.indices(npDim, dtype=np.float64)
        calibration = self.linearXCalibration.evaluate(yIndex.flatten(), xIndex.flatten()).reshape(npDim)

        outImage = maskedImage.image.getArray()*calibration  # element-wise product, not matrix product
        expect = makeCalibratedMaskedImage(image, mask, variance, outImage, calibration, self.calibrationErr)

        subBox = lsst.geom.Box2I(lsst.geom.Point2I(2, 4), lsst.geom.Point2I(4, 5))
        subImage = maskedImage.subset(subBox)
        photoCalib = lsst.afw.image.PhotoCalib(self.linearXCalibration, self.calibrationErr)
        result = photoCalib.calibrateImage(subImage)
        self.assertMaskedImagesAlmostEqual(expect.subset(subBox), result)

    def testNonPositiveMeans(self):
        # no negative calibrations
        with(self.assertRaises(lsst.pex.exceptions.InvalidParameterError)):
            lsst.afw.image.PhotoCalib(-1.0)
        # no negative errors
        with(self.assertRaises(lsst.pex.exceptions.InvalidParameterError)):
            lsst.afw.image.PhotoCalib(1.0, -1.0)

        # no negative calibration mean when computed from the bounded field
        negativeCalibration = lsst.afw.math.ChebyshevBoundedField(self.bbox,
                                                                  np.array([[-self.calibration]]))
        with(self.assertRaises(lsst.pex.exceptions.InvalidParameterError)):
            lsst.afw.image.PhotoCalib(negativeCalibration)
        # no negative calibration error
        with(self.assertRaises(lsst.pex.exceptions.InvalidParameterError)):
            lsst.afw.image.PhotoCalib(self.constantCalibration, -1.0)

        # no negative explicit calibration mean
        with(self.assertRaises(lsst.pex.exceptions.InvalidParameterError)):
            lsst.afw.image.PhotoCalib(-1.0, 0, self.constantCalibration, True)
        # no negative calibration error
        with(self.assertRaises(lsst.pex.exceptions.InvalidParameterError)):
            lsst.afw.image.PhotoCalib(1.0, -1.0, self.constantCalibration, True)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
