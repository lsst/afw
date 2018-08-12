#
# LSST Data Management System
# Copyright 2017 LSST Corporation.
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
import math
import unittest

import numpy as np
from numpy.testing import assert_allclose

from lsst.pex.exceptions import TypeError
from lsst.daf.base import PropertyList
import lsst.geom
import lsst.afw.geom as afwGeom
import lsst.utils.tests
from lsst.afw.geom.wcsUtils import createTrivialWcsMetadata, deleteBasicWcsMetadata, \
    getCdMatrixFromMetadata, getSipMatrixFromMetadata, getImageXY0FromMetadata, \
    hasSipMatrix, makeSipMatrixMetadata, makeTanSipMetadata, \
    computePixelToDistortedPixel, makeDistortedTanWcs


def makeRotationMatrix(angle, scale):
    angleRad = angle.asRadians()
    sinAng = math.sin(angleRad)
    cosAng = math.cos(angleRad)
    return np.array([
        ([cosAng, sinAng]),
        ([-sinAng, cosAng]),
    ], dtype=float)*scale


class BaseTestCase(lsst.utils.tests.TestCase):
    """Base class for testing makeDistortedTanWcs and
    computePixelToDistortedPixel
    """
    def setUp(self):
        # define the position and size of one CCD in the focal plane
        self.pixelSizeMm = 0.024  # mm/pixel
        self.ccdOrientation = 5*lsst.geom.degrees  # orientation of pixel w.r.t. focal plane
        self.plateScale = 0.15*lsst.geom.arcseconds  # angle/pixel
        self.bbox = lsst.geom.Box2I(lsst.geom.Point2I(0, 0), lsst.geom.Extent2I(2000, 4000))
        self.crpix = lsst.geom.Point2D(1000, 2000)
        self.crval = lsst.geom.SpherePoint(10, 40, lsst.geom.degrees)
        self.orientation = -45*lsst.geom.degrees
        self.scale = 1.0*lsst.geom.arcseconds
        # position of 0,0 pixel position in focal plane
        self.ccdPositionMm = lsst.geom.Point2D(25.0, 10.0)
        self.pixelToFocalPlane = self.makeAffineTransform(
            offset=lsst.geom.Extent2D(self.ccdPositionMm),
            rotation=self.ccdOrientation,
            scale=self.pixelSizeMm,
        )
        cdMatrix = afwGeom.makeCdMatrix(scale=self.scale, orientation=self.orientation)
        self.tanWcs = afwGeom.makeSkyWcs(crpix=self.crpix, crval=self.crval, cdMatrix=cdMatrix)
        self.radPerMm = self.plateScale.asRadians()/self.pixelSizeMm  # at center of field
        bboxD = lsst.geom.Box2D(self.bbox)
        self.pixelPoints = bboxD.getCorners()
        self.pixelPoints.append(bboxD.getCenter())

    def makeAffineTransform(self, offset=(0, 0), rotation=0*lsst.geom.degrees, scale=1.0):
        """Make an affine TransformPoint2ToPoint2 that first adds the specified offset,
        then scales and rotates the result
        """
        rotScale = lsst.geom.AffineTransform(lsst.geom.LinearTransform.makeScaling(scale) *
                                             lsst.geom.LinearTransform.makeRotation(rotation))
        offset = lsst.geom.AffineTransform(lsst.geom.Extent2D(*offset))
        # AffineTransform a*b = b.then(a)
        return afwGeom.makeTransform(rotScale*offset)


class MakeDistortedTanWcsTestCase(BaseTestCase):
    """Test lsst.afw.geom.makeDistortedTanWcs
    """

    def testNoDistortion(self):
        """Test makeDistortedTanWcs using an affine transform for pixelToFocalPlane

        Construct pixelToFocalPlane to match the plate scale used to
        generate self.tanWcs, the input to makeDistortedTanWcs. Thus the WCS
        returned by makeDistortedTanWcs should match self.tanWcs.
        """
        focalPlaneToFieldAngle = self.makeAffineTransform(scale=self.radPerMm)
        wcs = makeDistortedTanWcs(
            tanWcs=self.tanWcs,
            pixelToFocalPlane=self.pixelToFocalPlane,
            focalPlaneToFieldAngle=focalPlaneToFieldAngle,
        )
        self.assertWcsAlmostEqualOverBBox(wcs, self.tanWcs, bbox=self.bbox)

    def testDistortion(self):
        """Test makeDistortedTanWcs using a non-affine transform for pixelToFocalPlane
        """
        # Compute a distorted wcs that matches self.tanWcs at the center of the field;
        # the amount of distortion is 10s of pixels over the detector
        fieldAngleToFocalPlane = afwGeom.makeRadialTransform([0.0, 1/self.radPerMm, 0.0, 1000/self.radPerMm])
        focalPlaneToFieldAngle = fieldAngleToFocalPlane.inverted()
        focalPlaneToTanFieldAngle = self.makeAffineTransform(scale=self.radPerMm)
        wcs = makeDistortedTanWcs(
            tanWcs=self.tanWcs,
            pixelToFocalPlane=self.pixelToFocalPlane,
            focalPlaneToFieldAngle=focalPlaneToFieldAngle,
        )

        # At the center of the focal plane both WCS should give the same sky position
        pixelAtCtr = self.pixelToFocalPlane.applyInverse(lsst.geom.Point2D(0, 0))
        tanSkyAtCtr = self.tanWcs.pixelToSky(pixelAtCtr)
        skyAtCtr = wcs.pixelToSky(pixelAtCtr)
        self.assertPairsAlmostEqual(tanSkyAtCtr, skyAtCtr)

        # At all reasonable sky points the following field angles should be almost equal:
        #   sky -> tanWcs.skyToPixel -> pixelToFocalPlane -> focalPlaneToTanFieldAngle
        #   sky -> wcs.skyToPixel -> pixelToFocalPlane -> focalPlaneToFieldAngle
        # where focalPlaneToTanFieldAngle is the linear approximation to
        # focalPlaneToFieldAngle at the center of the field (where tanWcs and wcs match),
        # since for a given pointing, field angle gives position on the sky
        skyPoints = self.tanWcs.pixelToSky(self.pixelPoints)

        tanFieldAnglePoints = focalPlaneToTanFieldAngle.applyForward(
            self.pixelToFocalPlane.applyForward(self.tanWcs.skyToPixel(skyPoints)))
        fieldAnglePoints = focalPlaneToFieldAngle.applyForward(
            self.pixelToFocalPlane.applyForward(wcs.skyToPixel(skyPoints)))
        assert_allclose(tanFieldAnglePoints, fieldAnglePoints)

        # The inverse should also be true: for a set of field angle points
        # the following sky positions should be almost equal:
        # fieldAngle -> fieldAngleToTanFocalPlane -> focalPlaneToPixel -> tanWcs.pixelToSky
        # fieldAngle -> fieldAngleToFocalPlane -> focalPlaneToPixel -> wcs.pixelToSky
        focalPlaneToPixel = self.pixelToFocalPlane.inverted()
        fieldAngleToTanFocalPlane = focalPlaneToTanFieldAngle.inverted()
        tanSkyPoints2 = self.tanWcs.pixelToSky(
            focalPlaneToPixel.applyForward(
                fieldAngleToTanFocalPlane.applyForward(fieldAnglePoints)))

        skyPoints2 = wcs.pixelToSky(
            focalPlaneToPixel.applyForward(
                fieldAngleToFocalPlane.applyForward(fieldAnglePoints)))

        self.assertSpherePointListsAlmostEqual(tanSkyPoints2, skyPoints2)


class ComputePixelToDistortedPixelTestCase(BaseTestCase):
    """Test lsst.afw.geom.computePixelToDistortedPixel
    """

    def testNoDistortion(self):
        """Test computePixelToDistortedPixel without distortion

        Use an affine transform for pixelToFocalPlane; the transform
        returned by computePixelToDistortedPixel should be the identity transform
        """
        focalPlaneToFieldAngle = self.makeAffineTransform(scale=self.radPerMm)
        pixelToDistortedPixel = computePixelToDistortedPixel(
            pixelToFocalPlane=self.pixelToFocalPlane,
            focalPlaneToFieldAngle=focalPlaneToFieldAngle,
        )
        bboxD = lsst.geom.Box2D(self.bbox)
        pixelPoints = bboxD.getCorners()
        pixelPoints.append(bboxD.getCenter())

        assert_allclose(pixelToDistortedPixel.applyForward(pixelPoints), pixelPoints)
        assert_allclose(pixelToDistortedPixel.applyInverse(pixelPoints), pixelPoints)

    def testDistortion(self):
        """Test computePixelToDistortedPixel with distortion

        pixelToDistortedPixel -> self.tanWcs should match a WCS
        created with makeDistortedTanWcs
        """
        focalPlaneToFieldAngle = afwGeom.makeRadialTransform([0.0, self.radPerMm, 0.0, self.radPerMm])
        pixelToDistortedPixel = computePixelToDistortedPixel(
            pixelToFocalPlane=self.pixelToFocalPlane,
            focalPlaneToFieldAngle=focalPlaneToFieldAngle,
        )
        # Do not try to make pixelToDistortedPixel -> self.tanWcs into a WCS
        # because the frame names will be wrong; use a TransformPoint2Tolsst.geom.SpherePoint instead
        tanWcsTransform = afwGeom.TransformPoint2ToSpherePoint(self.tanWcs.getFrameDict())
        pixelToDistortedSky = pixelToDistortedPixel.then(tanWcsTransform)

        wcs = makeDistortedTanWcs(
            tanWcs=self.tanWcs,
            pixelToFocalPlane=self.pixelToFocalPlane,
            focalPlaneToFieldAngle=focalPlaneToFieldAngle,
        )

        bboxD = lsst.geom.Box2D(self.bbox)
        pixelPoints = bboxD.getCorners()
        pixelPoints.append(bboxD.getCenter())

        skyPoints1 = pixelToDistortedSky.applyForward(pixelPoints)
        skyPoints2 = wcs.pixelToSky(pixelPoints)
        self.assertSpherePointListsAlmostEqual(skyPoints1, skyPoints2)

        pixelPoints1 = pixelToDistortedSky.applyInverse(skyPoints1)
        pixelPoints2 = wcs.skyToPixel(skyPoints1)
        assert_allclose(pixelPoints1, pixelPoints2)


class DetailTestCase(lsst.utils.tests.TestCase):
    """Test functions in the detail sub-namespace
    """
    def setUp(self):
        # Actual WCS from processing Suprime-Cam
        self.width = 2048
        self.height = 4177
        metadata = PropertyList()
        for name, value in (
            ('NAXIS', 2),
            ('EQUINOX', 2000.0000000000),
            ('RADESYS', "ICRS"),
            ('CRPIX1', -3232.7544925483),
            ('CRPIX2', 4184.4881091129),
            ('CD1_1', -5.6123808607273e-05),
            ('CD1_2', 2.8951544956703e-07),
            ('CD2_1', 2.7343044348306e-07),
            ('CD2_2', 5.6100888336445e-05),
            ('CRVAL1', 5.6066137655191),
            ('CRVAL2', -0.60804032498548),
            ('CUNIT1', "deg"),
            ('CUNIT2', "deg"),
            ('A_ORDER', 5),
            ('A_0_2', 1.9749832126246e-08),
            ('A_0_3', 9.3734869173527e-12),
            ('A_0_4', 1.8812994578840e-17),
            ('A_0_5', -2.3524013652433e-19),
            ('A_1_1', -9.8443908806559e-10),
            ('A_1_2', -4.9278297504858e-10),
            ('A_1_3', -2.8491604610001e-16),
            ('A_1_4', 2.3185723720750e-18),
            ('A_2_0', 4.9546089730708e-08),
            ('A_2_1', -8.8592221672777e-12),
            ('A_2_2', 3.3560100338765e-16),
            ('A_2_3', 3.0469486185035e-21),
            ('A_3_0', -4.9332471706700e-10),
            ('A_3_1', -5.3126029725748e-16),
            ('A_3_2', 4.7795824885726e-18),
            ('A_4_0', 1.3128844828963e-16),
            ('A_4_1', 4.4014452170715e-19),
            ('A_5_0', 2.1781986904162e-18),
            ('B_ORDER', 5),
            ('B_0_2', -1.0607653075899e-08),
            ('B_0_3', -4.8693887937365e-10),
            ('B_0_4', -1.0363305097301e-15),
            ('B_0_5', 1.9621640066919e-18),
            ('B_1_1', 3.0340657679481e-08),
            ('B_1_2', -5.0763819284853e-12),
            ('B_1_3', 2.8987281654754e-16),
            ('B_1_4', 1.8253389678593e-19),
            ('B_2_0', -2.4772849184248e-08),
            ('B_2_1', -4.9775588352207e-10),
            ('B_2_2', -3.6806326254887e-16),
            ('B_2_3', 4.4136985315418e-18),
            ('B_3_0', -1.7807191001742e-11),
            ('B_3_1', -2.4136396882531e-16),
            ('B_3_2', 2.9165413645768e-19),
            ('B_4_0', 4.1029951148438e-16),
            ('B_4_1', 2.3711874424169e-18),
            ('B_5_0', 4.9333635889310e-19),
            ('AP_ORDER', 5),
            ('AP_0_1', -5.9740855298291e-06),
            ('AP_0_2', -2.0433429597268e-08),
            ('AP_0_3', -8.6810071023434e-12),
            ('AP_0_4', -2.4974690826778e-17),
            ('AP_0_5', 1.9819631102516e-19),
            ('AP_1_0', -4.5896648256716e-05),
            ('AP_1_1', -1.5248993348644e-09),
            ('AP_1_2', 5.0283116166943e-10),
            ('AP_1_3', 4.3796281513144e-16),
            ('AP_1_4', -2.1447889127908e-18),
            ('AP_2_0', -4.7550300344365e-08),
            ('AP_2_1', 1.0924172283232e-11),
            ('AP_2_2', -4.9862026098260e-16),
            ('AP_2_3', -5.4470851768869e-20),
            ('AP_3_0', 5.0130654116966e-10),
            ('AP_3_1', 6.8649554020012e-16),
            ('AP_3_2', -4.2759588436342e-18),
            ('AP_4_0', -3.6306802581471e-16),
            ('AP_4_1', -5.3885285875084e-19),
            ('AP_5_0', -1.8802693525108e-18),
            ('BP_ORDER', 5),
            ('BP_0_1', -2.6627855995942e-05),
            ('BP_0_2', 1.1143451873584e-08),
            ('BP_0_3', 4.9323396530135e-10),
            ('BP_0_4', 1.1785185735421e-15),
            ('BP_0_5', -1.6169957016415e-18),
            ('BP_1_0', -5.7914490267576e-06),
            ('BP_1_1', -3.0565765766244e-08),
            ('BP_1_2', 5.7727475030971e-12),
            ('BP_1_3', -4.0586821113726e-16),
            ('BP_1_4', -2.0662723654322e-19),
            ('BP_2_0', 2.3705520015164e-08),
            ('BP_2_1', 5.0530823594352e-10),
            ('BP_2_2', 3.8904979943489e-16),
            ('BP_2_3', -3.8346209540986e-18),
            ('BP_3_0', 1.9505421473262e-11),
            ('BP_3_1', 1.7583146713289e-16),
            ('BP_3_2', -3.4876779564534e-19),
            ('BP_4_0', -3.3690937119054e-16),
            ('BP_4_1', -2.0853007589561e-18),
            ('BP_5_0', -5.5344298912288e-19),
            ('CTYPE1', "RA---TAN-SIP"),
            ('CTYPE2', "DEC--TAN-SIP"),
        ):
            metadata.set(name, value)
        self.metadata = metadata

    def testCreateTrivialWcsAsPropertySet(self):
        wcsName = "Z"  # arbitrary
        xy0 = lsst.geom.Point2I(47, -200)  # arbitrary
        metadata = createTrivialWcsMetadata(wcsName=wcsName, xy0=xy0)
        desiredNameValueList = (  # names are missing wcsName suffix
            ("CRPIX1", 1.0),
            ("CRPIX2", 1.0),
            ("CRVAL1", xy0[0]),
            ("CRVAL2", xy0[1]),
            ("CTYPE1", "LINEAR"),
            ("CTYPE2", "LINEAR"),
            ("CUNIT1", "PIXEL"),
            ("CUNIT2", "PIXEL"),
        )
        self.assertEqual(len(metadata.names(True)), len(desiredNameValueList))
        for name, value in desiredNameValueList:
            self.assertEqual(metadata.getScalar(name + wcsName), value)

    def testDeleteBasicWcsMetadata(self):
        wcsName = "Q"  # arbitrary
        metadata = createTrivialWcsMetadata(wcsName=wcsName, xy0=lsst.geom.Point2I(0, 0))
        # add the other keywords that will be deleted
        for i in range(2):
            for j in range(2):
                metadata.set("CD%d_%d%s" % (i+1, j+1, wcsName), 0.2)  # arbitrary nonzero value
        metadata.set("WCSAXES%s" % (wcsName,), 2)
        # add a keyword that will not be deleted
        metadata.set("NAXIS1", 100)
        self.assertEqual(len(metadata.names(True)), 14)

        # deleting data for a different WCS will delete nothing
        deleteBasicWcsMetadata(metadata=metadata, wcsName="B")
        self.assertEqual(len(metadata.names(True)), 14)

        # deleting data for the right WCS deletes all but one keyword
        deleteBasicWcsMetadata(metadata=metadata, wcsName=wcsName)
        self.assertEqual(len(metadata.names(True)), 1)
        self.assertEqual(metadata.getScalar("NAXIS1"), 100)

        # try with a smattering of keywords (should silently ignore the missing ones)
        metadata.set("WCSAXES%s" % (wcsName,), 2)
        metadata.set("CD1_2%s" % (wcsName,), 0.5)
        metadata.set("CRPIX2%s" % (wcsName,), 5)
        metadata.set("CRVAL1%s" % (wcsName,), 55)
        deleteBasicWcsMetadata(metadata=metadata, wcsName=wcsName)
        self.assertEqual(len(metadata.names(True)), 1)
        self.assertEqual(metadata.getScalar("NAXIS1"), 100)

    def testGetImageXY0FromMetadata(self):
        wcsName = "Z"  # arbitrary
        xy0 = lsst.geom.Point2I(47, -200)  # arbitrary with a negative value to check rounding
        metadata = createTrivialWcsMetadata(wcsName=wcsName, xy0=xy0)

        # reading the wrong wcsName should be treated as no data available
        xy0WrongWcsName = getImageXY0FromMetadata(metadata=metadata, wcsName="X", strip=True)
        self.assertEqual(xy0WrongWcsName, lsst.geom.Point2I(0, 0))
        self.assertEqual(len(metadata.names(True)), 8)

        # deleting one of the required keywords should be treated as no data available
        for namePrefixToRemove in ("CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2"):
            nameToRemove = namePrefixToRemove + wcsName
            removedValue = metadata.getScalar(nameToRemove)
            metadata.remove(nameToRemove)
            xy0MissingWcsKey = getImageXY0FromMetadata(metadata=metadata, wcsName=wcsName, strip=True)
            self.assertEqual(xy0MissingWcsKey, lsst.geom.Point2I(0, 0))
            self.assertEqual(len(metadata.names(True)), 7)
            # restore removed item
            metadata.set(nameToRemove, removedValue)
            self.assertEqual(len(metadata.names(True)), 8)

        # setting CRPIX1, 2 to something other than 1 should be treated as no data available
        for i in (1, 2):
            nameToChange = "CRPIX%d%s" % (i, wcsName)
            metadata.set(nameToChange, 1.1)
            xy0WrongWcsName = getImageXY0FromMetadata(metadata=metadata, wcsName=wcsName, strip=True)
            self.assertEqual(xy0WrongWcsName, lsst.geom.Point2I(0, 0))
            self.assertEqual(len(metadata.names(True)), 8)
            # restore altered CRPIX value
            metadata.set(nameToChange, 1.0)
            self.assertEqual(len(metadata.names(True)), 8)

        # use the correct WCS name but don't strip
        xy0RightWcsName = getImageXY0FromMetadata(metadata, wcsName, strip=False)
        self.assertEqual(xy0RightWcsName, xy0)
        self.assertEqual(len(metadata.names(True)), 8)

        # use the correct WCS and strip usable metadata
        xy0RightWcsName = getImageXY0FromMetadata(metadata, wcsName, strip=True)
        self.assertEqual(xy0RightWcsName, xy0)
        self.assertEqual(len(metadata.names(True)), 0)

    def testGetSipMatrixFromMetadata(self):
        """Test getSipMatrixFromMetadata and makeSipMatrixMetadata
        """
        for badName in ("X", "AA"):
            self.assertFalse(hasSipMatrix(self.metadata, badName))
            with self.assertRaises(TypeError):
                getSipMatrixFromMetadata(self.metadata, badName)

        for name in ("A", "B", "AP", "BP"):
            self.assertTrue(hasSipMatrix(self.metadata, name))
            sipMatrix = getSipMatrixFromMetadata(self.metadata, name)
            width = self.metadata.getScalar("%s_ORDER" % (name,)) + 1
            self.assertEqual(sipMatrix.shape, (width, width))
            for i in range(width):
                for j in range(width):
                    # SIP matrix terms use 0-based indexing
                    cardName = "%s_%d_%d" % (name, i, j)
                    if self.metadata.exists(cardName):
                        self.assertEqual(sipMatrix[i, j], self.metadata.getScalar(cardName))
                    else:
                        self.assertEqual(sipMatrix[i, j], 0.0)

            metadata = makeSipMatrixMetadata(sipMatrix, name)
            for name in metadata.names(False):
                value = metadata.getScalar(name)
                if (name.endswith("ORDER")):
                    self.assertEqual(width, value + 1)
                else:
                    self.assertEqual(value, self.metadata.getScalar(name))
                    self.assertNotEqual(value, 0.0)  # 0s are omitted

        # try metadata with only the ORDER keyword; the matrix should be all zeros
        # except for the invalid case of order < 0
        for order in (-3, -1, 0, 3):
            metadata2 = PropertyList()
            metadata2.set("W_ORDER", order)
            if order < 0:
                # invalid order
                self.assertFalse(hasSipMatrix(metadata2, "W"))
                with self.assertRaises(TypeError):
                    getSipMatrixFromMetadata(metadata2, "W")
            else:
                self.assertTrue(hasSipMatrix(metadata2, "W"))
                zeroMatrix = getSipMatrixFromMetadata(metadata2, "W")
                self.assertEqual(zeroMatrix.shape, (order + 1, order + 1))
                for i in range(order + 1):
                    for j in range(order + 1):
                        self.assertEqual(zeroMatrix[i, j], 0.0)

    def testGetCdMatrixFromMetadata(self):
        cdMatrix = getCdMatrixFromMetadata(self.metadata)
        for i in range(2):
            for j in range(2):
                cardName = "CD%d_%d" % (i + 1, j + 1)
                self.assertEqual(cdMatrix[i, j], self.metadata.getScalar(cardName))

        metadata = PropertyList()
        with self.assertRaises(TypeError):
            getCdMatrixFromMetadata(metadata)
        metadata.add("CD2_1", 0.56)  # just one term, with an arbitrary value
        cdMatrix2 = getCdMatrixFromMetadata(metadata)
        for i in range(2):
            for j in range(2):
                # CD matrix terms use 1-based indexing
                cardName = "CD%d_%d" % (i + 1, j + 1)
                if i == 1 and j == 0:
                    self.assertEqual(cdMatrix2[i, j], 0.56)
                else:
                    self.assertEqual(cdMatrix2[i, j], 0.0)

    def testMakeTanSipMetadata(self):
        """Test makeTanSipMetadata
        """
        crpix = lsst.geom.Point2D(self.metadata.getScalar("CRPIX1") - 1,
                                  self.metadata.getScalar("CRPIX2") - 1)
        crval = lsst.geom.SpherePoint(self.metadata.getScalar("CRVAL1"),
                                      self.metadata.getScalar("CRVAL2"), lsst.geom.degrees)
        cdMatrix = getCdMatrixFromMetadata(self.metadata)
        sipA = getSipMatrixFromMetadata(self.metadata, "A")
        sipB = getSipMatrixFromMetadata(self.metadata, "B")
        sipAp = getSipMatrixFromMetadata(self.metadata, "AP")
        sipBp = getSipMatrixFromMetadata(self.metadata, "BP")
        forwardMetadata = makeTanSipMetadata(
            crpix=crpix,
            crval=crval,
            cdMatrix=cdMatrix,
            sipA=sipA,
            sipB=sipB,
        )
        self.assertFalse(forwardMetadata.exists("AP_ORDER"))
        self.assertFalse(forwardMetadata.exists("BP_ORDER"))

        fullMetadata = makeTanSipMetadata(
            crpix=crpix,
            crval=crval,
            cdMatrix=cdMatrix,
            sipA=sipA,
            sipB=sipB,
            sipAp=sipAp,
            sipBp=sipBp,
        )
        for cardName in ("CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2", "CTYPE1", "CTYPE2",
                         "CUNIT1", "CUNIT2", "RADESYS"):
            self.assertTrue(forwardMetadata.exists(cardName))
            self.assertTrue(fullMetadata.exists(cardName))
        for name, matrix in (("A", sipA), ("B", sipB)):
            self.checkSipMetadata(name, matrix, forwardMetadata)
            self.checkSipMetadata(name, matrix, fullMetadata)
        for name, matrix in (("AP", sipAp), ("BP", sipBp)):
            self.checkSipMetadata(name, matrix, fullMetadata)

    def checkSipMetadata(self, name, sipMatrix, metadata):
        width = metadata.getScalar("%s_ORDER" % (name,)) + 1
        self.assertEqual(width, sipMatrix.shape[0])
        for i in range(width):
            for j in range(width):
                cardName = "%s_%s_%s" % (name, i, j)
                value = sipMatrix[i, j]
                if value != 0 or metadata.exists(cardName):
                    self.assertEqual(value, metadata.getScalar(cardName))


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
