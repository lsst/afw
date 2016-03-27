from __future__ import absolute_import, division
#
# LSST Data Management System
# See the COPYRIGHT and LICENSE files in the top-level directory of this
# package for notices and licensing terms.
#
import math
import numpy
import lsst.afw.geom as afwGeom

__all__ = ["makePixelToTanPixel"]

def makePixelToTanPixel(bbox, orientation, focalPlaneToPupil, pixelSizeMm, plateScale):
    """!Make an XYTransform whose forward direction converts PIXEL to TAN_PIXEL for one detector

    PIXELS and TAN_PIXELS are defined in @ref afwCameraGeomCoordSys in doc/cameraGeom.dox

    @param[in] bbox  detector bounding box (an lsst.afw.geom.Box2I)
    @param[in] orientation  orientation of detector in focal plane (an lsst.afw.cameraGeom.Orientation)
    @param[in] focalPlaneToPupil  an lsst.afw.math.XYTransform that converts from focal plane (mm)
        to pupil coordinates (radians) in the forward direction
    @param[in] pixelSizeMm  size of the pixel in mm in X and Y (an lsst.afw.geom.Extent2D)
    @param[in] plateScale  plate scale of the camera in arcsec/mm (a double)
    """
    pixelToFocalPlane = orientation.makePixelFpTransform(pixelSizeMm)

    meanPixelSizeMm = (pixelSizeMm[0] + pixelSizeMm[1]) / 2.0
    radPerMeanPix = afwGeom.Angle(plateScale, afwGeom.arcseconds).asRadians() * meanPixelSizeMm

    detCtrPix = afwGeom.Box2D(bbox).getCenter()
    detCtrTanPix = detCtrPix # by definition

    detCtrPupil = focalPlaneToPupil.forwardTransform(pixelToFocalPlane.forwardTransform(detCtrPix))

    pupilTanPixAngRad = -orientation.getYaw().asRadians()
    pupilTanPixSin = math.sin(pupilTanPixAngRad)
    pupilTanPixCos = math.cos(pupilTanPixAngRad)
    tanPixToPupilRotMat = numpy.array((
        (pupilTanPixCos, pupilTanPixSin),
        (-pupilTanPixSin, pupilTanPixCos),
    )) * radPerMeanPix
    tanPixToPupilRotTransform = afwGeom.AffineTransform(tanPixToPupilRotMat)

    tanPixCtrMinus0Pupil = tanPixToPupilRotTransform(detCtrTanPix)
    tanPix0Pupil = numpy.array(detCtrPupil) - numpy.array(tanPixCtrMinus0Pupil)

    tanPixToPupilAffine = afwGeom.AffineTransform(tanPixToPupilRotMat, numpy.array(tanPix0Pupil))
    pupilToTanPix = afwGeom.AffineXYTransform(tanPixToPupilAffine.invert())
    return afwGeom.MultiXYTransform((pixelToFocalPlane, focalPlaneToPupil, pupilToTanPix))
