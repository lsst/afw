from __future__ import absolute_import, division
# 
# LSST Data Management System
# Copyright 2014 LSST Corporation.
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
import numpy
from lsst.afw.geom import Box2D, AffineTransform, AffineXYTransform, MultiXYTransform

__all__ = ["makePixelToTanPixel"]

def makePixelToTanPixel(bbox, orientation, focalPlaneToPupil, pixelSizeMm, plateScale):
    """Make an XYTransform whose forward direction converts PIXEL to TAN_PIXEL for one detector

    @param[in] bbox: detector bounding box
    @param[in] orientation: orientation of detector in focal plane
    @param[in] focalPlaneToPupil: XYTransform that converts from focal plane (mm)
        to pupil coordinates (arcsec) in the forward direction
    @param[in] pixelSizeMm: size of the pixel in mm in X and Y
    @param[in] plateScale: plate scale of the camera in arcsec/mm
    """
    pixelToFocalPlane = orientation.makePixelFpTransform(pixelSizeMm)

    meanPixelSizeMm = (pixelSizeMm[0] + pixelSizeMm[1]) / 2.0
    arcSecPerMeanPix = plateScale * meanPixelSizeMm   # arcsec/mm * mm/pix

    detCtrPix = Box2D(bbox).getCenter()
    detCtrTanPix = detCtrPix # by definition

    detCtrPupil = focalPlaneToPupil.forwardTransform(pixelToFocalPlane.forwardTransform(detCtrPix))

    pupilTanPixAngRad = -orientation.getYaw().asRadians()
    pupilTanPixSin = math.sin(pupilTanPixAngRad)
    pupilTanPixCos = math.cos(pupilTanPixAngRad)
    tanPixToPupilRotMat = numpy.array((
        (pupilTanPixCos, pupilTanPixSin),
        (-pupilTanPixSin, pupilTanPixCos),
    )) * arcSecPerMeanPix
    tanPixToPupilRotTransform = AffineTransform(tanPixToPupilRotMat)

    tanPixCtrMinus0Pupil = tanPixToPupilRotTransform(detCtrTanPix)
    tanPix0Pupil = numpy.array(detCtrPupil) - numpy.array(tanPixCtrMinus0Pupil)

    tanPixToPupilAffine = AffineTransform(tanPixToPupilRotMat, numpy.array(tanPix0Pupil))
    pupilToTanPix = AffineXYTransform(tanPixToPupilAffine.invert())
    return MultiXYTransform((pixelToFocalPlane, focalPlaneToPupil, pupilToTanPix))
