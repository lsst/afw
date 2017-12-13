from __future__ import absolute_import, division, print_function
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
import lsst.afw.geom as afwGeom

__all__ = ["makePixelToTanPixel"]


def makePixelToTanPixel(bbox, orientation, focalPlaneToField, pixelSizeMm):
    """!Make a Transform whose forward direction converts PIXELS to TAN_PIXELS for one detector

    PIXELS and TAN_PIXELS are defined in @ref afwCameraGeomCoordSys in doc/cameraGeom.dox

    @param[in] bbox  detector bounding box (an lsst.afw.geom.Box2I)
    @param[in] orientation  orientation of detector in focal plane (an lsst.afw.cameraGeom.Orientation)
    @param[in] focalPlaneToField  an lsst.afw.geom.Transform that converts from focal plane (mm)
        to field angle coordinates (radians) in the forward direction
    @param[in] pixelSizeMm  size of the pixel in mm in X and Y (an lsst.afw.geom.Extent2D)
    @return a TransformPoint2ToPoint2 whose forward direction converts PIXELS to TAN_PIXELS
    """
    pixelToFocalPlane = orientation.makePixelFpTransform(pixelSizeMm)
    pixelToField = pixelToFocalPlane.then(focalPlaneToField)
    # fieldToTanPix is affine and matches fieldToPix at field center
    # Note: focal plane to field angle is typically a radial transform,
    # and linearizing the inverse transform of that may fail,
    # so linearize the forward direction instead. (pixelToField is pixelToFocalPlane,
    # an affine transform, followed by focalPlaneToField,
    # so the same consideration applies to pixelToField)
    pixAtFieldCtr = pixelToField.applyInverse(afwGeom.Point2D(0, 0))
    tanPixToFieldAffine = afwGeom.linearizeTransform(pixelToField,
                                                     pixAtFieldCtr)
    fieldToTanPix = afwGeom.makeTransform(tanPixToFieldAffine.invert())

    return pixelToField.then(fieldToTanPix)
