# This file is part of afw.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
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
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__all__ = ["makePixelToTanPixel"]

import lsst.geom
import lsst.afw.geom


def makePixelToTanPixel(bbox, orientation, focalPlaneToField, pixelSizeMm):
    """Make a Transform whose forward direction converts PIXELS to TAN_PIXELS
    for one detector.

    Parameters
    ----------
    bbox : `lsst.geom.Box2I`
        Detector bounding box.
    orientation : `lsst.afw.cameraGeom.Orientation`
        Orientation of detector in focal plane.
    focalPlaneToField : `lsst.afw.geom.TransformPoint2ToPoint2`
        A transform that converts from focal plane (mm) to field angle
        coordinates (radians) in the forward direction.
    pixelSizeMm : `lsst.geom.Extent2D`
        Size of the pixel in mm in X and Y.

    Returns
    -------
    transform : `lsst.afw.geom.TransformPoint2ToPoint2`
        A transform whose forward direction converts PIXELS to TAN_PIXELS.

    Notes
    -----
    PIXELS and TAN_PIXELS are described in the CameraGeom documentation under
    :ref:`camera coordinate systems<section_Camera_Coordinate_Systems>`.
    """
    pixelToFocalPlane = orientation.makePixelFpTransform(pixelSizeMm)
    pixelToField = pixelToFocalPlane.then(focalPlaneToField)
    # fieldToTanPix is affine and matches fieldToPix at field center
    # Note: focal plane to field angle is typically a radial transform,
    # and linearizing the inverse transform of that may fail,
    # so linearize the forward direction instead. (pixelToField is pixelToFocalPlane,
    # an affine transform, followed by focalPlaneToField,
    # so the same consideration applies to pixelToField)
    pixAtFieldCtr = pixelToField.applyInverse(lsst.geom.Point2D(0, 0))
    tanPixToFieldAffine = lsst.afw.geom.linearizeTransform(pixelToField, pixAtFieldCtr)
    fieldToTanPix = lsst.afw.geom.makeTransform(tanPixToFieldAffine.inverted())

    return pixelToField.then(fieldToTanPix)
