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

__all__ = ["getSipMatrixFromMetadata", "computePixelToDistortedPixel"]

import lsst.geom
from ..transformFactory import linearizeTransform, makeTransform
from .wcsUtils import _getSipMatrixFromMetadata


def getSipMatrixFromMetadata(metadata, name):
    """Extract a SIP matrix from FITS TAN-SIP WCS metadata.

    Omitted coefficients are set to 0 and all coefficients may be omitted.

    Parameters
    ----------
    metadata : `lsst.daf.base.PropertySet`
        FITS metadata.
    name : `str`
        Name of TAN-SIP matrix (``"A"``, ``"B"``, ``"Ap"``, or ``"Bp"``).

    Returns
    -------
    `numpy.array`
        The SIP matrix.

    Raises
    ------
    TypeError
        If the order keyword ``<name>_ORDER`` (e.g. ``AP_ORDER``) is not found,
        the value of the order keyword cannot be read as an integer,
        the value of the order keyword is negative,
        or if a matrix parameter (e.g. ``AP_5_0``) cannot be read as a float.
    """
    arr = _getSipMatrixFromMetadata(metadata, name)
    if arr.shape == ():  # order=0
        arr.shape = (1, 1)
    return arr


def computePixelToDistortedPixel(pixelToFocalPlane, focalPlaneToFieldAngle):
    """Compute the transform ``pixelToDistortedPixel``, which applies optical
    distortion specified by ``focalPlaneToFieldAngle``.

    The resulting transform is designed to be used to convert a pure TAN WCS
    to a WCS that includes a model for optical distortion. In detail,
    the initial WCS will contain these frames and transforms::

        PIXELS frame -> pixelToIwc -> IWC frame ->  gridToIwc -> SkyFrame

    To produce the WCS with distortion, replace ``pixelToIwc`` with::

        pixelToDistortedPixel -> pixelToIwc

    Parameters
    ----------
    pixelToFocalPlane : `lsst.afw.geom.TransformPoint2ToPoint2`
        Transform parent pixel coordinates to focal plane coordinates
    focalPlaneToFieldAngle : `lsst.afw.geom.TransformPoint2ToPoint2`
        Transform focal plane coordinates to field angle coordinates

    Returns
    -------
    pixelToDistortedPixel : `lsst.afw.geom.TransformPoint2ToPoint2`
        A transform that applies the effect of the optical distortion model.
    """
    # return pixelToFocalPlane -> focalPlaneToFieldAngle -> tanFieldAngleToocalPlane -> focalPlaneToPixel
    focalPlaneToTanFieldAngle = makeTransform(linearizeTransform(focalPlaneToFieldAngle,
                                                                 lsst.geom.Point2D(0, 0)))
    return pixelToFocalPlane.then(focalPlaneToFieldAngle) \
        .then(focalPlaneToTanFieldAngle.inverted()) \
        .then(pixelToFocalPlane.inverted())
