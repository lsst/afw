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

__all__ = ["getSipMatrixFromMetadata", "makeDistortedTanWcs", "computePixelToDistortedPixel"]

import lsst.geom
from ..transformFactory import linearizeTransform, makeTransform
from ..skyWcs import makeModifiedWcs
from .wcsUtils import _getSipMatrixFromMetadata


def getSipMatrixFromMetadata(metadata, name):
    """Extract a SIP matrix from FITS TAN-SIP WCS metadata.

    Omitted coefficients are set to 0 and all coefficients may be omitted.

    Parameters
    ----------
    metadata : `lsst.daf.base.PropertyList`
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


def makeDistortedTanWcs(tanWcs, pixelToFocalPlane, focalPlaneToFieldAngle):
    """Compute a WCS that includes a model of optical distortion.

    This is useful in the common case that the initial WCS entirely ignores
    the effect of optical distortion.

    Parameters
    ----------
    tanWcs : `lsst.afw.geom.SkyWcs`
        A pure TAN WCS, such as is usually provided in raw data.
        This should have no existing compensation for optical distortion
        (though it may include an ``ACTUAL_PIXELS`` frame to model pixel-level
        distortions).
    pixelToFocalPlane : `lsst.afw.geom.TransformPoint2ToPoint2`
        Transform parent pixel coordinates to focal plane coordinates.
        This models the location of the CCD on the focal plane
        and is almost always an affine transformation.
        This can be obtained from the detector of an exposure.
    focalPlaneToFieldAngle : `lsst.afw.geom.TransformPoint2ToPoint2`
        Transform focal plane coordinates to field angle coordinates.
        This is a model for optical distortion, and is often a radial
        polynomial. This can be obtained from the camera geometry.


    Returns
    -------
    lsst.afw.geom.SkyWcs
        A copy of `tanWcs` that includes the effect of optical distortion.

    Raises
    ------
    RuntimeError
        If the current frame of `wcs` is not a SkyFrame;
    LookupError
        If 2-dimensional Frames with Domain "PIXELS" and "IWC"
        are not all found.
    """
    # The math is as follows:
    #
    # Our input TAN WCS is:
    #     tanWcs = PIXELS frame -> pixelToIwc -> IWC frame ->  iwcToSky -> SkyFrame
    # See lsst.afw.geom.SkyWcs for a description of these frames.
    # tanWcs may also contain an ACTUAL_PIXELS frame before the PIXELS frame;
    # if so it will be preserved, but it is irrelevant to the computation
    # and so not discussed further.
    #
    # Our desired WCS must still contain the PIXELS and IWC frames.
    # The distortion will be inserted just after the PIXELS frame,
    # So the new WCS will be as follows:
    #     wcs = PIXELS frame -> pixelToDistortedPixel -> pixelToIwc -> IWC frame -> iwcToSky -> SkyFrame
    #
    # We compute pixelToDistortedPixel as follows...
    #
    # We will omit the frames from now on, for simplicity. Thus:
    #     tanWcs = pixelToIwc -> iwcToSksy
    # and:
    #     wcs =  pixelToDistortedPixel -> pixelToIwc -> iwcToSky
    #
    # We also know pixelToFocalPlane and focalPlaneToFieldAngle,
    # and can use them as follows:
    #
    # The tan WCS can be expressed as:
    #     tanWcs = pixelToFocalPlane -> focalPlaneToTanFieldAngle -> fieldAngleToIwc -> iwcToSky
    # where:
    #     - focalPlaneToTanFieldAngle is the linear approximation to
    #       focalPlaneToFieldAngle at the center of the focal plane
    #
    # The desired WCS can be expressed as:
    #     wcs = pixelToFocalPlane -> focalPlaneToFieldAngle -> fieldAngleToIwc -> iwcToSky
    #
    # By equating the two expressions for tanWcs, we get:
    #     pixelToIwc = pixelToFocalPlane -> focalPlaneToTanFieldAngle -> fieldAngleToIwc
    #     fieldAngleToIwc = tanFieldAngleToFocalPlane -> focalPlaneToPixel -> pixelToIwc
    #
    # By equating the two expressions for desired wcs we get:
    #     pixelToDistortedPixel -> pixelToIwc = pixelToFocalPlane -> focalPlaneToFieldAngle -> fieldAngleToIwc
    #
    # Substitute our expression for fieldAngleToIwc from tanWcs into the
    # previous equation, we get:
    #     pixelToDistortedPixel -> pixelToIwc
    #         = pixelToFocalPlane -> focalPlaneToFieldAngle -> tanFieldAngleToFocalPlane -> focalPlaneToPixel
    #           -> pixelToIwc
    #
    # Thus:
    #     pixelToDistortedPixel
    #         = pixelToFocalPlane -> focalPlaneToFieldAngle -> tanFieldAngleToFocalPlane -> focalPlaneToPixel

    pixelToDistortedPixel = computePixelToDistortedPixel(pixelToFocalPlane, focalPlaneToFieldAngle)
    return makeModifiedWcs(pixelTransform=pixelToDistortedPixel, wcs=tanWcs, modifyActualPixels=False)


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
