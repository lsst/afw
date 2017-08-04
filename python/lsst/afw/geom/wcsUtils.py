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
from __future__ import absolute_import, division, print_function

__all__ = ["makeDistortedTanWcs", "computePixelToDistortedPixel"]

import astshim as ast
from lsst.afw.geom import Point2D, linearizeTransform, makeTransform, SkyWcs


def findFrame2ByDomain(frameSet, domain):
    """Find a 2-axis Frame given its domain name

    Set that frame to current and return its index if found,
    else throw LookupError
    """
    template = ast.Frame(2, "Domain={}".format(domain))
    if frameSet.findFrame(template) is None:
        raise LookupError("Could not find a 2-axis frame with Domain={}".format(domain))
    return frameSet.current


def makeDistortedTanWcs(tanWcs, pixelToFocalPlane, focalPlaneToPupil):
    """
    Compute a WCS that includes a model of optical distortion.

    Parameters
    ----------
    tanWcs : lsst.afw.geom.SkyWcs
        A pure TAN WCS, such as is usually provided in raw data.
        This should have no existing compensation for optical distortion
        (though it may include an ACTUALPIXEL0 frame to model pixel-level
        distortions).
    pixelToFocalPlane : lsst.afw.geom.TransformPoint2ToPoint2
        Transform parent pixel coordinates to focal plane coordinates
    focalPlaneToPupil : lsst.afw.geom.TransformPoint2ToPoint2
        Transform focal plane coordinates to pupil coordinates

    Returns
    -------
    lsst.afw.geom.SkyWcs
        A copy of `tanWcs` that includes the effect of optical distortion.
        The initial IWC frame's Domain is renamed from "IWC" to "TANIWC".
        The initial (current) SkyFrame has its domain set to "TANSKY".
        A new IWC frame is added: a copy of the old, with domain "IWC".
        A new SkyFrame is added: a copy of the old, with domain "SKY",
        and this will be the current frame.
        Thus the new WCS contains the original transformation,
        but is set to use the new transformation.

    Raises
    ------
    RuntimeError
        If the current frame of `wcs` is not a SkyFrame;
        its domain does not matter, as long as it is not "TANWCS".
    RuntimeError
        If a SkyFrame with domain "TANWCS" is found.
    LookupError
        If 2-dimensional Frames with Domain "PIXEL0", "GRID" and "IWCS"
        are not all found.

    Notes
    -----
    The math is as follows:

    Our input TAN WCS is:
        tanWcs = PIXEL0 frame -> pixelToGrid -> GRID frame
                 -> gridToIwc -> IWC frame ->  iwcToSky -> SkyFrame
    See lsst.afw.geom.SkyWcs for a description of these frames.
    tanWcs may also contain an ACTUALPIXEL0 frame before the PIXEL0 frame;
    it is irrelevant to the computation, and so not discussed further.

    Our desired WCS must still contain the GRID and IWC frames,
    so it will be as follows:
        wcs = PIXEL0 frame -> pixelToGrid -> gridToDistortedGrid -> GRID frame
              -> gridToIwc -> IWC frame -> iwcToSky -> SkyFrame

    We will omit the frames from now on, for simplicity. Thus:
        tanWcs = pixelToGrid -> gridToIwc -> iwcToSksy
    and:
        wcs = pixelToGrid -> gridToDistortedGrid -> gridToIwc -> iwcToSky

    We also know transforms pixelToFocalPlane and focalPlaneToPupil,
    and can use them as follows:

    The tan WCS can be expressed as:
        tanWcs = pixelToFocalPlane -> focalPlaneToTanPupil -> pupilToIwc
                 -> iwcToSky
    where:
        - focalPlaneToTanPupil is focalPlaneToPupil evaluated at the center
          of the focal plane

    The desired WCS can be expressed as:
        wcs = pixelToFocalPlane -> focalPlaneToPupil -> pupilToIwc -> iwcToSky

    By equating the two expressions for tanWcs, we get:
        pixelToGrid -> gridToIwc = pixelToFocalPlane -> focalPlaneToTanPupil
                                   -> pupilToIwc
        pupilToIwc = tanPupilToFocalPlane -> focalPlaneToPixel -> pixelToGrid
                     -> gridToIwc

    By equating the two expressions for desired wcs we get:
        pixelToGrid -> gridToDistortedGrid -> gridToIwc
            = pixelToFocalPlane -> focalPlaneToPupil -> pupilToIwc

    Substitute our expression for pupilToIcs into the latter:
        pixelToGrid -> gridToDistortedGrid -> gridToIwc
            = pixelToFocalPlane -> focalPlaneToPupil -> tanPupilToFocalPlane
              -> focalPlaneToPixel -> pixelToGrid -> gridToIwc

    Thus:
        pixelToGrid -> gridToDistortedGrid
            = pixelToFocalPlane -> focalPlaneToPupil -> tanPupilToFocalPlane
              -> focalPlaneToPixel -> pixelToGrid
    and:
        gridToDistortedGrid
            = gridToPixel -> pixelToFocalPlane -> focalPlaneToPupil
              -> tanPupilToFocalPlane -> focalPlaneToPixel -> pixelToGrid
    Thus:
        gridToDistortedGrid
            = gridToPixel -> pixelToDistortedPixel -> pixelToGrid
    where:
        pixelToDistortedPixel
            = pixelToFocalPlane -> focalPlaneToPupil -> tanPupilToFocalPlane
              -> focalPlaneToPixel
    """
    frameSet = tanWcs.getFrameSet()
    skyFrame = frameSet.getFrame(frameSet.CURRENT)
    assert isinstance(skyFrame, ast.SkyFrame)
    tanSkyIndex = frameSet.current

    # make sure that a SkyFrame with domain SKYWCS does not already exist
    template = ast.SkyFrame("Domain=TANSKY")
    if frameSet.findFrame(template) is not None:
        raise RuntimeError("tanWcs already has distortion; it contains a sky frame with Domain=TANWCS")

    skyFrame.domain = "SKY"  # use this copy for sky with distortion
    frameSet.current = tanSkyIndex
    frameSet.domain = "TANSKY"

    pixelIndex = findFrame2ByDomain(frameSet, "PIXEL0")
    gridIndex = findFrame2ByDomain(frameSet, "GRID")
    tanIwcIndex = findFrame2ByDomain(frameSet, "IWC")
    iwcFrame = frameSet.getFrame(tanIwcIndex)  # a copy to use for IWC with distortion
    frameSet.domain = "TANIWC"

    pixelToGrid = frameSet.getMapping(pixelIndex, gridIndex)
    gridToIwc = frameSet.getMapping(gridIndex, tanIwcIndex)
    iwcToSky = frameSet.getMapping(tanIwcIndex, tanSkyIndex)

    pixelToDistortedPixel = computePixelToDistortedPixel(pixelToFocalPlane, focalPlaneToPupil)
    gridToDistortedGrid = pixelToGrid.getInverse().then(pixelToDistortedPixel).then(pixelToGrid)
    frameSet.addMapping(gridIndex, gridToDistortedGrid.then(gridToIwc), iwcFrame)
    frameSet.addMapping(frameSet.CURRENT, iwcToSky, skyFrame)
    return SkyWcs(frameSet)


def computePixelToDistortedPixel(pixelToFocalPlane, focalPlaneToPupil):
    """
    Compute the transform pixelToDistortedPixel, which applies optical distortion specified focalPlaneToPupil

    The resulting transform is designed to be used to convert a pure TAN WCS to a WCS that includes
    a model for optical distortion. In detail, the initial WCS will contain these frames and transforms:
        PIXEL0 frame -> pixelToGrid -> GRID frame -> gridToIwc -> IWC frame ->  gridToIwc -> SkyFrame
    To produce the WCS with distortion, replace gridToIwc with:
        gridToPixel -> pixelToDistortedPixel -> pixelToGrid -> gridToIWC

    Parameters
    ----------
    pixelToFocalPlane : lsst.afw.geom.TransformPoint2ToPoint2
        Transform parent pixel coordinates to focal plane coordinates
    focalPlaneToPupil : lsst.afw.geom.TransformPoint2ToPoint2
        Transform focal plane coordinates to pupil coordinates

    Returns
    -------
    pixelToDistortedPixel : lsst.afw.geom.TransformPoint2ToPoint2
        A transform that applies the effect of the optical distortion model.
    """
    # return pixelToFocalPlane -> focalPlaneToPupil -> tanPupilToocalPlane -> focalPlaneToPixel
    tanFocalPlaneToPupil = makeTransform(linearizeTransform(focalPlaneToPupil, Point2D(0, 0)))
    return pixelToFocalPlane.then(focalPlaneToPupil) \
        .then(tanFocalPlaneToPupil.getInverse()) \
        .then(pixelToFocalPlane.getInverse())
