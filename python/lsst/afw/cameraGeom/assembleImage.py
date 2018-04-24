#
# LSST Data Management System
# Copyright 2008, 2009, 2010 LSST Corporation.
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
from . import copyDetector

__ALL__ = ['assembleAmplifierImage', 'assembleAmplifierRawImage',
           'updateAmpGeometryForAssembledCcd']

# dict of doFlip: slice
_SliceDict = {
    False: slice(None, None, 1),
    True: slice(None, None, -1),
}


def _insertPixelChunk(outView, inView, amplifier, hasArrays):
    # For the sake of simplicity and robustness, this code does not short-circuit the case flipX=flipY=False.
    # However, it would save a bit of time, including the cost of making numpy array views.
    # If short circuiting is wanted, do it here.

    xSlice = _SliceDict[amplifier.getRawFlipX()]
    ySlice = _SliceDict[amplifier.getRawFlipY()]
    if hasArrays:
        # MaskedImage
        inArrList = inView.getArrays()
        outArrList = outView.getArrays()
    else:
        inArrList = [inView.getArray()]
        outArrList = [outView.getArray()]

    for inArr, outArr in zip(inArrList, outArrList):
        # y,x because numpy arrays are transposed w.r.t. afw Images
        outArr[:] = inArr[ySlice, xSlice]


def assembleAmplifierImage(destImage, rawImage, amplifier):
    """!Assemble the amplifier region of an image from a raw image

    @param[in,out] destImage  assembled image (lsst.afw.image.Image or MaskedImage);
        the region amplifier.getBBox() is overwritten with the assembled amplifier image
    @param[in] rawImage  raw image (same type as destImage)
    @param[in] amplifier  amplifier geometry: lsst.afw.cameraGeom.Amplifier with raw amplifier info

    @throw RuntimeError if:
    - image types do not match
    - amplifier has no raw amplifier info
    """
    if not amplifier.getHasRawInfo():
        raise RuntimeError("amplifier must contain raw amplifier info")
    if type(destImage.Factory) != type(rawImage.Factory):  # noqa E721
        raise RuntimeError("destImage type = %s != %s = rawImage type" %
                           type(destImage.Factory).__name__, type(rawImage.Factory).__name__)
    inView = rawImage.Factory(rawImage, amplifier.getRawDataBBox())
    outView = destImage.Factory(destImage, amplifier.getBBox())

    _insertPixelChunk(outView, inView, amplifier,
                      hasattr(rawImage, "getArrays"))


def assembleAmplifierRawImage(destImage, rawImage, amplifier):
    """!Assemble the amplifier region of a raw CCD image

    For most cameras this is a no-op: the raw image already is an assembled CCD image.
    However, it is useful for camera such as LSST for which each amplifier image is a separate image.

    @param[in,out] destImage  CCD image (lsst.afw.image.Image or MaskedImage);
        the region amplifier.getRawAmplifier().getBBox() is overwritten with the raw amplifier image
    @param[in] rawImage  raw image (same type as destImage)
    @param[in] amplifier  amplifier geometry: lsst.afw.cameraGeom.Amplifier with raw amplifier info

    @throw RuntimeError if:
    - image types do not match
    - amplifier has no raw amplifier info
    """
    if not amplifier.getHasRawInfo():
        raise RuntimeError("amplifier must contain raw amplifier info")
    if type(destImage.Factory) != type(rawImage.Factory):  # noqa E721
        raise RuntimeError("destImage type = %s != %s = rawImage type" %
                           type(destImage.Factory).__name__, type(rawImage.Factory).__name__)
    inBBox = amplifier.getRawBBox()
    inView = rawImage.Factory(rawImage, inBBox)
    outBBox = amplifier.getRawBBox()
    outBBox.shift(amplifier.getRawXYOffset())
    outView = destImage.Factory(destImage, outBBox)

    _insertPixelChunk(outView, inView, amplifier,
                      hasattr(rawImage, "getArrays"))


def makeUpdatedDetector(ccd):
    """Return a Detector that has had the definitions of amplifier geometry updated post assembly
    """
    ampInfoCatalog = ccd.getAmpInfoCatalog().copy(deep=True)

    for amp in ampInfoCatalog:
        assert amp.getHasRawInfo()

        bbox = amp.getRawBBox()
        awidth, aheight = bbox.getDimensions()
        #
        # Figure out how far flipping the amp LR and/or TB offsets the bboxes
        #
        boxMin0 = bbox.getMin()     # initial position of rawBBox's LLC corner
        if amp.getRawFlipX():
            bbox.flipLR(awidth)
        if amp.getRawFlipY():
            bbox.flipTB(aheight)
        shift = boxMin0 - bbox.getMin()

        for bboxName in ("",
                         "HorizontalOverscan",
                         "Data",
                         "VerticalOverscan",
                         "Prescan"):
            bbox = getattr(amp, "getRaw%sBBox" % bboxName)()
            if amp.getRawFlipX():
                bbox.flipLR(awidth)
            if amp.getRawFlipY():
                bbox.flipTB(aheight)
            bbox.shift(amp.getRawXYOffset() + shift)

            getattr(amp, "setRaw%sBBox" % bboxName)(bbox)
        #
        # All of these have now been transferred to the per-amp geometry
        #
        amp.setRawXYOffset(afwGeom.ExtentI(0, 0))
        amp.setRawFlipX(False)
        amp.setRawFlipY(False)

    return copyDetector(ccd, ampInfoCatalog)
