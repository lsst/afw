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

__all__ = ['assembleAmplifierImage', 'assembleAmplifierRawImage',
           'makeUpdatedDetector']

import lsst.geom

# dict of doFlip: slice
_SliceDict = {
    False: slice(None, None, 1),
    True: slice(None, None, -1),
}


def _insertPixelChunk(outView, inView, amplifier, hasArrays, refAmplifier=None):
    """Copy pixels to outView from inView, respecting amplifier flips.

    Parameters
    ----------
    outView : `lsst.afw.image.Image`
       Image to copy to.
    inView : `lsst.afw.image.Image`
       Image to copy from.
    amplifier : `lsst.afw.cameraGeom.Amplifier`
       Amplifier for input geometry.
    hasArrays : `bool`
       Are there multiple image arrays to copy?
    refAmplifier : `lsst.afw.cameraGeom.Amplifier`, optional
       Amplifier to match the orientation of

    Notes
    -----
    For the sake of simplicity and robustness, this code does not
    short-circuit the case flipX=flipY=False.  However, it would save
    a bit of time, including the cost of making numpy array views.  If
    short circuiting is wanted, do it here.
    """
    if refAmplifier is None:
        xSlice = _SliceDict[amplifier.getRawFlipX()]
        ySlice = _SliceDict[amplifier.getRawFlipY()]
    else:
        xSlice = _SliceDict[amplifier.getRawFlipX() ^ refAmplifier.getRawFlipX()]
        ySlice = _SliceDict[amplifier.getRawFlipY() ^ refAmplifier.getRawFlipY()]

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

# def amplifierViewAsReference(rawImage, amplifier, refAmplifier):
#     outView = rawImage.Factory(amplifier.getBBox())
#     xSlice = _SliceDict[amplifier.getRawFlipX() ^ refAmplifier.getRawFlipX()]
#     ySlice = _SliceDict[amplifier.getRawFlipY() ^ refAmplifier.getRawFlipY()]

#     if hasattr(rawImage, "getArrays"):

# def assembleAmplifier(destImage, rawImage, amplifier, assemblyState, repair=False):
#     """Assemble an image to the desired assembly state.

#     Parameters
#     ----------
#     destImage : `lsst.afw.image.Image`
#        Output image.
#     rawImage : `lsst.afw.image.Image`
#        Input image (same type as destImage).
#     amplifier : `lsst.afw.cameraGeom.Amplifier`
#        Amplifier with input camera geometry.
#     assemblyState : `lsst.afw.cameraGeom.AssemblyState`
#        State to assemble the destImage.
#     repair : `bool`, optional
#        Attempt to fix inconsistent geometry information.
#     """
#     currentAssemblyState = amplifier.getAssemblyState()
#     if assemblyState == currentAssemblyState:
#         destImage = rawImage
#         return amplifier
#     elif assemblyState < currentAssemblyState:
#         raise f"Cannot assemble amplifier to earlier state: {assemblyState} {amplifier.getAssemblyState()}"
#     else:
#         outAmp = amplifier.rebuild()
#         if currentAssemblyState == AssemblyState.SPLIT:
#             if assemblyState == AssemblyState.RAW:
#                 assembleAmplifierRawImage(destImage, rawImage, amplifier)
#             elif assemblyState == AssemblyState.SCIENCE:
#                 assembleAmplifierImage(destImage, rawImage, amplifier)
#         elif currentAssemblyState == AssemblyState.RAW:
#             if assemblyState == AssemblyState.SCIENCE:
#                 assembleAmplifierImage(destImage, rawImage, amplifier)


def assembleAmplifierImage(destImage, rawImage, amplifier):
    """Assemble the amplifier region of an image from a raw image.

    Parameters
    ----------
    destImage : `lsst.afw.image.Image` or `lsst.afw.image.MaskedImage`
        Assembled image; the region amplifier.getBBox() is overwritten with
        the assembled amplifier image.
    rawImage : `lsst.afw.image.Image` or `lsst.afw.image.MaskedImage`
        Raw image (same type as destImage).
    amplifier : `lsst.afw.cameraGeom.Amplifier`
        Amplifier geometry, with raw amplifier info.

    Raises
    ------
    RuntimeError
        Raised if image types do not match or amplifier has no raw amplifier info.
    """
    if not amplifier.getHasRawInfo():
        raise RuntimeError("amplifier must contain raw amplifier info")
    if type(destImage.Factory) != type(rawImage.Factory):  # noqa: E721
        raise RuntimeError("destImage type = %s != %s = rawImage type" %
                           type(destImage.Factory).__name__, type(rawImage.Factory).__name__)
    inView = rawImage.Factory(rawImage, amplifier.getRawDataBBox())
    outView = destImage.Factory(destImage, amplifier.getBBox())

    _insertPixelChunk(outView, inView, amplifier,
                      hasattr(rawImage, "getArrays"))


def assembleAmplifierRawImage(destImage, rawImage, amplifier):
    """Assemble the amplifier region of a raw CCD image.

    For most cameras this is a no-op: the raw image already is an assembled
    CCD image.
    However, it is useful for camera such as LSST for which each amplifier
    image is a separate image.

    Parameters
    ----------
    destImage : `lsst.afw.image.Image` or `lsst.afw.image.MaskedImage`
        CCD Image; the region amplifier.getRawAmplifier().getBBox()
        is overwritten with the raw amplifier image.
    rawImage : `lsst.afw.image.Image` or `lsst.afw.image.MaskedImage`
        Raw image (same type as destImage).
    amplifier : `lsst.afw.cameraGeom.Amplifier`
        Amplifier geometry with raw amplifier info

    Raises
    ------
    RuntimeError
        Raised if image types do not match or amplifier has no raw amplifier info.
    """
    if not amplifier.getHasRawInfo():
        raise RuntimeError("amplifier must contain raw amplifier info")
    if type(destImage.Factory) != type(rawImage.Factory):  # noqa: E721
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
    """Return a Detector that has had the definitions of amplifier geometry
    updated post assembly.

    Parameters
    ----------
    ccd : `lsst.afw.image.Detector`
        The detector to copy and update.
    """
    builder = ccd.rebuild()
    for amp in builder.getAmplifiers():
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
        amp.setRawXYOffset(lsst.geom.ExtentI(0, 0))
        amp.setRawFlipX(False)
        amp.setRawFlipY(False)

    return builder.finish()
