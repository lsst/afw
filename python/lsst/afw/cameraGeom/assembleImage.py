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

# dict of doFlip: slice
_SliceDict = {
    False: slice(None,None,1)
    True:  slice(None,None,-1)
}

def assembleAmplifierImage(destImage, rawImage, amplifier):
    """Assemble the amplifier region of an image from a raw image

    @param[in,out] destImage: assembled image (lsst.afw.image.Image or MaskedImage);
        the region amplifier.getBBox() is overwritten with the assembled amplifier image
    @param[in] rawImage: raw image (same type as destImage)
    @param[in] amplifier: amplifier geometry: lsst.afw.cameraGeom.Amplifier with raw amplifier data

    @throw RuntimeError if:
    - image types do not match
    - amplifier has no raw amplifier data
    """
    if not amplifier.hasRawAmplifier():
        raise RuntimeError("amplifier must contain raw amplifier data")
    if type(destImage.Factory) != type(rawImage.Factory):
        raise RuntimeError("destImage type = %s != %s = rawImage type" % \
            type(destImage.Factory).__name__, type(rawImage.Factory).__name__)
    rawAmp = amplifier.getRawAmplifier()
    inView = rawImage.Factory(rawImage, rawAmp.getDataBBox(), False)
    outView = destImage.Factory(destImage, amplifier.getBBox(), False)

    # For the sake of simplicity and robustness, this code does not short-circuit the case flipX=flipY=False.
    # However, it would save a bit of time, including the cost of making numpy array views.
    # If short circuiting is wanted, do it here.

    xSlice = _SliceDict[rawAmp.getFlipX]
    ySlice = _SliceDict[rawAmp.getFlipY]
    if hasattr(rawImage, "getArrays"):
        # MaskedImage
        inArrList = inView.getArrays()
        outArrList = outView.getArrays()
    else:
        inArrList = [inView.getArray()]
        outArrList = [outView.getArray()]

    for inArr, outArr in itertools.izip(inArrList, outArrList):
        outArr[:] = inArr[ySlice, xSlice] # y,x because numpy arrays are transposed w.r.t. afw Images

def assembleAmplifierImage(destImage, rawImage, amplifier):
    """Assemble the amplifier region of a raw CCD image

    For most cameras this is a no-op: the raw image already is an assembled CCD image.
    However, it is useful for camera such as LSST for which each amplifier image is a separate image.

    @param[in,out] destImage: CCD image (lsst.afw.image.Image or MaskedImage);
        the region amplifier.getRawAmplifier().getBBox() is overwritten with the raw amplifier image
    @param[in] rawImage: raw image (same type as destImage)
    @param[in] amplifier: amplifier geometry: lsst.afw.cameraGeom.Amplifier with raw amplifier data

    @throw RuntimeError if:
    - image types do not match
    - amplifier has no raw amplifier data
    """
    if not amplifier.hasRawAmplifier():
        raise RuntimeError("amplifier must contain raw amplifier data")
    if type(destImage.Factory) != type(rawImage.Factory):
        raise RuntimeError("destImage type = %s != %s = rawImage type" % \
            type(destImage.Factory).__name__, type(rawImage.Factory).__name__)
    rawAmp = amplifier.getRawAmplifier()
    inBBox = rawAmp.getBBox()
    inView = rawImage.Factory(rawImage, inBBox(), False)
    outBBox = rawAmp.getBBox()
    outBBox.shift(rawAmp.getXYOffset())
    outView = destImage.Factory(destImage, outBBox(), False)

    outView <<= inView

