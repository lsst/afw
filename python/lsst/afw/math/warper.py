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
import sys
import lsst.pex.logging as pexLog
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import mathLib

__all__ = ["Warper"]

def computeWarpedBBox(destWcs, srcBBox, srcWcs):
    """Compute the bounding box of a warped image
    
    The bounding box includes all warped pixels and it may be a bit oversize.
    
    @param destWcs: WCS of warped exposure
    @param srcBBox: parent bounding box of unwarped image
    @param srcWcs: WCS of unwarped image

    @return destBBox: bounding box of warped exposure
    """
    srcPosBox = afwGeom.Box2D(srcBBox)
    destPosBox = afwGeom.Box2D()
    for inX in (srcPosBox.getMinX(), srcPosBox.getMaxX()):
        for inY in (srcPosBox.getMinY(), srcPosBox.getMaxY()):
            destPos = destWcs.skyToPixel(srcWcs.pixelToSky(afwGeom.Point2D(inX, inY)))
            destPosBox.include(destPos)
    destBBox = afwGeom.Box2I(destPosBox, afwGeom.Box2I.EXPAND)
    return destBBox

class Warper(object):
    """Warp images
    """
    def __init__(self, warpingKernelName, interpLength=10, cacheSize=0):
        """Create a Warper
        
        Inputs:
        - warpingKernelName: argument to lsst.afw.math.makeWarpingKernel
        - interpLength: interpLength argument to lsst.afw.warpExposure
        - cacheSize: size of computeCache
        """
        self._warpingKernel = mathLib.makeWarpingKernel(warpingKernelName)
        self._warpingKernel.computeCache(cacheSize)
        self._interpLength = int(interpLength)

    @classmethod
    def fromPolicy(cls, policy):
        """Create a Warper from a policy
        
        @param policy: see policy/WarpDictionary.paf
        """
        return cls(
            warpingKernelName = policy.getString("warpingKernelName"),
            interpLength = policy.getInt("interpLength"),
            cacheSize = policy.getInt("cacheSize"),
        )

    def warpExposure(self, destWcs, srcExposure, border=0, maxBBox=None):
        """Warp an exposure
        
        @param destWcs: WCS of warped exposure
        @param srcExposure: exposure to warp
        @param border: grow bbox of warped exposure by this amount in all directions (pixels);
            if negative then the bbox is shrunk;
            border is applied before maxBBox
        @param maxBBox: maximum allowed bbox of warped exposure (an afwGeom.Box2I);
            if None then the warped exposure will be just big enough to contain all warped pixels;
            if provided then the warped exposure may be smaller, and so missing some warped pixels

        @return warpedExposure: warped exposure (of same type as srcExposure)
        """
        warpedMaskedImage = self.warpImage(
            destWcs = destWcs,
            srcImage = srcExposure.getMaskedImage(),
            srcWcs = srcExposure.getWcs(),
            border = border, maxBBox = maxBBox,
        )
        return srcExposure.Factory(warpedMaskedImage, destWcs)

    def warpImage(self, destWcs, srcImage, srcWcs, border=0, maxBBox=None):
        """Warp an image or masked image
        
        @param destWcs: WCS of warped image
        @param srcImage: image or masked image to warp
        @param srcWcs: WCS of image
        @param border grow bbox of warped exposure by this amount in all directions (pixels);
            if negative then the bbox is shrunk;
            border is applied before maxBBox
        @param maxBBox: maximum allowed bbox of warped exposure (an afwGeom.Box2I);
            if None then the warped exposure will be just big enough to contain all warped pixels;
            if provided then the warped exposure may be smaller, and so missing some warped pixels

        @return warpedImage: warped image or masked image (of same type as srcImage)
        """
        destBBox = computeWarpedBBox(destWcs, srcImage.getBBox(afwImage.PARENT), srcWcs)
        if border:
            destBBox.grow(border)
        if maxBBox:
            destBBox.clip(maxBBox)
        warpedImage = srcImage.Factory(destBBox)
        mathLib.warpImage(warpedImage, destWcs, srcImage, srcWcs, self._warpingKernel, self._interpLength)
        return warpedImage

