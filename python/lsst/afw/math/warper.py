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

__all__ = ["Warper", "WarperConfig"]

import lsst.pex.config as pexConfig
import lsst.geom
import lsst.afw.image as afwImage
from . import mathLib


def computeWarpedBBox(destWcs, srcBBox, srcWcs):
    """Compute the bounding box of a warped image.

    The bounding box includes all warped pixels and it may be a bit oversize.

    Parameters
    ----------
    destWcs : `lsst.afw.geom.SkyWcs`
        WCS of warped exposure
    srcBBox : `lsst.geom.Box2I`
        parent bounding box of unwarped image
    srcWcs : `lsst.afw.geom.SkyWcs`
        WCS of unwarped image

    Returns
    -------
    destBBox: `lsst.geom.Box2I`
        bounding box of warped exposure
    """
    srcPosBox = lsst.geom.Box2D(srcBBox)
    destPosBox = lsst.geom.Box2D()
    for inX in (srcPosBox.getMinX(), srcPosBox.getMaxX()):
        for inY in (srcPosBox.getMinY(), srcPosBox.getMaxY()):
            destPos = destWcs.skyToPixel(srcWcs.pixelToSky(inX, inY))
            destPosBox.include(destPos)
    destBBox = lsst.geom.Box2I(destPosBox, lsst.geom.Box2I.EXPAND)
    return destBBox


_DefaultInterpLength = 10
_DefaultCacheSize = 1000000


class WarperConfig(pexConfig.Config):
    warpingKernelName = pexConfig.ChoiceField(
        dtype=str,
        doc="Warping kernel",
        default="lanczos3",
        allowed={
            "bilinear": "bilinear interpolation",
            "lanczos3": "Lanczos kernel of order 3",
            "lanczos4": "Lanczos kernel of order 4",
            "lanczos5": "Lanczos kernel of order 5",
        }
    )
    maskWarpingKernelName = pexConfig.ChoiceField(
        dtype=str,
        doc="Warping kernel for mask (use ``warpingKernelName`` if '')",
        default="bilinear",
        allowed={
            "": "use the regular warping kernel for the mask plane, as well as the image and variance planes",
            "bilinear": "bilinear interpolation",
            "lanczos3": "Lanczos kernel of order 3",
            "lanczos4": "Lanczos kernel of order 4",
            "lanczos5": "Lanczos kernel of order 5",
        }
    )
    interpLength = pexConfig.Field(
        dtype=int,
        doc="``interpLength`` argument to `lsst.afw.math.warpExposure`",
        default=_DefaultInterpLength,
    )
    cacheSize = pexConfig.Field(
        dtype=int,
        doc="``cacheSize`` argument to `lsst.afw.math.SeparableKernel.computeCache`",
        default=_DefaultCacheSize,
    )
    growFullMask = pexConfig.Field(
        dtype=int,
        doc="mask bits to grow to full width of image/variance kernel,",
        default=afwImage.Mask.getPlaneBitMask("EDGE"),
    )


class Warper:
    """Warp images.

    Parameters
    ----------
    warpingKernelName : `str`
        see `WarperConfig.warpingKernelName`
    interpLength : `int`, optional
        ``interpLength`` argument to `lsst.afw.math.warpExposure`
    cacheSize : `int`, optional
        size of computeCache
    maskWarpingKernelName : `str`, optional
        name of mask warping kernel (if ``""`` then use ``warpingKernelName``);
        see `WarperConfig.maskWarpingKernelName`
    growFullMask : `int`, optional
        mask bits to grow to full width of image/variance kernel
    """
    ConfigClass = WarperConfig

    def __init__(self,
                 warpingKernelName,
                 interpLength=_DefaultInterpLength,
                 cacheSize=_DefaultCacheSize,
                 maskWarpingKernelName="",
                 growFullMask=afwImage.Mask.getPlaneBitMask("EDGE"),):
        self._warpingControl = mathLib.WarpingControl(
            warpingKernelName, maskWarpingKernelName, cacheSize, interpLength, growFullMask)

    @classmethod
    def fromConfig(cls, config):
        """Create a Warper from a config.

        Parameters
        ----------
        config : `WarperConfig`
            The config to initialize the Warper with.
        """
        return cls(
            warpingKernelName=config.warpingKernelName,
            maskWarpingKernelName=config.maskWarpingKernelName,
            interpLength=config.interpLength,
            cacheSize=config.cacheSize,
            growFullMask=config.growFullMask,
        )

    def getWarpingKernel(self):
        """Get the warping kernel.
        """
        return self._warpingControl.getWarpingKernel()

    def getMaskWarpingKernel(self):
        """Get the mask warping kernel.
        """
        return self._warpingControl.getMaskWarpingKernel()

    def warpExposure(self, destWcs, srcExposure, border=0, maxBBox=None, destBBox=None):
        """Warp an exposure.

        Parameters
        -----------
        destWcs : `lsst.afw.geom.SkyWcs`
            WCS of warped exposure
        srcExposure
            exposure to warp
        border : `int`, optional
            grow bbox of warped exposure by this amount in all directions
            (in pixels); if negative then the bbox is shrunk; border is applied
            before ``maxBBox``; ignored if ``destBBox`` is not `None`
        maxBBox : `lsst.geom.Box2I`, optional
            maximum allowed parent bbox of warped exposure; if `None` then the
            warped exposure will be just big enough to contain all warped pixels;
            if provided then the warped exposure may be smaller, and so
            missing some warped pixels; ignored if ``destBBox`` is not `None`
        destBBox : `lsst.geom.Box2I`, optional
            exact parent bbox of warped exposure; if `None` then ``border`` and
            ``maxBBox`` are used to determine the bbox, otherwise ``border``
            and ``maxBBox`` are ignored

        Returns
        -------
        destExposure : same type as ``srcExposure``
            warped exposure

        Notes
        -----
        calls `lsst.afw.math.warpExposure` insted of `~Warper.warpImage` because the former
        copies attributes such as ``Calib``, and that should be done in one place
        """
        destBBox = self._computeDestBBox(
            destWcs=destWcs,
            srcImage=srcExposure.getMaskedImage(),
            srcWcs=srcExposure.getWcs(),
            border=border,
            maxBBox=maxBBox,
            destBBox=destBBox,
        )
        destExposure = srcExposure.Factory(destBBox, destWcs)
        mathLib.warpExposure(destExposure, srcExposure, self._warpingControl)
        return destExposure

    def warpImage(self, destWcs, srcImage, srcWcs, border=0, maxBBox=None, destBBox=None):
        """Warp an image or masked image.

        Parameters
        ----------
        destWcs : `lsst.afw.geom.SkyWcs`
            WCS of warped image
        srcImage
            image or masked image to warp
        srcWcs : `lsst.afw.geom.SkyWcs`
            WCS of image
        border : `int`, optional
            grow bbox of warped image by this amount in all directions
            (in pixels); if negative then the bbox is shrunk; border is applied
            before ``maxBBox``; ignored if ``destBBox`` is not `None`
        maxBBox : `lsst.geom.Box2I`, optional
            maximum allowed parent bbox of warped image; if `None` then the
            warped image will be just big enough to contain all warped pixels;
            if provided then the warped image may be smaller, and so
            missing some warped pixels; ignored if ``destBBox`` is not `None`
        destBBox : `lsst.geom.Box2I`, optional
            exact parent bbox of warped image; if `None` then ``border`` and
            ``maxBBox`` are used to determine the bbox, otherwise ``border``
            and ``maxBBox`` are ignored

        Returns
        -------
        destImage : same type as ``srcExposure``
            warped image or masked image
        """
        destBBox = self._computeDestBBox(
            destWcs=destWcs,
            srcImage=srcImage,
            srcWcs=srcWcs,
            border=border,
            maxBBox=maxBBox,
            destBBox=destBBox,
        )
        destImage = srcImage.Factory(destBBox)
        mathLib.warpImage(destImage, destWcs, srcImage,
                          srcWcs, self._warpingControl)
        return destImage

    def _computeDestBBox(self, destWcs, srcImage, srcWcs, border, maxBBox, destBBox):
        """Process destBBox argument for warpImage and warpExposure.

        Parameters
        ----------
        destWcs : `lsst.afw.geom.SkyWcs`
            WCS of warped image
        srcImage
            image or masked image to warp
        srcWcs : `lsst.afw.geom.SkyWcs`
            WCS of image
        border : `int`, optional
            grow bbox of warped image by this amount in all directions
            (in pixels); if negative then the bbox is shrunk; border is applied
            before ``maxBBox``; ignored if ``destBBox`` is not `None`
        maxBBox : `lsst.geom.Box2I`, optional
            maximum allowed parent bbox of warped image; if `None` then the
            warped image will be just big enough to contain all warped pixels;
            if provided then the warped image may be smaller, and so
            missing some warped pixels; ignored if ``destBBox`` is not `None`
        destBBox : `lsst.geom.Box2I`, optional
            exact parent bbox of warped image; if `None` then ``border`` and
            ``maxBBox`` are used to determine the bbox, otherwise ``border``
            and ``maxBBox`` are ignored
        """
        if destBBox is None:  # warning: == None fails due to Box2I.__eq__
            destBBox = computeWarpedBBox(
                destWcs, srcImage.getBBox(afwImage.PARENT), srcWcs)
            if border:
                destBBox.grow(border)
            if maxBBox is not None:
                destBBox.clip(maxBBox)
        return destBBox
