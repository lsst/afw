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

__all__ = ["imageReadFitsWithOptions",]

import logging
import lsst.geom
from lsst.afw.fits import CompressionOptions
from ._imageLib import ImageOrigin

_LOG = logging.getLogger("lsst.afw.image")


# This must be added to a class as a *classmethod*, for example:
#
# @continueclass
# class MaskX:
#     readFitsWithOptions = classmethod(imageReadFitsWithOptions)
def imageReadFitsWithOptions(cls, source, options):
    """Read an Image, Mask, MaskedImage or Exposure from  a FITS file,
    with options.

    Parameters
    ----------
    source : `str`
        Fits file path from which to read image, mask, masked image
        or exposure.
    options : `lsst.daf.base.PropertySet` or `dict`
        Read options:

        - llcX: bbox minimum x (int)
        - llcY: bbox minimum y (int, must be present if llcX is present)
        - width: bbox width (int, must be present if llcX is present)
        - height: bbox height (int, must be present if llcX is present)
        - imageOrigin: one of "LOCAL" or "PARENT" (has no effect unless
            a bbox is specified by llcX, etc.)

        Alternatively, a bounding box () can be passed on with the
        ``"bbox"'' (`lsst.geom.Box2I`) key.

    Raises
    ------
    RuntimeError
        If options contains an unknown value for "imageOrigin"
    lsst.pex.exceptions.NotFoundError
        If options contains "llcX" and is missing any of
        "llcY", "width", or "height".
    """
    origin = ImageOrigin.PARENT
    bbox = lsst.geom.Box2I()
    if "bbox" in options:
        bbox = options["bbox"]
    elif "llcX" in options:
        llcX = int(options["llcX"])
        llcY = int(options["llcY"])
        width = int(options["width"])
        height = int(options["height"])
        bbox = lsst.geom.Box2I(lsst.geom.Point2I(llcX, llcY), lsst.geom.Extent2I(width, height))
        if "imageOrigin" in options:
            originStr = str(options["imageOrigin"])
            if (originStr == "LOCAL"):
                origin = ImageOrigin.LOCAL
            elif (originStr == "PARENT"):
                origin = ImageOrigin.PARENT
            else:
                raise RuntimeError("Unknown ImageOrigin type {}".format(originStr))

    return cls(source, bbox=bbox, origin=origin)


def imageWriteFitsWithOptions(self, dest, options, item="image"):
    """Write an Image or Mask to FITS, with options

    Parameters
    ----------
    dest : `str`
        Fits file path to which to write the image or mask.
    options : `~collections.abc.Mapping`
        Write options. The item ``item`` is accessed (which defaults to
        "image"). It must contain a mapping with data for
        `lsst.afw.fits.CompressionOptions.from_mapping`, or `None` for no
        compression.
    item : `str`, optional
        Item to read from the ``options`` parameter.
    """
    if options is not None:
        writeOptions = CompressionOptions.from_mapping(options[item])
        self.writeFits(dest, writeOptions)
    else:
        self.writeFits(dest)


def exposureWriteFitsWithOptions(self, dest, options):
    """Write an Exposure or MaskedImage to FITS, with options

    Parameters
    ----------
    dest : `str`
        Fits file path to which to write the exposure or masked image.
    options : `~collections.abc.Mapping`
        Write options. The items "image", "mask" and "variance" are read.
        Each must contain a mapping with data for
        `lsst.afw.fits.CompressionOptions.from_mapping`, or `None` for no
        compression.
    """
    if options is not None:
        writeOptionDict = {name + "Options": CompressionOptions.from_mapping(plane_options)
                           for name in ("image", "mask", "variance")
                           if (plane_options := options[name]) is not None}
        self.writeFits(dest, **writeOptionDict)
    else:
        self.writeFits(dest)
