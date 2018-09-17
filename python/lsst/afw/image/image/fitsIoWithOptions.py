#
# LSST Data Management System
# Copyright 2018 LSST/AURA.
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
__all__ = ["imageReadFitsWithOptions",
           "imageWriteFitsWithOptions", "exposureWriteFitsWithOptions"]

import lsst.geom
from lsst.log import Log
from lsst.afw.fits import ImageWriteOptions
from . import image


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
    options : `lsst.daf.base.PropertyList`
        Read options:

        - llcX: bbox minimum x (int)
        - llcY: bbox minimum y (int, must be present if llcX is present)
        - width: bbox width (int, must be present if llcX is present)
        - height: bbox height (int, must be present if llcX is present)
        - imageOrigin: one of "LOCAL" or "PARENT" (has no effect unless
            a bbox is specified by llcX, etc.)

    Raises
    ------
    RuntimeError
        If options contains an unknown value for "imageOrigin"
    lsst.pex.exceptions.NotFoundError
        If options contains "llcX" and is missing any of
        "llcY", "width", or "height".
    """
    bbox = lsst.geom.Box2I()
    if options.exists("llcX"):
        llcX = options.getInt("llcX")
        llcY = options.getInt("llcY")
        width = options.getInt("width")
        height = options.getInt("height")
        bbox = lsst.geom.Box2I(lsst.geom.Point2I(llcX, llcY), lsst.geom.Extent2I(width, height))
    origin = image.PARENT
    if options.exists("imageOrigin"):
        originStr = options.getString("imageOrigin")
        if (originStr == "LOCAL"):
            origin = image.LOCAL
        elif (originStr == "PARENT"):
            origin = image.PARENT
        else:
            raise RuntimeError("Unknown ImageOrigin type {}".format(originStr))

    return cls(source, bbox=bbox, origin=origin)


def imageWriteFitsWithOptions(self, dest, options):
    """Write an Image or Mask to FITS, with options

    Parameters
    ----------
    dest : `str`
        Fits file path to which to write the image or mask.
    options : `lsst.daf.base.PropertySet
        Write options. The item "image" is read. It must contain an
        `lsst.daf.base.PropertySet` with data for
        ``lsst.afw.fits.ImageWriteOptions``.
    """
    if options is not None:
        try:
            writeOptions = ImageWriteOptions(options.getPropertySet("image"))
        except Exception as e:
            log = Log.getLogger("lsst.afw.image")
            log.warn("Could not parse options; writing with defaults: {}".format(e))
        else:
            self.writeFits(dest, writeOptions)
            return
    self.writeFits(dest)


def exposureWriteFitsWithOptions(self, dest, options):
    """Write an Exposure or MaskedImage to FITS, with options

    Parameters
    ----------
    dest : `str`
        Fits file path to which to write the exposure or masked image.
    options : `lsst.daf.base.PropertySet`
        Write options. The items "image", "mask" and "variance" are read.
        Each must be an `lsst.daf.base.PropertySet` with data for
        ``lsst.afw.fits.ImageWriteOptions``.
    """
    if options is not None:
        try:
            writeOptionDict = {name + "Options": ImageWriteOptions(options.getPropertySet(name))
                               for name in ("image", "mask", "variance")}
        except Exception as e:
            log = Log.getLogger("lsst.afw.image")
            log.warn("Could not parse options; writing with defaults: {}".format(e))
        else:
            self.writeFits(dest, **writeOptionDict)
            return
    self.writeFits(dest)
