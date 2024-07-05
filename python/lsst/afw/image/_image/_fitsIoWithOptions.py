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

__all__ = ["imageReadFitsWithOptions",
           "imageWriteFitsWithOptions", "exposureWriteFitsWithOptions"]

import logging
import lsst.geom
from lsst.afw.fits import ImageWriteOptions
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
    options : `lsst.daf.base.PropertySet`
        Write options. The item ``item`` is read (which defaults to "image").
        It must contain an `lsst.daf.base.PropertySet` with data for
        ``lsst.afw.fits.ImageWriteOptions``.
    item : `str`, optional
        Item to read from the ``options`` parameter.
    """
    if options is not None:
        try:
            writeOptions = ImageWriteOptions(options.getPropertySet(item))
        except Exception:
            _LOG.exception("Could not parse item %s from options; writing with defaults.", item)
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
        except Exception:
            _LOG.exception("Could not parse options; writing with defaults.")
        else:
            self.writeFits(dest, **writeOptionDict)
            return
    self.writeFits(dest)
