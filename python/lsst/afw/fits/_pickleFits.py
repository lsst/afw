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

from lsst.afw.fits import MemFileManager, ImageWriteOptions, ImageCompressionOptions

import lsst.afw.table
import lsst.afw.image


def reduceToFits(obj):
    """Pickle to FITS

    Intended to be used by the ``__reduce__`` method of a class.

    Parameters
    ----------
    obj
        any object with a ``writeFits`` method taking a
        `~lsst.afw.fits.MemFileManager` and possibly an
        `~lsst.afw.fits.ImageWriteOptions`.

    Returns
    -------
    reduced : `tuple` [callable, `tuple`]
        a tuple in the format returned by `~object.__reduce__`
    """
    manager = MemFileManager()
    options = ImageWriteOptions(ImageCompressionOptions(ImageCompressionOptions.NONE))
    table = getattr(obj, 'table', None)
    if isinstance(table, lsst.afw.table.BaseTable):
        # table objects don't take `options`
        obj.writeFits(manager)
    else:
        # MaskedImage and Exposure both require options for each plane (image, mask, variance)
        if isinstance(obj, (lsst.afw.image.MaskedImage, lsst.afw.image.Exposure)):
            obj.writeFits(manager, options, options, options)
        else:
            obj.writeFits(manager, options)
    size = manager.getLength()
    data = manager.getData()
    return (unreduceFromFits, (obj.__class__, data, size))


def unreduceFromFits(cls, data, size):
    """Unpickle from FITS

    Unpack data produced by `reduceToFits`. This method is used by the
    pickling framework and should not need to be called from user code.

    Parameters
    ----------
    cls : `type`
        the class of object to unpickle. Must have a class-level ``readFits``
        method taking a `~lsst.afw.fits.MemFileManager`.
    data : `bytes`
        an in-memory representation of the object, compatible with
        `~lsst.afw.fits.MemFileManager`
    size : `int`
        the length of `data`

    Returns
    -------
    unpickled : ``cls``
        the object represented by ``data``
    """
    manager = MemFileManager(size)
    manager.setData(data, size)
    return cls.readFits(manager)
