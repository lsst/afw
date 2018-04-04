from __future__ import absolute_import, division, print_function

from lsst.afw.fits.fitsLib import MemFileManager, ImageWriteOptions, ImageCompressionOptions


def reduceToFits(obj):
    """Pickle to FITS

    Intended to be used by the __reduce__ method of a class.

    Assumes the existence of a "writeFits" method on the object.
    """
    manager = MemFileManager()
    options = ImageWriteOptions(ImageCompressionOptions(ImageCompressionOptions.NONE))
    try:
        obj.writeFits(manager, options)
    except Exception:
        obj.writeFits(manager)
    size = manager.getLength()
    data = manager.getData()
    return (unreduceFromFits, (obj.__class__, data, size))


def unreduceFromFits(cls, data, size):
    """Unpickle from FITS

    Unpacks data produced by reduceToFits.

    Assumes the existence of a "readFits" method on the object.
    """
    manager = MemFileManager(size)
    manager.setData(data, size)
    return cls.readFits(manager)
