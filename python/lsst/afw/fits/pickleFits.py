from lsst.afw.fits.fitsLib import MemFileManager, ImageWriteOptions, ImageCompressionOptions

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
