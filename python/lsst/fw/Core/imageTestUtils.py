#!/usr/bin/env python
"""Subroutines to move data between numpy arrays and lsst Images and MaskedImages.

Please only use these for testing; they are too slow for production work!
Eventually Image, Mask and MaskedImage will offer much better ways to do this.
"""
import numpy
import lsst.fw.Core.fwLib as fw

def arrayFromImage(im, dtype=float):
    """Return a numpy array representation of an image.
    The data is presently copied but do not rely on that.
    """
    arr = numpy.zeros([im.getCols(), im.getRows()], dtype=dtype)
    for row in range(im.getRows()):
        for col in range(im.getCols()):
            arr[col, row] = im.getPtr(col, row)
    return arr

def arrayFromMask(im, dtype=int):
    """Return a numpy array representation of a mask.
    The data is presently copied but do not rely on that.
    """
    arr = numpy.zeros([im.getCols(), im.getRows()], dtype=dtype)
    for row in range(im.getRows()):
        for col in range(im.getCols()):
            arr[col, row] = ord(im.getPtr(col, row))
    return arr

def arraysFromMaskedImage(maskedImage):
    """Return a tuple of (image, variance, mask) arrays from a MaskedImage.
    The data is presently copied but do not rely on that.
    """
    return (
            arrayFromImage(maskedImage.getImage().get()),
            arrayFromImage(maskedImage.getVariance().get()),
            arrayFromMask(maskedImage.getMask().get()),
        )

def getImageVarianceMaskFromMaskedImage(maskedImage):
    """Return the image, variance and mask from a MaskedImage.
    Image and variance are of type lsst.fw.Core.fwLib.ImageD
    and mask is of type lsst.fw.Core.fwLib.MaskD.
    The data is NOT copied.
    """
    imPtr = maskedImage.getImage()
    im = imPtr.get()
    im.this.disown()
    varPtr = maskedImage.getVariance()
    var = varPtr.get()
    var.this.disown()
    maskPtr = maskedImage.getMask()
    mask = maskPtr.get()
    mask.this.disown()
    return (im, var, mask)

def imageFromArray(arr):
    """Create an lsst.fw.Core.fwLib.ImageD from a numpy array.
    The data is presently copied but do not rely on that.
    """
    im = fw.ImageD(arr.shape[0], arr.shape[1])
    for row in range(im.getRows()):
        for col in range(im.getCols()):
            im.set(col, row, arr[col, row])
    return im

def maskFromArray(arr):
    """Create an lsst.fw.Core.fwLib.MaskD from a numpy array
    The data is presently copied but do not rely on that.
    """
    mask = fw.MaskD(arr.shape[0], arr.shape[1])
    for row in range(mask.getRows()):
        for col in range(mask.getCols()):
            mask.set(col, row, int(arr[col, row]))
    return mask

def maskedImageFromArrays(imVarMaskArrays):
    """Create a MaskedImage from a tuple of (image, variance, mask) arrays
    The data is presently copied but do not rely on that.
    """
    imArr, varArr, maskArr = imVarMaskArrays
    if not (imArr.shape == varArr.shape == maskArr.shape):
        raise RuntimeError("The arrays must all be the same shape")
    maskedImage = fw.MaskedImageD(imArr.shape[0], imArr.shape[1])
    im, var, mask = getImageVarianceMaskFromMaskedImage(maskedImage)
    for row in range(maskedImage.getRows()):
        for col in range(maskedImage.getCols()):
            im.set(col, row, imArr[col, row])
            var.set(col, row, varArr[col, row])
            mask.set(col, row, maskArr[col, row])
    return maskedImage

if __name__ == "__main__":
    maskedImage = fw.MaskedImageD()
    maskedImage.readFits("data/small")
    bb = fw.BBox2i(200, 100, 50, 50)
    siPtr = maskedImage.getSubImage(bb)
    si = siPtr.get()
    si.this.disown()
    siArrays = arraysFromMaskedImage(si)
    siCopy = maskedImageFromArrays(siArrays)
    maskedImage.writeFits("mi")
    si.writeFits("si")
    siCopy.writeFits("siCopy")
    siCopy -= si
    siCopy.writeFits("siNull")

