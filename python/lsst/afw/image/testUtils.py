#!/usr/bin/env python
##\file
## \brief Utilities to help write tests, mostly using numpy 
##
## Subroutines to move data between numpy arrays and lsst::afw::image classes
## Mask, Image and MaskedImage.
## 
## Please only use these for testing; they are too slow for production work!
## Eventually Image, Mask and MaskedImage will offer much better ways to do this.

import numpy
import lsst.afw.image as afwImage

def arrayFromImage(im, dtype=float):
    """Return a numpy array representation of an image.
    The data is presently copied but do not rely on that.
    """
    arr = numpy.zeros([im.getWidth(), im.getHeight()], dtype=dtype)
    for row in range(im.getHeight()):
        for col in range(im.getWidth()):
            arr[col, row] = im.get(col, row)
    return arr

def arrayFromMask(im, dtype=int):
    """Return a numpy array representation of a mask.
    The data is presently copied but do not rely on that.
    
    Warning: will fail for uint8 masks because lssgImageTypes.i maps uint8 to char;
    use ord(im.getPtr(col, row)) to get each pixel value
    """
    arr = numpy.zeros([im.getWidth(), im.getHeight()], dtype=dtype)
    for row in range(im.getHeight()):
        for col in range(im.getWidth()):
            arr[col, row] = im.get(col, row)
    return arr

def arraysFromMaskedImage(maskedImage):
    """Return a tuple of (image, variance, mask) arrays from a MaskedImage.
    The data is presently copied but do not rely on that.
    """
    return (
            arrayFromImage(maskedImage.getImage()),
            arrayFromImage(maskedImage.getVariance()),
            arrayFromMask(maskedImage.getMask()),
        )

def getImageVarianceMaskFromMaskedImage(maskedImage):
    """Return the image, variance and mask from a MaskedImage.
    Image and variance are of type lsst.afwImage.ImageD
    and mask is of type lsst.afwImage.MaskD.
    The data is NOT copied.
    """
    return (maskedImage.getImage(), maskedImage.getVariance(), maskedImage.getMask())

def imageFromArray(im, arr):
    """Create an lsst.afwImage.ImageD from a numpy array.
    The data is presently copied but do not rely on that.
    """

    assert im.getWidth() == arr.shape[0] and im.getHeight() == arr.shape[1]
    
    for row in range(im.getHeight()):
        for col in range(im.getWidth()):
            im.set(col, row, arr[col, row])

    return im

def maskFromArray(mask, arr):
    """Create an lsst.afwImage.MaskD from a numpy array
    The data is presently copied but do not rely on that.
    """

    assert mask.getWidth() == arr.shape[0] and mask.getHeight() == arr.shape[1]
    
    mask = afwImage.MaskD(arr.shape[0], arr.shape[1])
    for row in range(mask.getHeight()):
        for col in range(mask.getWidth()):
            mask.set(col, row, int(arr[col, row]))

    return mask

def maskedImageFromArrays(maskedImage, imVarMaskArrays):
    """Create a MaskedImage from a tuple of (image, variance, mask) arrays
    The data is presently copied but do not rely on that.
    """
    imArr, varArr, maskArr = imVarMaskArrays
    if not (imArr.shape == varArr.shape == maskArr.shape):
        raise RuntimeError("The arrays must all be the same shape")

    assert maskedImage.getWidth() == imArr.shape[0] and maskedImage.getHeight() == imArr.shape[1]

    im, var, mask = getImageVarianceMaskFromMaskedImage(maskedImage)
    for row in range(maskedImage.getHeight()):
        for col in range(maskedImage.getWidth()):
            im.set(col, row, imArr[col, row])
            var.set(col, row, varArr[col, row])
            mask.set(col, row, maskArr[col, row])

    return maskedImage

if __name__ == "__main__":
    maskedImage = afwImage.MaskedImageD("data/small")
    bb = afwImage.BBox(afwImage.PointI(200, 100), 50, 50)
    siPtr = maskedImage.Factory(maskedImage, bb)

    siArrays = arraysFromMaskedImage(si)
    siCopy = maskedImage.Factory(maskedImage.getDimensions())
    siCopy = maskedImageFromArrays(siCopy, siArrays)

    maskedImage.writeFits("mi")
    si.writeFits("si")
    siCopy.writeFits("siCopy")
    siCopy -= si
    siCopy.writeFits("siNull")

