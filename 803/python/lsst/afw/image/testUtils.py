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
    Return None if mask is None.
    The data is presently copied but do not rely on that.
    """
    if im == None:
        return None
    arr = numpy.zeros([im.getWidth(), im.getHeight()], dtype=dtype)
    for row in range(im.getHeight()):
        for col in range(im.getWidth()):
            arr[col, row] = im.get(col, row)
    return arr

def arrayFromMask(mask, dtype=int):
    """Return a numpy array representation of a mask.
    Return None if mask is None.
    The data is presently copied but do not rely on that.
    """
    if mask == None:
        return None
    arr = numpy.zeros([mask.getWidth(), mask.getHeight()], dtype=dtype)
    for row in range(mask.getHeight()):
        for col in range(mask.getWidth()):
            arr[col, row] = mask.get(col, row)
    return arr

def arraysFromMaskedImage(maskedImage):
    """Return a tuple of (image, mask, variance) arrays from a MaskedImage.
    Return None for any missing component.
    The data is presently copied but do not rely on that.
    """
    return (
            arrayFromImage(maskedImage.getImage(True)),
            arrayFromMask(maskedImage.getMask(True)),
            arrayFromImage(maskedImage.getVariance(True)),
        )

def getImageMaskVarianceFromMaskedImage(maskedImage):
    """Return the image, mask and variance from a MaskedImage.
    The data is NOT copied.
    """
    return (maskedImage.getImage(True), maskedImage.getMask(True), maskedImage.getVariance(True))

def imageFromArray(arr, retType=afwImage.ImageF):
    """Return an Image representation of a numpy array.
    The data is presently copied but do not rely on that.
    """
    im = retType(arr.shape[0], arr.shape[1])
    setImageFromArray(im, arr)
    return im

def maskFromArray(arr):
    """Return a mask representation of a numpy array
    The data is presently copied but do not rely on that.
    """
    mask = afwImage.MaskU(arr.shape[0], arr.shape[1])
    setMaskFromArray(mask, arr)
    return mask

def maskedImageFromArrays(imMaskVarArrays, retType=afwImage.MaskedImageF):
    """Return a MaskedImage representation of a tuple of (image, mask, variance) numpy arrays.
    The data is presently copied but do not rely on that.
    """
    imArr = imMaskVarArrays[0]
    maskedImage = retType(imArr.shape[0], imArr.shape[1])
    setMaskedImageFromArrays(maskedImage, imMaskVarArrays)
    return maskedImage

def setImageFromArray(im, arr):
    """Set an existing lsst.afwImage.Image (of any type) from a numpy array.
    The data is presently copied but do not rely on that.
    """
    assert im.getWidth() == arr.shape[0] and im.getHeight() == arr.shape[1]
    
    for row in range(im.getHeight()):
        for col in range(im.getWidth()):
            im.set(col, row, arr[col, row])
    
def setMaskFromArray(mask, arr):
    """Set an existing lsst.afwImage.Mask from a numpy array
    The data is presently copied but do not rely on that.
    """
    assert mask.getWidth() == arr.shape[0] and mask.getHeight() == arr.shape[1]

    for row in range(mask.getHeight()):
        for col in range(mask.getWidth()):
            mask.set(col, row, int(arr[col, row]))

def setMaskedImageFromArrays(maskedImage, imMaskVarArrays):
    """Set an existing lsst.afwImage.MaskedImage (of any type) from a of a tuple of (image, mask, variance) numpy arrays.
    If image or variance arrays are None then that component is not set.
    The data is presently copied but do not rely on that.
    """
    imArr, maskArr, varArr = imMaskVarArrays
    if not (imArr.shape == varArr.shape == maskArr.shape):
        raise RuntimeError("The arrays must all be the same shape")
    if not (maskedImage.getWidth() == imArr.shape[0] and maskedImage.getHeight() == imArr.shape[1]):
        raise RuntimeError("The arrays must be the same shape as maskedImage")

    im, mask, var = getImageMaskVarianceFromMaskedImage(maskedImage)
    for row in range(maskedImage.getHeight()):
        for col in range(maskedImage.getWidth()):
            im.set(col, row, imArr[col, row])
            if mask:
                mask.set(col, row, maskArr[col, row])
            if var:
                var.set(col, row, varArr[col, row])

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

