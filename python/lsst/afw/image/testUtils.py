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
    """Set an existing lsst.afwImage.MaskedImage (of any type) from a tuple
    of (image, mask, variance) numpy arrays.
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

def imagesDiffer(imageArr1, imageArr2, skipMaskArr=None, rtol=1.0e-05, atol=1e-08):
    """Compare the pixels of two image arrays; return True if close, False otherwise
    
    Inputs:
    - image1: first image to compare
    - image2: second image to compare
    - skipMaskArr: pixels to ignore; nonzero values are skipped
    - rtol: relative tolerance (see below)
    - atol: absolute tolerance (see below)
    
    rtol and atol are positive, typically very small numbers.
    The relative difference (rtol * abs(b)) and the absolute difference "atol" are added together
    to compare against the absolute difference between "a" and "b".
    
    Return a string describing the error if the images differ significantly, an empty string otherwise
    """
    retStrs = []
    if skipMaskArr != None:
        maskedArr1 = numpy.ma.array(imageArr1, copy=False, mask = skipMaskArr)
        maskedArr2 = numpy.ma.array(imageArr2, copy=False, mask = skipMaskArr)
        filledArr1 = maskedArr1.filled(0.0)
        filledArr2 = maskedArr2.filled(0.0)
    else:
        filledArr1 = imageArr1
        filledArr2 = imageArr2

    nan1 = numpy.isnan(filledArr1)
    nan2 = numpy.isnan(filledArr2)
    if numpy.any(nan1 != nan2):
        retStrs.append("NaNs differ")

    posinf1 = numpy.isposinf(filledArr1)
    posinf2 = numpy.isposinf(filledArr2)
    if numpy.any(posinf1 != posinf2):
        retStrs.append("+infs differ")

    neginf1 = numpy.isneginf(filledArr1)
    neginf2 = numpy.isneginf(filledArr2)
    if numpy.any(neginf1 != neginf2):
        retStrs.append("-infs differ")

    # compare values that should be comparable (are neither infinite, nan nor masked)
    valSkipMaskArr = nan1 | nan2 | posinf1 | posinf2 | neginf1 | neginf2
    if skipMaskArr != None:
        valSkipMaskArr |= skipMaskArr
    valMaskedArr1 = numpy.ma.array(imageArr1, copy=False, mask = valSkipMaskArr)
    valMaskedArr2 = numpy.ma.array(imageArr2, copy=False, mask = valSkipMaskArr)
    valFilledArr1 = valMaskedArr1.filled(0.0)
    valFilledArr2 = valMaskedArr2.filled(0.0)
    
    if not numpy.allclose(valFilledArr1, valFilledArr2, rtol=rtol, atol=atol):
        errArr = numpy.abs(valFilledArr1 - valFilledArr2)
        maxErr = errArr.max()
        maxPosInd = numpy.where(errArr==maxErr)
        maxPosTuple = (maxPosInd[0][0], maxPosInd[1][0])
        errStr = "maxDiff=%s at position %s; value=%s vs. %s" % \
            (maxErr,maxPosTuple, valFilledArr1[maxPosInd][0], valFilledArr2[maxPosInd][0])
        retStrs.insert(0, errStr)
    return "; ".join(retStrs)

def masksDiffer(maskArr1, maskArr2, skipMaskArr=None):
    """Compare the pixels of two mask arrays; return True if they match, False otherwise
    
    Inputs:
    - mask1: first image to compare
    - mask2: second image to compare
    - skipMaskArr: pixels to ignore; nonzero values are skipped
    
    Return a string describing the error if the images differ significantly, an empty string otherwise
    """
    retStr = ""
    if skipMaskArr != None:
        maskedArr1 = numpy.ma.array(maskArr1, copy=False, mask = skipMaskArr)
        maskedArr2 = numpy.ma.array(maskArr2, copy=False, mask = skipMaskArr)
        filledArr1 = maskedArr1.filled(0.0)
        filledArr2 = maskedArr2.filled(0.0)
    else:
        filledArr1 = maskArr1
        filledArr2 = maskArr2

    if numpy.any(filledArr1 != filledArr2):
        errArr = numpy.abs(filledArr1 - filledArr2)
        maxErr = errArr.max()
        maxPosInd = numpy.where(errArr==maxErr)
        maxPosTuple = (maxPosInd[0][0], maxPosInd[1][0])
        retStr = "maxDiff=%s at position %s; value=%s vs. %s" % \
            (maxErr,maxPosTuple, filledArr1[maxPosInd][0], filledArr2[maxPosInd][0])
        retStr = "masks differ"
    return retStr

def maskedImagesDiffer(maskedImageArrSet1, maskedImageArrSet2,
    doImage=True, doMask=True, doVariance=True, skipMaskArr=None, rtol=1.0e-05, atol=1e-08):
    """Compare pixels from two masked images
    
    Inputs:
    - maskedImageArrSet1: first masked image to compare as (image, mask, variance) arrays
    - maskedImageArrSet2: second masked image to compare as (image, mask, variance) arrays
    - doImage: compare image planes if True
    - doMask: compare mask planes if True
    - doVariance: compare variance planes if True
    - skipMaskArr: pixels to ingore on the image, mask and variance arrays; nonzero values are skipped
    - rtol: relative tolerance (see below)
    - atol: absolute tolerance (see below)
    
    rtol and atol are positive, typically very small numbers.
    The relative difference (rtol * abs(b)) and the absolute difference "atol" are added together
    to compare against the absolute difference between "a" and "b".
    
    Return a string describing the error if the images differ significantly, an empty string otherwise
    """
    retStrs = []
    for ind, (doPlane, planeName) in enumerate(((doImage, "image"),
                                                (doMask, "mask"),
                                                (doVariance, "variance"))):
        if not doPlane:
            continue

        if planeName == "mask":
            errStr = masksDiffer(maskedImageArrSet1[ind], maskedImageArrSet2[ind], skipMaskArr=skipMaskArr)
            if errStr:
                retStrs.append(errStr)
        else:
            errStr = imagesDiffer(maskedImageArrSet1[ind], maskedImageArrSet2[ind],
                skipMaskArr=skipMaskArr, rtol=rtol, atol=atol)
            if errStr:
                retStrs.append("%s planes differ: %s" % (planeName, errStr))
    return " | ".join(retStrs)


if __name__ == "__main__":
    maskedImage = afwImage.MaskedImageD("data/small")
    bb = afwImage.BBox(afwImage.PointI(200, 100), 50, 50)
    siPtr = maskedImage.Factory(maskedImage, bb)

    siArrays = arraysFromMaskedImage(siPtr)
    siCopy = maskedImage.Factory(maskedImage.getDimensions())
    siCopy = maskedImageFromArrays(siCopy, siArrays)

    maskedImage.writeFits("mi")
    siPtr.writeFits("si")
    siCopy.writeFits("siCopy")
    siCopy -= siPtr
    siCopy.writeFits("siNull")

