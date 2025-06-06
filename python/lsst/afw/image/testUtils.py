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

# the asserts are automatically imported so unit tests can find them without special imports;
# the other functions are hidden unless explicitly asked for
__all__ = ["assertImagesAlmostEqual", "assertImagesEqual", "assertMasksEqual",
           "assertMaskedImagesAlmostEqual", "assertMaskedImagesEqual"]

import numpy as np

import lsst.utils.tests
from ._image import ImageF
from ._basicUtils import makeMaskedImageFromArrays


def makeGaussianNoiseMaskedImage(dimensions, sigma, variance=1.0):
    """Make a gaussian noise MaskedImageF

    Inputs:
    - dimensions: dimensions of output array (cols, rows)
    - sigma; sigma of image plane's noise distribution
    - variance: constant value for variance plane
    """
    npSize = (dimensions[1], dimensions[0])
    image = np.random.normal(loc=0.0, scale=sigma,
                             size=npSize).astype(np.float32)
    mask = np.zeros(npSize, dtype=np.int32)
    variance = np.zeros(npSize, dtype=np.float32) + variance

    return makeMaskedImageFromArrays(image, mask, variance)


def makeRampImage(bbox, start=0, stop=None, imageClass=ImageF):
    """!Make an image whose values are a linear ramp

    @param[in] bbox  bounding box of image (an lsst.geom.Box2I)
    @param[in] start  starting ramp value, inclusive
    @param[in] stop  ending ramp value, inclusive; if None, increase by integer values
    @param[in] imageClass  type of image (e.g. lsst.afw.image.ImageF)
    """
    im = imageClass(bbox)
    imDim = im.getDimensions()
    numPix = imDim[0]*imDim[1]
    imArr = im.getArray()
    if stop is None:
        # increase by integer values
        stop = start + numPix - 1
    rampArr = np.linspace(start=start, stop=stop,
                          endpoint=True, num=numPix, dtype=imArr.dtype)
    # numpy arrays are transposed w.r.t. afwImage
    imArr[:] = np.reshape(rampArr, (imDim[1], imDim[0]))
    return im


@lsst.utils.tests.inTestCase
def assertImagesAlmostEqual(testCase, image0, image1, skipMask=None,
                            rtol=1.0e-05, atol=1e-08, msg="Images differ"):
    """!Assert that two images are almost equal, including non-finite values

    @param[in] testCase  unittest.TestCase instance the test is part of;
                        an object supporting one method: fail(self, msgStr)
    @param[in] image0  image 0, an lsst.afw.image.Image, lsst.afw.image.Mask,
        or transposed numpy array (see warning)
    @param[in] image1  image 1, an lsst.afw.image.Image, lsst.afw.image.Mask,
        or transposed numpy array (see warning)
    @param[in] skipMask  mask of pixels to skip, or None to compare all pixels;
        an lsst.afw.image.Mask, lsst.afw.image.Image, or transposed numpy array (see warning);
        all non-zero pixels are skipped
    @param[in] rtol  maximum allowed relative tolerance; more info below
    @param[in] atol  maximum allowed absolute tolerance; more info below
    @param[in] msg  exception message prefix; details of the error are appended after ": "

    The images are nearly equal if all pixels obey:
        |val1 - val0| <= rtol*|val1| + atol
    or, for float types, if nan/inf/-inf pixels match.

    @warning the comparison equation is not symmetric, so in rare cases the assertion
    may give different results depending on which image comes first.

    @warning the axes of numpy arrays are transposed with respect to Image and Mask data.
    Thus for example if image0 and image1 are both lsst.afw.image.ImageD with dimensions (2, 3)
    and skipMask is a numpy array, then skipMask must have shape (3, 2).

    @throw self.failureException (usually AssertionError) if any of the following are true
    for un-skipped pixels:
    - non-finite values differ in any way (e.g. one is "nan" and another is not)
    - finite values differ by too much, as defined by atol and rtol

    @throw TypeError if the dimensions of image0, image1 and skipMask do not match,
    or any are not of a numeric data type.
    """
    errStr = imagesDiffer(
        image0, image1, skipMask=skipMask, rtol=rtol, atol=atol)
    if errStr:
        testCase.fail(f"{msg}: {errStr}")


@lsst.utils.tests.inTestCase
def assertImagesEqual(*args, **kwds):
    """!Assert that two images are exactly equal, including non-finite values.

    All arguments are forwarded to assertAnglesAlmostEqual aside from atol and rtol,
    which are set to zero.
    """
    return assertImagesAlmostEqual(*args, atol=0, rtol=0, **kwds)


@lsst.utils.tests.inTestCase
def assertMasksEqual(testCase, mask0, mask1, skipMask=None, msg="Masks differ"):
    """!Assert that two masks are equal

    @param[in] testCase  unittest.TestCase instance the test is part of;
                        an object supporting one method: fail(self, msgStr)
    @param[in] mask0  mask 0, an lsst.afw.image.Mask, lsst.afw.image.Image,
        or transposed numpy array (see warning)
    @param[in] mask1  mask 1, an lsst.afw.image.Mask, lsst.afw.image.Image,
        or transposed numpy array (see warning)
    @param[in] skipMask  mask of pixels to skip, or None to compare all pixels;
        an lsst.afw.image.Mask, lsst.afw.image.Image, or transposed numpy array (see warning);
        all non-zero pixels are skipped
    @param[in] msg  exception message prefix; details of the error are appended after ": "

    @warning the axes of numpy arrays are transposed with respect to Mask and Image.
    Thus for example if mask0 and mask1 are both lsst.afw.image.Mask with dimensions (2, 3)
    and skipMask is a numpy array, then skipMask must have shape (3, 2).

    @throw self.failureException (usually AssertionError) if any any un-skipped pixels differ

    @throw TypeError if the dimensions of mask0, mask1 and skipMask do not match,
    or any are not of a numeric data type.
    """
    errStr = imagesDiffer(mask0, mask1, skipMask=skipMask, rtol=0, atol=0)
    if errStr:
        testCase.fail(f"{msg}: {errStr}")


@lsst.utils.tests.inTestCase
def assertMaskedImagesAlmostEqual(
    testCase, maskedImage0, maskedImage1,
    doImage=True, doMask=True, doVariance=True, skipMask=None,
    rtol=1.0e-05, atol=1e-08, msg="Masked images differ",
):
    """!Assert that two masked images are nearly equal, including non-finite values

    @param[in] testCase  unittest.TestCase instance the test is part of;
                        an object supporting one method: fail(self, msgStr)
    @param[in] maskedImage0  masked image 0 (an lsst.afw.image.MaskedImage or
        collection of three transposed numpy arrays: image, mask, variance)
    @param[in] maskedImage1  masked image 1 (an lsst.afw.image.MaskedImage or
        collection of three transposed numpy arrays: image, mask, variance)
    @param[in] doImage  compare image planes if True
    @param[in] doMask  compare mask planes if True
    @param[in] doVariance  compare variance planes if True
    @param[in] skipMask  mask of pixels to skip, or None to compare all pixels;
        an lsst.afw.image.Mask, lsst.afw.image.Image, or transposed numpy array;
        all non-zero pixels are skipped
    @param[in] rtol  maximum allowed relative tolerance; more info below
    @param[in] atol  maximum allowed absolute tolerance; more info below
    @param[in] msg  exception message prefix; details of the error are appended after ": "

    The mask planes must match exactly. The image and variance planes are nearly equal if all pixels obey:
        |val1 - val0| <= rtol*|val1| + atol
    or, for float types, if nan/inf/-inf pixels match.

    @warning the comparison equation is not symmetric, so in rare cases the assertion
    may give different results depending on which masked image comes first.

    @warning the axes of numpy arrays are transposed with respect to MaskedImage data.
    Thus for example if maskedImage0 and maskedImage1 are both lsst.afw.image.MaskedImageD
    with dimensions (2, 3) and skipMask is a numpy array, then skipMask must have shape (3, 2).

    @throw self.failureException (usually AssertionError) if any of the following are true
    for un-skipped pixels:
    - non-finite image or variance values differ in any way (e.g. one is "nan" and another is not)
    - finite values differ by too much, as defined by atol and rtol
    - mask pixels differ at all

    @throw TypeError if the dimensions of maskedImage0, maskedImage1 and skipMask do not match,
    either image or variance plane is not of a numeric data type,
    either mask plane is not of an integer type (unsigned or signed),
    or skipMask is not of a numeric data type.
    """
    if hasattr(maskedImage0, "image"):
        maskedImageArrList0 = (maskedImage0.image.array,
                               maskedImage0.mask.array,
                               maskedImage0.variance.array)
    else:
        maskedImageArrList0 = maskedImage0
    if hasattr(maskedImage1, "image"):
        maskedImageArrList1 = (maskedImage1.image.array,
                               maskedImage1.mask.array,
                               maskedImage1.variance.array)
    else:
        maskedImageArrList1 = maskedImage1

    for arrList, arg, name in (
        (maskedImageArrList0, maskedImage0, "maskedImage0"),
        (maskedImageArrList1, maskedImage1, "maskedImage1"),
    ):
        try:
            assert len(arrList) == 3
            # check that array shapes are all identical
            # check that image and variance are float or int of some kind
            # and mask is int of some kind
            for i in (0, 2):
                assert arrList[i].shape == arrList[1].shape
                assert arrList[i].dtype.kind in ("b", "i", "u", "f", "c")
            assert arrList[1].dtype.kind in ("b", "i", "u")
        except Exception:
            raise TypeError(f"{name}={arg!r} is not a supported type")

    errStrList = []
    for ind, (doPlane, planeName) in enumerate(((doImage, "image"),
                                                (doMask, "mask"),
                                                (doVariance, "variance"))):
        if not doPlane:
            continue

        if planeName == "mask":
            errStr = imagesDiffer(maskedImageArrList0[ind], maskedImageArrList1[ind], skipMask=skipMask,
                                  rtol=0, atol=0)
            if errStr:
                errStrList.append(errStr)
        else:
            errStr = imagesDiffer(maskedImageArrList0[ind], maskedImageArrList1[ind],
                                  skipMask=skipMask, rtol=rtol, atol=atol)
            if errStr:
                errStrList.append(f"{planeName} planes differ: {errStr}")

    if errStrList:
        errStr = "; ".join(errStrList)
        testCase.fail(f"{msg}: {errStr}")


@lsst.utils.tests.inTestCase
def assertMaskedImagesEqual(*args, **kwds):
    """!Assert that two masked images are exactly equal, including non-finite values.

    All arguments are forwarded to assertMaskedImagesAlmostEqual aside from atol and rtol,
    which are set to zero.
    """
    return assertMaskedImagesAlmostEqual(*args, atol=0, rtol=0, **kwds)


def imagesDiffer(image0, image1, skipMask=None, rtol=1.0e-05, atol=1e-08):
    """!Compare the pixels of two image or mask arrays; return True if close, False otherwise

    @param[in] image0  image 0, an lsst.afw.image.Image, lsst.afw.image.Mask,
        or transposed numpy array (see warning)
    @param[in] image1  image 1, an lsst.afw.image.Image, lsst.afw.image.Mask,
        or transposed numpy array (see warning)
    @param[in] skipMask  mask of pixels to skip, or None to compare all pixels;
        an lsst.afw.image.Mask, lsst.afw.image.Image, or transposed numpy array (see warning);
        all non-zero pixels are skipped
    @param[in] rtol  maximum allowed relative tolerance; more info below
    @param[in] atol  maximum allowed absolute tolerance; more info below

    The images are nearly equal if all pixels obey:
        |val1 - val0| <= rtol*|val1| + atol
    or, for float types, if nan/inf/-inf pixels match.

    @warning the comparison equation is not symmetric, so in rare cases the assertion
    may give different results depending on which image comes first.

    @warning the axes of numpy arrays are transposed with respect to Image and Mask data.
    Thus for example if image0 and image1 are both lsst.afw.image.ImageD with dimensions (2, 3)
    and skipMask is a numpy array, then skipMask must have shape (3, 2).

    @return a string which is non-empty if the images differ

    @throw TypeError if the dimensions of image0, image1 and skipMask do not match,
    or any are not of a numeric data type.
    """
    errStrList = []
    imageArr0 = image0.getArray() if hasattr(image0, "getArray") else image0
    imageArr1 = image1.getArray() if hasattr(image1, "getArray") else image1
    skipMaskArr = skipMask.getArray() if hasattr(skipMask, "getArray") else skipMask

    # check the inputs
    arrArgNameList = [
        (imageArr0, image0, "image0"),
        (imageArr1, image1, "image1"),
    ]
    if skipMask is not None:
        arrArgNameList.append((skipMaskArr, skipMask, "skipMask"))
    for i, (arr, arg, name) in enumerate(arrArgNameList):
        try:
            assert arr.dtype.kind in ("b", "i", "u", "f", "c")
        except Exception:
            raise TypeError(f"{name!r}={arg!r} is not a supported type")
        if i != 0:
            if arr.shape != imageArr0.shape:
                raise TypeError(f"{name} shape = {arr.shape} != {imageArr0.shape} = image0 shape")

    # np.allclose mis-handled unsigned ints in numpy 1.8
    # and subtraction doesn't give the desired answer in any case
    # so cast unsigned arrays into int64 (there may be a simple
    # way to safely use a smaller data type but I've not found it)
    if imageArr0.dtype.kind == "u":
        imageArr0 = imageArr0.astype(
            np.promote_types(imageArr0.dtype, np.int8))
    if imageArr1.dtype.kind == "u":
        imageArr1 = imageArr1.astype(
            np.promote_types(imageArr1.dtype, np.int8))

    if skipMaskArr is not None:
        skipMaskArr = np.array(skipMaskArr, dtype=bool)
        maskedArr0 = np.ma.array(imageArr0, copy=False, mask=skipMaskArr)
        maskedArr1 = np.ma.array(imageArr1, copy=False, mask=skipMaskArr)
        filledArr0 = maskedArr0.filled(0.0)
        filledArr1 = maskedArr1.filled(0.0)
    else:
        skipMaskArr = None
        filledArr0 = imageArr0
        filledArr1 = imageArr1

    try:
        np.array([np.nan], dtype=imageArr0.dtype)
        np.array([np.nan], dtype=imageArr1.dtype)
    except Exception:
        # one or both images does not support non-finite values (nan, etc.)
        # so just use value comparison
        valSkipMaskArr = skipMaskArr
    else:
        # both images support non-finite values, of which numpy has exactly three: nan, +inf and -inf;
        # compare those individually in order to give useful diagnostic output
        nan0 = np.isnan(filledArr0)
        nan1 = np.isnan(filledArr1)
        if np.any(nan0 != nan1):
            errStrList.append("NaNs differ")

        posinf0 = np.isposinf(filledArr0)
        posinf1 = np.isposinf(filledArr1)
        if np.any(posinf0 != posinf1):
            errStrList.append("+infs differ")

        neginf0 = np.isneginf(filledArr0)
        neginf1 = np.isneginf(filledArr1)
        if np.any(neginf0 != neginf1):
            errStrList.append("-infs differ")

        valSkipMaskArr = nan0 | nan1 | posinf0 | posinf1 | neginf0 | neginf1
        if skipMaskArr is not None:
            valSkipMaskArr |= skipMaskArr

    # compare values that should be comparable (are finite and not masked)
    valMaskedArr1 = np.ma.array(imageArr0, copy=False, mask=valSkipMaskArr)
    valMaskedArr2 = np.ma.array(imageArr1, copy=False, mask=valSkipMaskArr)
    valFilledArr1 = valMaskedArr1.filled(0.0)
    valFilledArr2 = valMaskedArr2.filled(0.0)

    if not np.allclose(valFilledArr1, valFilledArr2, rtol=rtol, atol=atol):
        errArr = np.abs(valFilledArr1 - valFilledArr2)
        maxErr = errArr.max()
        maxAbsInd = np.where(errArr == maxErr)
        maxAbsTuple = (int(maxAbsInd[0][0]), int(maxAbsInd[1][0]))
        # NOTE: use the second image, because the numpy test is:
        # |a - b| <= (atol + rtol * |b|)
        allcloseLimit = rtol*np.abs(valFilledArr2) + atol
        failing = np.where(errArr >= allcloseLimit)
        # We want value of the largest absolute error.
        maxFailing = errArr[failing].max()
        maxFailingInd = np.where(errArr == maxFailing)
        maxFailingTuple = (maxFailingInd[0][0], maxFailingInd[1][0])
        errStr = (f"{len(failing[0])} pixels failing np.allclose(), worst is: "
                  f"|{valFilledArr1[maxFailingTuple]} - {valFilledArr2[maxFailingTuple]}| = "
                  f"{maxFailing} > {allcloseLimit[maxFailingTuple]} "
                  f"(rtol*abs(image2)+atol with rtol={rtol}, atol={atol}) "
                  f"at position {maxFailingTuple}, and maximum absolute error: "
                  f"|{valFilledArr1[maxAbsInd][0]} - {valFilledArr2[maxAbsInd][0]}| = {maxErr} "
                  f"at position {maxAbsTuple}.")
        errStrList.insert(0, errStr)

    return "; ".join(errStrList)
