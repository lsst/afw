#
# LSST Data Management System
# Copyright 2015-2016 LSST/AURA
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
# see <https://www.lsstcorp.org/LegalNotices/>.
#

import numpy as np

import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
from .rgb import replaceSaturatedPixels, getZScale


def computeIntensity(imageR, imageG=None, imageB=None):
    """Return a naive total intensity from the red, blue, and green intensities

    Parameters
    ----------
    imageR : `lsst.afw.image.MaskedImage`, `lsst.afw.image.Image`, or `numpy.ndarray`, (Nx, Ny)
        intensity of image that'll be mapped to red; or intensity if imageG and imageB are None
    imageG : `lsst.afw.image.MaskedImage`, `lsst.afw.image.Image`, or `numpy.ndarray`, (Nx, Ny)
        intensity of image that'll be mapped to green; or None
    imageB : `lsst.afw.image.MaskedImage`, `lsst.afw.image.Image`, or `numpy.ndarray`, (Nx, Ny)
        intensity of image that'll be mapped to blue; or None

    Returns
    -------
    image : type of ``imageR``, ``imageG``, and `imageB``
    """
    if imageG is None or imageB is None:
        assert imageG is None and imageB is None, \
            "Please specify either a single image or red, green, and blue images"
        return imageR

    imageRGB = [imageR, imageG, imageB]

    for i, c in enumerate(imageRGB):
        if hasattr(c, "getImage"):
            c = imageRGB[i] = c.getImage()
        if hasattr(c, "getArray"):
            imageRGB[i] = c.getArray()

    intensity = (imageRGB[0] + imageRGB[1] + imageRGB[2])/float(3)
    #
    # Repack into whatever type was passed to us
    #
    Image = afwImage.ImageU if intensity.dtype == 'uint16' else afwImage.ImageF

    if hasattr(imageR, "getImage"):  # a maskedImage
        intensity = afwImage.makeMaskedImage(Image(intensity))
    elif hasattr(imageR, "getArray"):
        intensity = Image(intensity)

    return intensity


class Mapping:
    """Base class to map red, blue, green intensities into uint8 values

    Parameters
    ----------
    minimum : `float` or sequence of `float`
        Intensity that should be mapped to black. If an array, has three
        elements for R, G, B.
    image
        The image to be used to calculate the mapping.
        If provided, also the default for makeRgbImage()
    """

    def __init__(self, minimum=None, image=None):
        self._uint8Max = float(np.iinfo(np.uint8).max)

        try:
            len(minimum)
        except TypeError:
            minimum = 3*[minimum]
        assert len(minimum) == 3, "Please provide 1 or 3 values for minimum"

        self.minimum = minimum
        self._image = image

    def makeRgbImage(self, imageR=None, imageG=None, imageB=None,
                     xSize=None, ySize=None, rescaleFactor=None):
        """Convert 3 arrays, imageR, imageG, and imageB into a numpy RGB image

        imageR : `lsst.afw.image.Image` or `numpy.ndarray`, (Nx, Ny)
            Image to map to red (if `None`, use the image passed to the ctor)
        imageG : `lsst.afw.image.Image` or `numpy.ndarray`, (Nx, Ny), optional
            Image to map to green (if `None`, use imageR)
        imageB : `lsst.afw.image.Image` or `numpy.ndarray`, (Nx, Ny), optional
            Image to map to blue (if `None`, use imageR)
        xSize : `int`, optional
            Desired width of RGB image. If ``ySize`` is `None`, preserve aspect ratio
        ySize : `int`, optional
            Desired height of RGB image
        rescaleFactor : `float`, optional
            Make size of output image ``rescaleFactor*size`` of the input image
        """
        if imageR is None:
            if self._image is None:
                raise RuntimeError(
                    "You must provide an image (or pass one to the constructor)")
            imageR = self._image

        if imageG is None:
            imageG = imageR
        if imageB is None:
            imageB = imageR

        imageRGB = [imageR, imageG, imageB]
        for i, c in enumerate(imageRGB):
            if hasattr(c, "getImage"):
                c = imageRGB[i] = c.getImage()
            if hasattr(c, "getArray"):
                imageRGB[i] = c.getArray()

        if xSize is not None or ySize is not None:
            assert rescaleFactor is None, "You may not specify a size and rescaleFactor"
            h, w = imageRGB[0].shape
            if ySize is None:
                ySize = int(xSize*h/float(w) + 0.5)
            elif xSize is None:
                xSize = int(ySize*w/float(h) + 0.5)

            size = (ySize, xSize)  # n.b. y, x order for scipy
        elif rescaleFactor is not None:
            size = float(rescaleFactor)  # an int is intepreted as a percentage
        else:
            size = None

        if size is not None:
            try:
                import scipy.misc
            except ImportError as e:
                raise RuntimeError(
                    "Unable to rescale as scipy.misc is unavailable: %s" % e)

            for i, im in enumerate(imageRGB):
                imageRGB[i] = scipy.misc.imresize(
                    im, size, interp='bilinear', mode='F')

        return np.dstack(self._convertImagesToUint8(*imageRGB)).astype(np.uint8)

    def intensity(self, imageR, imageG, imageB):
        """Return the total intensity from the red, blue, and green intensities

        Notes
        -----
        This is a naive computation, and may be overridden by subclasses
        """
        return computeIntensity(imageR, imageG, imageB)

    def mapIntensityToUint8(self, intensity):
        """Map an intensity into the range of a uint8, [0, 255] (but not converted to uint8)
        """
        with np.errstate(invalid='ignore', divide='ignore'):  # n.b. np.where can't and doesn't short-circuit
            return np.where(intensity <= 0, 0,
                            np.where(intensity < self._uint8Max, intensity, self._uint8Max))

    def _convertImagesToUint8(self, imageR, imageG, imageB):
        """Use the mapping to convert images imageR, imageG, and imageB to a triplet of uint8 images
        """
        imageR = imageR - self.minimum[0]  # n.b. makes copy
        imageG = imageG - self.minimum[1]
        imageB = imageB - self.minimum[2]

        fac = self.mapIntensityToUint8(self.intensity(imageR, imageG, imageB))

        imageRGB = [imageR, imageG, imageB]
        with np.errstate(invalid="ignore"):  # suppress NAN warnings
            for c in imageRGB:
                c *= fac
                # individual bands can still be < 0, even if fac isn't
                c[c < 0] = 0

        pixmax = self._uint8Max
        # copies -- could work row by row to minimise memory usage
        r0, g0, b0 = imageRGB

        # n.b. np.where can't and doesn't short-circuit
        with np.errstate(invalid='ignore', divide='ignore'):
            for i, c in enumerate(imageRGB):
                c = np.where(r0 > g0,
                             np.where(r0 > b0,
                                      np.where(r0 >= pixmax, c*pixmax/r0, c),
                                      np.where(b0 >= pixmax, c*pixmax/b0, c)),
                             np.where(g0 > b0,
                                      np.where(g0 >= pixmax, c*pixmax/g0, c),
                                      np.where(b0 >= pixmax, c*pixmax/b0, c))).astype(np.uint8)
                c[c > pixmax] = pixmax

                imageRGB[i] = c

        return imageRGB


class LinearMapping(Mapping):
    """A linear map of red, blue, green intensities into uint8 values

    Parameters
    ----------
    minimum : `float` or sequence of `float`
        Intensity that should be mapped to black. If an array, has three
        elements for R, G, B.
    maximum : `float`
        Intensity that should be mapped to white
    image
        Image to estimate minimum/maximum if not explicitly set
    """

    def __init__(self, minimum=None, maximum=None, image=None):
        if minimum is None or maximum is None:
            assert image is not None, "You must provide an image if you don't set both minimum and maximum"

            stats = afwMath.makeStatistics(image, afwMath.MIN | afwMath.MAX)
            if minimum is None:
                minimum = stats.getValue(afwMath.MIN)
            if maximum is None:
                maximum = stats.getValue(afwMath.MAX)

        Mapping.__init__(self, minimum, image)
        self.maximum = maximum

        if maximum is None:
            self._range = None
        else:
            assert maximum - minimum != 0, "minimum and maximum values must not be equal"
            self._range = float(maximum - minimum)

    def mapIntensityToUint8(self, intensity):
        """Return an array which, when multiplied by an image, returns that
        image mapped to the range of a uint8, [0, 255] (but not converted to uint8)

        The intensity is assumed to have had ``minimum`` subtracted (as that
        can be done per-band)
        """
        with np.errstate(invalid='ignore', divide='ignore'):  # n.b. np.where can't and doesn't short-circuit
            return np.where(intensity <= 0, 0,
                            np.where(intensity >= self._range,
                                     self._uint8Max/intensity, self._uint8Max/self._range))


class ZScaleMapping(LinearMapping):
    """A mapping for a linear stretch chosen by the zscale algorithm
    (preserving colours independent of brightness)

    x = (I - minimum)/range

    Parameters
    ----------
    image
        Image whose parameters are desired
    nSamples : `int`
        The number of samples to use to estimate the zscale parameters
    contrast : `float`
    """

    def __init__(self, image, nSamples=1000, contrast=0.25):
        if not hasattr(image, "getArray"):
            image = afwImage.ImageF(image)
        z1, z2 = getZScale(image, nSamples, contrast)

        LinearMapping.__init__(self, z1, z2, image)


class AsinhMapping(Mapping):
    """A mapping for an asinh stretch (preserving colours independent of brightness)

    x = asinh(Q (I - minimum)/range)/Q

    Notes
    -----
    This reduces to a linear stretch if Q == 0

    See http://adsabs.harvard.edu/abs/2004PASP..116..133L
    """

    def __init__(self, minimum, dataRange, Q=8):
        Mapping.__init__(self, minimum)

        # 32bit floating point machine epsilon; sys.float_info.epsilon is 64bit
        epsilon = 1.0/2**23
        if abs(Q) < epsilon:
            Q = 0.1
        else:
            Qmax = 1e10
            if Q > Qmax:
                Q = Qmax

        if False:
            self._slope = self._uint8Max/Q  # gradient at origin is self._slope
        else:
            frac = 0.1                  # gradient estimated using frac*range is _slope
            self._slope = frac*self._uint8Max/np.arcsinh(frac*Q)

        self._soften = Q/float(dataRange)

    def mapIntensityToUint8(self, intensity):
        """Return an array which, when multiplied by an image, returns that image mapped to the range of a
        uint8, [0, 255] (but not converted to uint8)

        The intensity is assumed to have had minimum subtracted (as that can be done per-band)
        """
        with np.errstate(invalid='ignore', divide='ignore'):  # n.b. np.where can't and doesn't short-circuit
            return np.where(intensity <= 0, 0, np.arcsinh(intensity*self._soften)*self._slope/intensity)


class AsinhZScaleMapping(AsinhMapping):
    """A mapping for an asinh stretch, estimating the linear stretch by zscale

    x = asinh(Q (I - z1)/(z2 - z1))/Q

    Parameters
    ----------
    image
        The image to analyse, or a list of 3 images to be converted to an intensity image
    Q : `int`
        The asinh softening parameter
    pedestal : `float` or sequence of `float`, optional
        The value, or array of 3 values, to subtract from the images

        N.b. pedestal, if not None, is removed from the images when calculating the zscale
        stretch, and added back into Mapping.minimum[]

    See also
    --------
    AsinhMapping
    """

    def __init__(self, image, Q=8, pedestal=None):
        try:
            assert len(image) in (1, 3,), "Please provide 1 or 3 images"
        except TypeError:
            image = [image]

        if pedestal is not None:
            try:
                assert len(pedestal) in (
                    1, 3,), "Please provide 1 or 3 pedestals"
            except TypeError:
                pedestal = 3*[pedestal]

            image = list(image)        # needs to be mutable
            for i, im in enumerate(image):
                if pedestal[i] != 0.0:
                    if hasattr(im, "getImage"):
                        im = im.getImage()
                    if hasattr(im, "getArray"):
                        im = im.getArray()

                    image[i] = im - pedestal[i]  # n.b. a copy
        else:
            pedestal = len(image)*[0.0]

        image = computeIntensity(*image)

        zscale = ZScaleMapping(image)
        # zscale.minimum is always a triple
        dataRange = zscale.maximum - zscale.minimum[0]
        minimum = zscale.minimum

        for i, level in enumerate(pedestal):
            minimum[i] += level

        AsinhMapping.__init__(self, minimum, dataRange, Q)
        self._image = image             # support self.makeRgbImage()


def makeRGB(imageR, imageG=None, imageB=None, minimum=0, dataRange=5, Q=8, fileName=None,
            saturatedBorderWidth=0, saturatedPixelValue=None,
            xSize=None, ySize=None, rescaleFactor=None):
    """Make a set of three images into an RGB image using an asinh stretch and
    optionally write it to disk

    Parameters
    ----------
    imageR
    imageG
    imageB
    minimum : `float` or sequence of `float`
    dataRange
    Q : `int`
    fileName : `str`
        The output file. The suffix defines the format, and must be supported by matplotlib
    saturatedBorderWidth
        If saturatedBorderWidth is non-zero, replace saturated pixels with
        ``saturatedPixelValue``. Note that replacing saturated pixels requires
        that the input images be `lsst.afw.image.MaskedImage`.
    saturatedPixelValue
    xSize
    ySize
    rescaleFactor
    """
    if imageG is None:
        imageG = imageR
    if imageB is None:
        imageB = imageR

    if saturatedBorderWidth:
        if saturatedPixelValue is None:
            raise ValueError(
                "saturatedPixelValue must be set if saturatedBorderWidth is set")
        replaceSaturatedPixels(imageR, imageG, imageB,
                               saturatedBorderWidth, saturatedPixelValue)

    asinhMap = AsinhMapping(minimum, dataRange, Q)
    rgb = asinhMap.makeRgbImage(imageR, imageG, imageB,
                                xSize=xSize, ySize=ySize, rescaleFactor=rescaleFactor)

    if fileName:
        writeRGB(fileName, rgb)

    return rgb


def displayRGB(rgb, show=True):
    """Display an rgb image using matplotlib

    Parameters
    ----------
    rgb
        The RGB image in question
    show : `bool`
        If `True`, call `matplotlib.pyplot.show()`
    """
    import matplotlib.pyplot as plt
    plt.imshow(rgb, interpolation='nearest', origin="lower")
    if show:
        plt.show()
    return plt


def writeRGB(fileName, rgbImage):
    """Write an RGB image to disk

    Parameters
    ----------
    fileName : `str`
        The output file. The suffix defines the format, and must be supported by matplotlib

        Most versions of matplotlib support png and pdf (although the eps/pdf/svg writers may be buggy,
        possibly due an interaction with useTeX=True in the matplotlib settings).

        If your matplotlib bundles pil/pillow you should also be able to write jpeg and tiff files.
    rgbImage
        The image, as made by e.g. makeRGB
    """
    import matplotlib.image
    matplotlib.image.imsave(fileName, rgbImage)

#
# Support the legacy API
#


class asinhMappingF:  # noqa N801
    """Deprecated object used to support legacy API
    """

    def __init__(self, minimum, dataRange, Q):
        self.minimum = minimum
        self.dataRange = dataRange
        self.Q = Q


class _RgbImageF:
    """Deprecated object used to support legacy API
    """

    def __init__(self, imageR, imageG, imageB, mapping):
        asinh = AsinhMapping(mapping.minimum, mapping.dataRange, mapping.Q)
        self.rgb = asinh.makeRgbImage(imageR, imageG, imageB)

    def write(self, fileName):
        writeRGB(fileName, self.rgb)


def RgbImageF(imageR, imageG, imageB, mapping):
    """Deprecated legacy API
    """
    return _RgbImageF(imageR, imageG, imageB, mapping)
