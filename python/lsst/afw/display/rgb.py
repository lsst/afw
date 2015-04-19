import numpy as np

import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
from lsst.afw.display.displayLib import replaceSaturatedPixels, getZScale

def computeIntensity(imageR, imageG=None, imageB=None):
    """!Return a naive total intensity from the red, blue, and green intensities
    \param imageR intensity of image that'll be mapped to red; or intensity if imageG and imageB are None
    \param imageG intensity of image that'll be mapped to green; or None
    \param imageB intensity of image that'll be mapped to blue; or None

    Inputs may be MaskedImages, Images, or numpy arrays and the return is of the same type
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

    if hasattr(imageR, "getImage"): # a maskedImage
        intensity = afwImage.makeMaskedImage(Image(intensity))
    elif hasattr(imageR, "getArray"):
        intensity = Image(intensity)

    return intensity

class Mapping(object):
    """!Baseclass to map red, blue, green intensities into uint8 values"""
    
    def __init__(self, minimum=None, image=None):
        """!Create a mapping
        \param minimum  Intensity that should be mapped to black (a scalar or array for R, G, B)
        \param image The image to be used to calculate the mapping.  If provided, also the default for makeRgbImage()
        """
        self._uint8Max = float(np.iinfo(np.uint8).max)

        try:
            len(minimum)
        except:
            minimum = 3*[minimum]
        assert len(minimum) == 3, "Please provide 1 or 3 values for minimum"

        self.minimum = minimum
        self._image = image

    def makeRgbImage(self, imageR=None, imageG=None, imageB=None):
        """!Convert 3 arrays, imageR, imageG, and imageB into a numpy RGB image
        \param imageR Image to map to red (if None, use the image passed to the ctor)
        \param imageG Image to map to green (if None, use imageR)
        \param imageB Image to map to blue (if None, use imageR)

        N.b. images may be afwImage.Images or numpy arrays
        """
        if imageR is None:
            if self._image is None:
                raise RuntimeError("You must provide an image (or pass one to the constructor)")
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

        return np.dstack(self._convertImagesToUint8(*imageRGB)).astype(np.uint8)

    def intensity(self, imageR, imageG, imageB):
        """!Return the total intensity from the red, blue, and green intensities

        This is a naive computation, and may be overridden by subclasses
        """
        return computeIntensity(imageR, imageG, imageB)

    def mapIntensityToUint8(self, I):
        """Map an intensity into the range of a uint8, [0, 255] (but not converted to uint8)"""
        return np.where(I <= 0, 0, np.where(I < self._uint8Max, I, self._uint8Max))

    def _convertImagesToUint8(self, imageR, imageG, imageB):
        """Use the mapping to convert images imageR, imageG, and imageB to a triplet of uint8 images"""
        imageR = imageR - self.minimum[0]  # n.b. makes copy
        imageG = imageG - self.minimum[1]
        imageB = imageB - self.minimum[2]

        fac = self.mapIntensityToUint8(self.intensity(imageR, imageG, imageB))

        imageRGB = [imageR, imageG, imageB]
        for c in imageRGB:
            c *= fac
            c[c < 0] = 0                # individual bands can still be < 0, even if fac isn't

        pixmax = self._uint8Max
        r0, g0, b0 = imageRGB           # copies -- could work row by row to minimise memory usage

        with np.errstate(invalid='ignore', divide='ignore'): # n.b. np.where can't and doesn't short-circuit
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
    """!A linear map map of red, blue, green intensities into uint8 values"""

    def __init__(self, minimum=None, maximum=None, image=None):
        """!A linear stretch from [minimum, maximum]; if one or both are omitted use image minimum/maximum to set them

        \param minimum  Intensity that should be mapped to black (a scalar or array for R, G, B)
        \param maximum  Intensity that should be mapped to white (a scalar)
        """

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

    def mapIntensityToUint8(self, I):
        """Return an array which, when multiplied by an image, returns that image mapped to the range of a
        uint8, [0, 255] (but not converted to uint8)

        The intensity is assumed to have had minimum subtracted (as that can be done per-band)
        """
        with np.errstate(invalid='ignore', divide='ignore'): # n.b. np.where can't and doesn't short-circuit
            return np.where(I <= 0, 0,
                            np.where(I >= self._range, self._uint8Max/I, self._uint8Max/self._range))

class ZScaleMapping(LinearMapping):
    """!A mapping for a linear stretch chosen by the zscale algorithm
    (preserving colours independent of brightness)

    x = (I - minimum)/range
    """

    def __init__(self, image, nSamples=1000, contrast=0.25):
        """!A linear stretch from [z1, z2] chosen by the zscale algorithm
        \param nSamples The number of samples to use to estimate the zscale parameters
        \param contrast The number of samples to use to estimate the zscale parameters
        """

        if not hasattr(image, "getArray"):
            image = afwImage.ImageF(image)
        z1, z2 = getZScale(image, nSamples, contrast)

        LinearMapping.__init__(self, z1, z2, image)

class AsinhMapping(Mapping):
    """!A mapping for an asinh stretch (preserving colours independent of brightness)

    x = asinh(Q (I - minimum)/range)/Q

    This reduces to a linear stretch if Q == 0

    See http://adsabs.harvard.edu/abs/2004PASP..116..133L
    """

    def __init__(self, minimum, range, Q=8):
        Mapping.__init__(self, minimum)

        epsilon = 1.0/2**23            # 32bit floating point machine epsilon; sys.float_info.epsilon is 64bit
        if abs(Q) < epsilon:
            Q = 0.1
        else:
            Qmax = 1e10
            if Q > Qmax:
                Q = Qmax

        if False:
            self._slope = self._uint8Max/Q # gradient at origin is self._slope
        else:
            frac = 0.1                  # gradient estimated using frac*range is _slope
            self._slope = frac*self._uint8Max/np.arcsinh(frac*Q)

        self._soften = Q/float(range);

    def mapIntensityToUint8(self, I):
        """Return an array which, when multiplied by an image, returns that image mapped to the range of a
        uint8, [0, 255] (but not converted to uint8)

        The intensity is assumed to have had minimum subtracted (as that can be done per-band)
        """
        with np.errstate(invalid='ignore', divide='ignore'): # n.b. np.where can't and doesn't short-circuit
            return np.where(I <= 0, 0, np.arcsinh(I*self._soften)*self._slope/I)

class AsinhZScaleMapping(AsinhMapping):
    """!A mapping for an asinh stretch, estimating the linear stretch by zscale

    x = asinh(Q (I - z1)/(z2 - z1))/Q

    See AsinhMapping
    """

    def __init__(self, image, Q=12, pedestal=None):
        """!
        Create an asinh mapping from an image, setting the linear part of the stretch using zscale

        \param image The image to analyse, or a list of 3 images to be converted to an intensity image
        \param Q The asinh softening parameter
        \param pedestal The value, or array of 3 values, to subtract from the images; or None

        N.b. pedestal, if not None, is removed from the images when calculating the zscale
        stretch, and added back into Mapping.minimum[]
        """
        try:
            assert len(image) in (1, 3,), "Please provide 1 or 3 images"
        except TypeError:
            image = [image]

        if pedestal is not None:
            try:
                assert len(pedestal) in (1, 3,), "Please provide 1 or 3 pedestals"
            except TypeError:
                pedestal = 3*[pedestal]

            image = list(image)        # needs to be mutable
            for i, im in enumerate(image):
                if pedestal[i] != 0.0:
                    if hasattr(im, "getImage"):
                        im = im.getImage()
                    if hasattr(im, "getArray"):
                        im = im.getArray()

                    image[i] = im - pedestal[i] # n.b. a copy
        else:
            pedestal = len(image)*[0.0]

        image = computeIntensity(*image)

        zscale = ZScaleMapping(image)
        range = zscale.maximum - zscale.minimum[0] # zscale.minimum is always a triple
        minimum = zscale.minimum

        for i, level in enumerate(pedestal):
            minimum[i] += level

        AsinhMapping.__init__(self, minimum, range, Q)
        self._image = image             # support self.makeRgbImage()

def makeRGB(imageR, imageG=None, imageB=None, minimum=0, range=5, Q=20, fileName=None,
            saturatedBorderWidth=0, saturatedPixelValue=None):
    """Make a set of three images into an RGB image using an asinh stretch and optionally write it to disk

    If saturatedBorderWidth is non-zero, replace saturated pixels with saturatedPixelValue.  Note
    that replacing saturated pixels requires that the input images be MaskedImages.
    """
    if imageG is None:
        imageG = imageR
    if imageB is None:
        imageB = imageR

    if saturatedBorderWidth:
        if saturatedPixelValue is None:
            raise ValueError("saturatedPixelValue must be set if saturatedBorderWidth is set")
        replaceSaturatedPixels(imageR, imageG, imageB, saturatedBorderWidth, saturatedPixelValue)

    asinhMap = AsinhMapping(minimum, range, Q)
    rgb = asinhMap.makeRgbImage(imageR, imageG, imageB)
    if fileName:
        writeRGB(fileName, rgb)

    return rgb

def displayRGB(rgb, show=True):
    """!Display an rgb image using matplotlib
    \param rgb  The RGB image in question
    \param show If true, call plt.show()
    """
    import matplotlib.pyplot as plt
    plt.imshow(rgb, interpolation='nearest', origin="lower")
    if show:
        plt.show()
    return plt

def writeRGB(fileName, rgbImage):
    """!Write an RGB image to disk
    \param fileName The output file.  The suffix defines the format, and must be supported by matplotlib
    \param rgbImage The image, as made by e.g. makeRGB

    Most versions of matplotlib support png and pdf (although the eps/pdf/svg writers may be buggy,
    possibly due an interaction with useTeX=True in the matplotlib settings).

    If your matplotlib bundles pil/pillow you should also be able to write jpeg and tiff files.
    """
    import matplotlib.image
    matplotlib.image.imsave(fileName, rgbImage)
        
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
# Support the legacy API
#
class asinhMappingF(object):
    """!\deprecated Object used to support legacy API"""
    def __init__(self, minimum, range, Q):
        self.minimum = minimum
        self.range = range
        self.Q = Q

class _RgbImageF(object):
    """!\deprecated Object used to support legacy API"""
    def __init__(self, imageR, imageG, imageB, mapping):
        """!\deprecated Legacy API"""
        asinh = AsinhMapping(mapping.minimum, mapping.range, mapping.Q)
        self.rgb = asinh.makeRgbImage(imageR, imageG, imageB)

    def write(self, fileName):
        """!\deprecated Legacy API"""
        writeRGB(fileName, self.rgb)

def RgbImageF(imageR, imageG, imageB, mapping):
    """!\deprecated Legacy API"""
    return _RgbImageF(imageR, imageG, imageB, mapping)
