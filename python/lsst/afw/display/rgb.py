import numpy as np

from lsst.afw.display.displayLib import replaceSaturatedPixels

class Mapping(object):
    """!Baseclass to map red, blue, green intensities into uint8 values"""
    
    def __init__(self, min):
        """!Create a mapping
        \param min  Intensity that should be mapped to black (a scalar or array for R, G, B)
        """
        
        self._uint8Max = float(np.iinfo(np.uint8).max)

        try:
            len(min)
        except:
            min = 3*[min]
        assert len(min) == 3, "Please provide 1 or 3 values for min"

        self._min = min

    def makeRgbImage(self, imageR, imageG, imageB):
        """!Convert 3 arrays, imageR, imageG, and imageB into a numpy RGB image

        N.b. images may be afwImages or numpy arrays
        """
        imageRGB = [imageR, imageG, imageB]
        for i, c in enumerate(imageRGB):
            if hasattr(c, "getImage"):
                c = imageRGB[i] = c.getImage()
            if hasattr(c, "getArray"):
                imageRGB[i] = c.getArray()

        return np.flipud(np.dstack(self._convertImagesToUint8(*imageRGB)).astype(np.uint8))

    def intensity(self, imageR, imageG, imageB):
        """!Return the total intensity from the red, blue, and green intensities"""
        return (imageR + imageG + imageB)/float(3);

    def mapIntensityToUint8(self, I):
        """Map an intensity into the range of a uint8, [0, 255] (but not converted to uint8)"""
        return np.where(I <= 0, 0, np.where(I < self._uint8Max, I, self._uint8Max))

    def _convertImagesToUint8(self, imageR, imageG, imageB):
        """Use the mapping to convert images imageR, imageG, and imageB to a triplet of uint8 images"""
        imageR = imageR - self._min[0]  # n.b. makes copy
        imageG = imageG - self._min[1]
        imageB = imageB - self._min[2]

        fac = self.mapIntensityToUint8(self.intensity(imageR, imageG, imageB))

        imageRGB = [imageR, imageG, imageB]

        for c in imageRGB:
            c *= fac
            c[c <= 0] = 0

        pixmax = self._uint8Max
        r0, g0, b0 = imageRGB           # copies -- could work row by row to minimise memory usage

        with np.errstate(invalid='ignore', divide='ignore'): # n.b. np.where doesn't (and can't) short-circuit
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

class AsinhMapping(Mapping):
    """!A mapping for an asinh stretch (preserving colours independent of brightness)

    x = asinh(Q (I - min)/range)/Q

    This reduces to a linear stretch if Q == 0

    See http://adsabs.harvard.edu/abs/2004PASP..116..133L
    """

    def __init__(self, min, range, Q):
        Mapping.__init__(self, min)

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
        return np.where(I <= 0, 0, np.arcsinh(I*self._soften)*self._slope/I)

def makeRGB(imageR, imageG, imageB, min=0, range=5, Q=20, fileName=None,
            saturatedBorderWidth=0, saturatedPixelValue=None):
    """Make a set of three images into an RGB image using an asinh stretch and optionally write it to disk

    If saturatedBorderWidth is non-zero, replace saturated pixels with saturatedPixelValue.  Note
    that replacing saturated pixels requires that the input images be MaskedImages.
    """
    if saturatedBorderWidth:
        if saturatedPixelValue is None:
            raise ValueError("saturatedPixelValue must be set if saturatedBorderWidth is set")
        replaceSaturatedPixels(imageR, imageG, imageB, saturatedBorderWidth, saturatedPixelValue)

    asinhMap = AsinhMapping(min, range, Q)
    rgb = asinhMap.makeRgbImage(imageR, imageG, imageB)
    if fileName:
        writeRGB(fileName, rgb)

    return rgb

def displayRGB(rgb):
    """Display an rgb image using matplotlib"""
    import matplotlib.pyplot as plt
    plt.imshow(rgb, interpolation='nearest')
    plt.show()

def writeRGB(fileName, rgbImage):
    """Write an RGB image (made by e.g. makeRGB) to fileName"""
    import matplotlib.image
    matplotlib.image.imsave(fileName, rgbImage)
        
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
# Support the legacy API
#
class asinhMappingF(object):
    def __init__(self, min, range, Q):
        self.min = min
        self.range = range
        self.Q = Q

class _RgbImageF(object):
    def __init__(self, imageR, imageG, imageB, mapping):
        asinh = AsinhMapping(mapping.min, mapping.range, mapping.Q)
        self.rgb = asinh.makeRgbImage(imageR, imageG, imageB)

    def write(self, fileName):
        writeRGB(fileName, self.rgb)

def RgbImageF(imageR, imageG, imageB, mapping):
    return _RgbImageF(imageR, imageG, imageB, mapping)
