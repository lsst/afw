import unittest

import numpy as np

import lsst.utils.tests
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
from lsst.afw.image.testUtils import makeRampImage


class MakeRampImageTestCase(lsst.utils.tests.TestCase):
    """!Unit tests for makeRampImage"""

    def testUnitInterval(self):
        """!Test small ramp images with unit interval for known values
        """
        for imageClass in (afwImage.ImageU, afwImage.ImageF, afwImage.ImageD):
            dim = afwGeom.Extent2I(7, 9)
            box = afwGeom.Box2I(afwGeom.Point2I(-1, 3), dim)
            numPix = dim[0]*dim[1]
            for start in (-5, 0, 4):
                if imageClass == afwImage.ImageU and start < 0:
                    continue
                predStop = start + numPix - 1  # for integer steps
                for stop in (None, predStop):
                    rampImage = makeRampImage(
                        bbox=box, start=start, stop=predStop, imageClass=imageClass)
                    predArr = np.arange(start, predStop+1)
                    self.assertEqual(len(predArr), numPix)
                    predArr.shape = (dim[1], dim[0])
                    self.assertImagesAlmostEqual(rampImage, predArr)

    def testNonUnitIntervals(self):
        """!Test a small ramp image with non-integer increments
        """
        for imageClass in (afwImage.ImageU, afwImage.ImageF, afwImage.ImageD):
            dim = afwGeom.Extent2I(7, 9)
            box = afwGeom.Box2I(afwGeom.Point2I(-1, 3), dim)
            numPix = dim[0]*dim[1]
            for start in (-5.1, 0, 4.3):
                if imageClass == afwImage.ImageU and start < 0:
                    continue
                for stop in (7, 1001.5, 5.4):
                    rampImage = makeRampImage(
                        bbox=box, start=start, stop=stop, imageClass=imageClass)
                    dtype = rampImage.getArray().dtype
                    predArr = np.linspace(
                        start, stop, num=numPix, endpoint=True, dtype=dtype)
                    predArr.shape = (dim[1], dim[0])
                    self.assertImagesAlmostEqual(rampImage, predArr)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
