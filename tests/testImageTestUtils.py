#pybind11#import unittest
#pybind11#
#pybind11#import numpy as np
#pybind11#
#pybind11#import lsst.utils.tests
#pybind11#import lsst.afw.geom as afwGeom
#pybind11#import lsst.afw.image as afwImage
#pybind11#from lsst.afw.image.testUtils import makeRampImage
#pybind11#
#pybind11#
#pybind11#class MakeRampImageTestCase(lsst.utils.tests.TestCase):
#pybind11#    """!Unit tests for makeRampImage"""
#pybind11#
#pybind11#    def testUnitInterval(self):
#pybind11#        """!Test small ramp images with unit interval for known values
#pybind11#        """
#pybind11#        for imageClass in (afwImage.ImageU, afwImage.ImageF, afwImage.ImageD):
#pybind11#            dim = afwGeom.Extent2I(7, 9)
#pybind11#            box = afwGeom.Box2I(afwGeom.Point2I(-1, 3), dim)
#pybind11#            numPix = dim[0]*dim[1]
#pybind11#            for start in (-5, 0, 4):
#pybind11#                if imageClass == afwImage.ImageU and start < 0:
#pybind11#                    continue
#pybind11#                predStop = start + numPix - 1  # for integer steps
#pybind11#                for stop in (None, predStop):
#pybind11#                    rampImage = makeRampImage(bbox=box, start=start, stop=predStop, imageClass=imageClass)
#pybind11#                    predArr = np.arange(start, predStop+1)
#pybind11#                    self.assertEqual(len(predArr), numPix)
#pybind11#                    predArr.shape = (dim[1], dim[0])
#pybind11#                    self.assertImagesNearlyEqual(rampImage, predArr)
#pybind11#
#pybind11#    def testNonUnitIntervals(self):
#pybind11#        """!Test a small ramp image with non-integer increments
#pybind11#        """
#pybind11#        for imageClass in (afwImage.ImageU, afwImage.ImageF, afwImage.ImageD):
#pybind11#            dim = afwGeom.Extent2I(7, 9)
#pybind11#            box = afwGeom.Box2I(afwGeom.Point2I(-1, 3), dim)
#pybind11#            numPix = dim[0]*dim[1]
#pybind11#            for start in (-5.1, 0, 4.3):
#pybind11#                if imageClass == afwImage.ImageU and start < 0:
#pybind11#                    continue
#pybind11#                for stop in (7, 1001.5, 5.4):
#pybind11#                    rampImage = makeRampImage(bbox=box, start=start, stop=stop, imageClass=imageClass)
#pybind11#                    dtype = rampImage.getArray().dtype
#pybind11#                    predArr = np.linspace(start, stop, num=numPix, endpoint=True, dtype=dtype)
#pybind11#                    predArr.shape = (dim[1], dim[0])
#pybind11#                    self.assertImagesNearlyEqual(rampImage, predArr)
#pybind11#
#pybind11#
#pybind11#class MemoryTester(lsst.utils.tests.MemoryTestCase):
#pybind11#    pass
#pybind11#
#pybind11#
#pybind11#def setup_module(module):
#pybind11#    lsst.utils.tests.init()
#pybind11#
#pybind11#
#pybind11#if __name__ == "__main__":
#pybind11#    lsst.utils.tests.init()
#pybind11#    unittest.main()
