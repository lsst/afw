import unittest

import lsst.utils.tests
import lsst.afw.image as afwImage


class ExposureInfoTestCase(lsst.utils.tests.TestCase):
    def testDefaultConstructor(self):
        expInfo = afwImage.ExposureInfo()
        self.assertFalse(expInfo.isSurfaceBrightness)
        self.assertFalse(expInfo.isFluence)
        self.assertEqual(expInfo.getImagePhotometricCalibrationType(),
                         afwImage.ImagePhotometricCalibrationType.NOTAPPLICABLE)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
