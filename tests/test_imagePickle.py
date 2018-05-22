#
# LSST Data Management System
# Copyright 2008-2013 LSST Corporation.
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
# see <http://www.lsstcorp.org/LegalNotices/>.
#

import unittest
import pickle

import numpy as np

import lsst.utils.tests
import lsst.geom
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage


class ImagePickleTestCase(lsst.utils.tests.TestCase):
    """A test case for Image pickling"""

    def setUp(self):
        self.xSize = 4
        self.ySize = 7
        self.x0 = 567
        self.y0 = 98765

    def createImage(self, factory=afwImage.ImageF):
        image = factory(self.xSize, self.ySize)
        image.setXY0(lsst.geom.Point2I(self.x0, self.y0))
        image.getArray()[:] = self.createPattern()
        return image

    def createMaskedImage(self, factory=afwImage.MaskedImageF):
        image = factory(self.xSize, self.ySize)
        image.setXY0(lsst.geom.Point2I(self.x0, self.y0))
        image.getImage().getArray()[:] = self.createPattern()
        image.getMask().getArray()[:] = self.createPattern()
        image.getVariance().getArray()[:] = self.createPattern()
        return image

    def createPattern(self):
        # NB: numpy operates 'backwards'
        yy, xx = np.ogrid[0:self.ySize, 0:self.xSize]
        return self.xSize*yy + xx

    def assertImagesEqual(self, image, original):
        self.assertEqual(image.__class__.__name__, original.__class__.__name__)
        self.assertEqual(image.getHeight(), original.getHeight())
        self.assertEqual(image.getWidth(), original.getWidth())
        self.assertEqual(image.getY0(), original.getY0())
        self.assertEqual(image.getX0(), original.getX0())
        for x in range(0, original.getWidth()):
            for y in range(0, image.getHeight()):
                self.assertEqual(image.get(x, y), original.get(x, y))

    def checkImages(self, original):
        image = pickle.loads(pickle.dumps(original))
        self.assertImagesEqual(image, original)

    def checkExposures(self, original):
        image = pickle.loads(pickle.dumps(original))
        self.assertImagesEqual(image.getMaskedImage(),
                               original.getMaskedImage())
        self.assertEqual(image.getWcs(), original.getWcs())

    def testImage(self):
        for Image in (afwImage.ImageU,
                      afwImage.ImageI,
                      afwImage.ImageF,
                      afwImage.ImageD,
                      afwImage.Mask,
                      ):
            image = self.createImage(Image)
            self.checkImages(image)

    def testMaskedImage(self):
        scale = 1.0*lsst.geom.arcseconds
        wcs = afwGeom.makeSkyWcs(crval=lsst.geom.SpherePoint(0.0*lsst.geom.degrees, 0.0*lsst.geom.degrees),
                                 crpix=lsst.geom.Point2D(0.0, 0.0),
                                 cdMatrix=afwGeom.makeCdMatrix(scale=scale))
        for MaskedImage in (afwImage.MaskedImageF,
                            afwImage.MaskedImageD,
                            ):
            image = self.createMaskedImage(MaskedImage)
            self.checkImages(image)
            exposure = afwImage.makeExposure(image, wcs)
            self.checkExposures(exposure)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
