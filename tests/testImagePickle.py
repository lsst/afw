#pybind11##!/usr/bin/env python
#pybind11#from __future__ import absolute_import, division
#pybind11#from builtins import range
#pybind11#
#pybind11##
#pybind11## LSST Data Management System
#pybind11## Copyright 2008-2013 LSST Corporation.
#pybind11##
#pybind11## This product includes software developed by the
#pybind11## LSST Project (http://www.lsst.org/).
#pybind11##
#pybind11## This program is free software: you can redistribute it and/or modify
#pybind11## it under the terms of the GNU General Public License as published by
#pybind11## the Free Software Foundation, either version 3 of the License, or
#pybind11## (at your option) any later version.
#pybind11##
#pybind11## This program is distributed in the hope that it will be useful,
#pybind11## but WITHOUT ANY WARRANTY; without even the implied warranty of
#pybind11## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#pybind11## GNU General Public License for more details.
#pybind11##
#pybind11## You should have received a copy of the LSST License Statement and
#pybind11## the GNU General Public License along with this program.  If not,
#pybind11## see <http://www.lsstcorp.org/LegalNotices/>.
#pybind11##
#pybind11#
#pybind11#import unittest
#pybind11#import pickle
#pybind11#
#pybind11#import numpy
#pybind11#
#pybind11#import lsst.utils.tests
#pybind11#import lsst.afw.coord as afwCoord
#pybind11#import lsst.afw.geom as afwGeom
#pybind11#import lsst.afw.image as afwImage
#pybind11#
#pybind11#
#pybind11#class ImagePickleTestCase(lsst.utils.tests.TestCase):
#pybind11#    """A test case for Image pickling"""
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        self.xSize = 4
#pybind11#        self.ySize = 7
#pybind11#        self.x0 = 567
#pybind11#        self.y0 = 98765
#pybind11#
#pybind11#    def createImage(self, factory=afwImage.ImageF):
#pybind11#        image = factory(self.xSize, self.ySize)
#pybind11#        image.setXY0(afwGeom.Point2I(self.x0, self.y0))
#pybind11#        image.getArray()[:] = self.createPattern()
#pybind11#        return image
#pybind11#
#pybind11#    def createMaskedImage(self, factory=afwImage.MaskedImageF):
#pybind11#        image = factory(self.xSize, self.ySize)
#pybind11#        image.setXY0(afwGeom.Point2I(self.x0, self.y0))
#pybind11#        image.getImage().getArray()[:] = self.createPattern()
#pybind11#        image.getMask().getArray()[:] = self.createPattern()
#pybind11#        image.getVariance().getArray()[:] = self.createPattern()
#pybind11#        return image
#pybind11#
#pybind11#    def createPattern(self):
#pybind11#        yy, xx = numpy.ogrid[0:self.ySize, 0:self.xSize]  # NB: numpy operates 'backwards'
#pybind11#        return self.xSize*yy + xx
#pybind11#
#pybind11#    def assertImagesEqual(self, image, original):
#pybind11#        self.assertEqual(image.__class__.__name__, original.__class__.__name__)
#pybind11#        self.assertEqual(image.getHeight(), original.getHeight())
#pybind11#        self.assertEqual(image.getWidth(), original.getWidth())
#pybind11#        self.assertEqual(image.getY0(), original.getY0())
#pybind11#        self.assertEqual(image.getX0(), original.getX0())
#pybind11#        for x in range(0, original.getWidth()):
#pybind11#            for y in range(0, image.getHeight()):
#pybind11#                self.assertEqual(image.get(x, y), original.get(x, y))
#pybind11#
#pybind11#    def checkImages(self, original):
#pybind11#        image = pickle.loads(pickle.dumps(original))
#pybind11#        self.assertImagesEqual(image, original)
#pybind11#
#pybind11#    def checkExposures(self, original):
#pybind11#        image = pickle.loads(pickle.dumps(original))
#pybind11#        self.assertImagesEqual(image.getMaskedImage(), original.getMaskedImage())
#pybind11#        self.assertEqual(image.getWcs(), original.getWcs())
#pybind11#
#pybind11#    def testImage(self):
#pybind11#        for Image in (afwImage.ImageU,
#pybind11#                      afwImage.ImageI,
#pybind11#                      afwImage.ImageF,
#pybind11#                      afwImage.ImageD,
#pybind11#                      afwImage.MaskU,
#pybind11#                      ):
#pybind11#            image = self.createImage(Image)
#pybind11#            self.checkImages(image)
#pybind11#
#pybind11#    def testMaskedImage(self):
#pybind11#        scale = (1.0*afwGeom.arcseconds).asDegrees()
#pybind11#        wcs = afwImage.makeWcs(afwCoord.Coord(0.0*afwGeom.degrees, 0.0*afwGeom.degrees),
#pybind11#                               afwGeom.Point2D(0.0, 0.0), scale, 0.0, 0.0, scale)
#pybind11#        for MaskedImage in (afwImage.MaskedImageF,
#pybind11#                            afwImage.MaskedImageD,
#pybind11#                            ):
#pybind11#            image = self.createMaskedImage(MaskedImage)
#pybind11#            self.checkImages(image)
#pybind11#            exposure = afwImage.makeExposure(image, wcs)
#pybind11#            self.checkExposures(exposure)
#pybind11#
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#
#pybind11#class TestMemory(lsst.utils.tests.MemoryTestCase):
#pybind11#    pass
#pybind11#
#pybind11#def setup_module(module):
#pybind11#    lsst.utils.tests.init()
#pybind11#
#pybind11#if __name__ == "__main__":
#pybind11#    lsst.utils.tests.init()
#pybind11#    unittest.main()
