#pybind11##!/usr/bin/env python
#pybind11#from __future__ import absolute_import, division
#pybind11#from builtins import range
#pybind11#
#pybind11##
#pybind11## LSST Data Management System
#pybind11## Copyright 2008, 2009, 2010 LSST Corporation.
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
#pybind11#import lsst.utils.tests
#pybind11#import lsst.afw.image as afwImage
#pybind11#
#pybind11#
#pybind11#class ImageIoTestCase(lsst.utils.tests.TestCase):
#pybind11#    """A test case for Image Persistence"""
#pybind11#
#pybind11#    def checkImages(self, image, original):
#pybind11#        # Check that two images are identical
#pybind11#        self.assertEqual(image.getHeight(), original.getHeight())
#pybind11#        self.assertEqual(image.getWidth(), original.getWidth())
#pybind11#        self.assertEqual(image.getY0(), original.getY0())
#pybind11#        self.assertEqual(image.getX0(), original.getX0())
#pybind11#        for x in range(0, original.getWidth()):
#pybind11#            for y in range(0, image.getHeight()):
#pybind11#                self.assertEqual(image.get(x, y), original.get(x, y))
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        # Create the additionalData PropertySet
#pybind11#        self.cols = 4
#pybind11#        self.rows = 4
#pybind11#
#pybind11#    def testIo(self):
#pybind11#        for Image in (afwImage.ImageU,
#pybind11#                      afwImage.ImageL,
#pybind11#                      afwImage.ImageI,
#pybind11#                      afwImage.ImageF,
#pybind11#                      afwImage.ImageD,
#pybind11#                      ):
#pybind11#            image = Image(self.cols, self.rows)
#pybind11#            for x in range(0, self.cols):
#pybind11#                for y in range(0, self.rows):
#pybind11#                    image.set(x, y, x + y)
#pybind11#
#pybind11#            with lsst.utils.tests.getTempFilePath("_%s.fits" % (Image.__name__,)) as filename:
#pybind11#                image.writeFits(filename)
#pybind11#                readImage = Image(filename)
#pybind11#
#pybind11#            self.checkImages(readImage, image)
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#
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
