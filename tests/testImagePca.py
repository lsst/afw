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
#pybind11#"""
#pybind11#Tests for PCA on Images
#pybind11#
#pybind11#Run with:
#pybind11#   python imagePca.py
#pybind11#or
#pybind11#   python
#pybind11#   >>> import imagePca; imagePca.run()
#pybind11#"""
#pybind11#
#pybind11#
#pybind11#import unittest
#pybind11#import numpy as np
#pybind11#import random
#pybind11#import math
#pybind11#import itertools
#pybind11#
#pybind11#import lsst.utils.tests
#pybind11#import lsst.pex.exceptions as pexExcept
#pybind11#import lsst.afw.geom as afwGeom
#pybind11#import lsst.afw.image.imageLib as afwImage
#pybind11#import lsst.afw.display.utils as displayUtils
#pybind11#import lsst.afw.display.ds9 as ds9
#pybind11#
#pybind11#try:
#pybind11#    type(display)
#pybind11#except NameError:
#pybind11#    display = False
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#
#pybind11#
#pybind11#class ImagePcaTestCase(lsst.utils.tests.TestCase):
#pybind11#    """A test case for ImagePca"""
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        random.seed(0)
#pybind11#        self.ImageSet = afwImage.ImagePcaF()
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        del self.ImageSet
#pybind11#
#pybind11#    def testInnerProducts(self):
#pybind11#        """Test inner products"""
#pybind11#
#pybind11#        width, height = 10, 20
#pybind11#        im1 = afwImage.ImageF(afwGeom.Extent2I(width, height))
#pybind11#        val1 = 10
#pybind11#        im1.set(val1)
#pybind11#
#pybind11#        im2 = im1.Factory(im1.getDimensions())
#pybind11#        val2 = 20
#pybind11#        im2.set(val2)
#pybind11#
#pybind11#        self.assertEqual(afwImage.innerProduct(im1, im1), width*height*val1*val1)
#pybind11#        self.assertEqual(afwImage.innerProduct(im1, im2), width*height*val1*val2)
#pybind11#
#pybind11#        im2.set(0, 0, 0)
#pybind11#        self.assertEqual(afwImage.innerProduct(im1, im2), (width*height - 1)*val1*val2)
#pybind11#
#pybind11#        im2.set(0, 0, val2)             # reinstate value
#pybind11#        im2.set(width - 1, height - 1, 1)
#pybind11#        self.assertEqual(afwImage.innerProduct(im1, im2), (width*height - 1)*val1*val2 + val1)
#pybind11#
#pybind11#    def testAddImages(self):
#pybind11#        """Test adding images to a PCA set"""
#pybind11#
#pybind11#        nImage = 3
#pybind11#        for i in range(nImage):
#pybind11#            im = afwImage.ImageF(afwGeom.Extent2I(21, 21))
#pybind11#            val = 1
#pybind11#            im.set(val)
#pybind11#
#pybind11#            self.ImageSet.addImage(im, 1.0)
#pybind11#
#pybind11#        vec = self.ImageSet.getImageList()
#pybind11#        self.assertEqual(len(vec), nImage)
#pybind11#        self.assertEqual(vec[nImage - 1].get(0, 0), val)
#pybind11#
#pybind11#        def tst():
#pybind11#            """Try adding an image with no flux"""
#pybind11#            self.ImageSet.addImage(im, 0.0)
#pybind11#
#pybind11#        self.assertRaises(pexExcept.OutOfRangeError, tst)
#pybind11#
#pybind11#    def testMean(self):
#pybind11#        """Test calculating mean image"""
#pybind11#
#pybind11#        width, height = 10, 20
#pybind11#
#pybind11#        values = (100, 200, 300)
#pybind11#        meanVal = 0
#pybind11#        for val in values:
#pybind11#            im = afwImage.ImageF(afwGeom.Extent2I(width, height))
#pybind11#            im.set(val)
#pybind11#
#pybind11#            self.ImageSet.addImage(im, 1.0)
#pybind11#            meanVal += val
#pybind11#
#pybind11#        meanVal = meanVal/len(values)
#pybind11#
#pybind11#        mean = self.ImageSet.getMean()
#pybind11#
#pybind11#        self.assertEqual(mean.getWidth(), width)
#pybind11#        self.assertEqual(mean.getHeight(), height)
#pybind11#        self.assertEqual(mean.get(0, 0), meanVal)
#pybind11#        self.assertEqual(mean.get(width - 1, height - 1), meanVal)
#pybind11#
#pybind11#    def testPca(self):
#pybind11#        """Test calculating PCA"""
#pybind11#        width, height = 200, 100
#pybind11#        numBases = 3
#pybind11#        numInputs = 3
#pybind11#
#pybind11#        bases = []
#pybind11#        for i in range(numBases):
#pybind11#            im = afwImage.ImageF(width, height)
#pybind11#            array = im.getArray()
#pybind11#            x, y = np.indices(array.shape)
#pybind11#            period = 5*(i+1)
#pybind11#            fx = np.sin(2*math.pi/period*x + 2*math.pi/numBases*i)
#pybind11#            fy = np.sin(2*math.pi/period*y + 2*math.pi/numBases*i)
#pybind11#            array[x, y] = fx + fy
#pybind11#            bases.append(im)
#pybind11#
#pybind11#        if display:
#pybind11#            mos = displayUtils.Mosaic(background=-10)
#pybind11#            ds9.mtv(mos.makeMosaic(bases), title="Basis functions", frame=1)
#pybind11#
#pybind11#        inputs = []
#pybind11#        for i in range(numInputs):
#pybind11#            im = afwImage.ImageF(afwGeom.Extent2I(width, height))
#pybind11#            im.set(0)
#pybind11#            for b in bases:
#pybind11#                im.scaledPlus(random.random(), b)
#pybind11#
#pybind11#            inputs.append(im)
#pybind11#            self.ImageSet.addImage(im, 1.0)
#pybind11#
#pybind11#        if display:
#pybind11#            mos = displayUtils.Mosaic(background=-10)
#pybind11#            ds9.mtv(mos.makeMosaic(inputs), title="Inputs", frame=2)
#pybind11#
#pybind11#        self.ImageSet.analyze()
#pybind11#
#pybind11#        eImages = []
#pybind11#        for img in self.ImageSet.getEigenImages():
#pybind11#            eImages.append(img)
#pybind11#
#pybind11#        if display:
#pybind11#            mos = displayUtils.Mosaic(background=-10)
#pybind11#            ds9.mtv(mos.makeMosaic(eImages), title="Eigenimages", frame=3)
#pybind11#
#pybind11#        self.assertEqual(len(eImages), numInputs)
#pybind11#
#pybind11#        # Test for orthogonality
#pybind11#        for i1, i2 in itertools.combinations(list(range(len(eImages))), 2):
#pybind11#            inner = afwImage.innerProduct(eImages[i1], eImages[i2])
#pybind11#            norm1 = eImages[i1].getArray().sum()
#pybind11#            norm2 = eImages[i2].getArray().sum()
#pybind11#            inner /= norm1*norm2
#pybind11#            self.assertAlmostEqual(inner, 0)
#pybind11#
#pybind11#    def testPcaNaN(self):
#pybind11#        """Test calculating PCA when the images can contain NaNs"""
#pybind11#
#pybind11#        width, height = 20, 10
#pybind11#
#pybind11#        values = (100, 200, 300)
#pybind11#        for i, val in enumerate(values):
#pybind11#            im = afwImage.ImageF(afwGeom.Extent2I(width, height))
#pybind11#            im.set(val)
#pybind11#
#pybind11#            if i == 1:
#pybind11#                im.set(width//2, height//2, np.nan)
#pybind11#
#pybind11#            self.ImageSet.addImage(im, 1.0)
#pybind11#
#pybind11#        self.ImageSet.analyze()
#pybind11#
#pybind11#        eImages = []
#pybind11#        for img in self.ImageSet.getEigenImages():
#pybind11#            eImages.append(img)
#pybind11#
#pybind11#        if display:
#pybind11#            mos = displayUtils.Mosaic(background=-10)
#pybind11#            ds9.mtv(mos.makeMosaic(eImages), frame=1)
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
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
