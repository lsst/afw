#!/usr/bin/env python
from __future__ import absolute_import, division
from builtins import range

#
# LSST Data Management System
# Copyright 2008, 2009, 2010 LSST Corporation.
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

"""
Tests for PCA on Images

Run with:
   python imagePca.py
or
   python
   >>> import imagePca; imagePca.run()
"""


import unittest
import numpy as np
import random
import math
import itertools

import lsst.utils.tests
import lsst.pex.exceptions as pexExcept
import lsst.afw.geom as afwGeom
import lsst.afw.image.imageLib as afwImage
import lsst.afw.display.utils as displayUtils
import lsst.afw.display.ds9 as ds9

try:
    type(display)
except NameError:
    display = False

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class ImagePcaTestCase(lsst.utils.tests.TestCase):
    """A test case for ImagePca"""

    def setUp(self):
        random.seed(0)
        self.ImageSet = afwImage.ImagePcaF()

    def tearDown(self):
        del self.ImageSet

    def testInnerProducts(self):
        """Test inner products"""

        width, height = 10, 20
        im1 = afwImage.ImageF(afwGeom.Extent2I(width, height))
        val1 = 10
        im1.set(val1)

        im2 = im1.Factory(im1.getDimensions())
        val2 = 20
        im2.set(val2)

        self.assertEqual(afwImage.innerProduct(im1, im1), width*height*val1*val1)
        self.assertEqual(afwImage.innerProduct(im1, im2), width*height*val1*val2)

        im2.set(0, 0, 0)
        self.assertEqual(afwImage.innerProduct(im1, im2), (width*height - 1)*val1*val2)

        im2.set(0, 0, val2)             # reinstate value
        im2.set(width - 1, height - 1, 1)
        self.assertEqual(afwImage.innerProduct(im1, im2), (width*height - 1)*val1*val2 + val1)

    def testAddImages(self):
        """Test adding images to a PCA set"""

        nImage = 3
        for i in range(nImage):
            im = afwImage.ImageF(afwGeom.Extent2I(21, 21))
            val = 1
            im.set(val)

            self.ImageSet.addImage(im, 1.0)

        vec = self.ImageSet.getImageList()
        self.assertEqual(len(vec), nImage)
        self.assertEqual(vec[nImage - 1].get(0, 0), val)

        def tst():
            """Try adding an image with no flux"""
            self.ImageSet.addImage(im, 0.0)

        self.assertRaises(pexExcept.OutOfRangeError, tst)

    def testMean(self):
        """Test calculating mean image"""

        width, height = 10, 20

        values = (100, 200, 300)
        meanVal = 0
        for val in values:
            im = afwImage.ImageF(afwGeom.Extent2I(width, height))
            im.set(val)

            self.ImageSet.addImage(im, 1.0)
            meanVal += val

        meanVal = meanVal/len(values)

        mean = self.ImageSet.getMean()

        self.assertEqual(mean.getWidth(), width)
        self.assertEqual(mean.getHeight(), height)
        self.assertEqual(mean.get(0, 0), meanVal)
        self.assertEqual(mean.get(width - 1, height - 1), meanVal)

    def testPca(self):
        """Test calculating PCA"""
        width, height = 200, 100
        numBases = 3
        numInputs = 3

        bases = []
        for i in range(numBases):
            im = afwImage.ImageF(width, height)
            array = im.getArray()
            x, y = np.indices(array.shape)
            period = 5*(i+1)
            fx = np.sin(2*math.pi/period*x + 2*math.pi/numBases*i)
            fy = np.sin(2*math.pi/period*y + 2*math.pi/numBases*i)
            array[x, y] = fx + fy
            bases.append(im)

        if display:
            mos = displayUtils.Mosaic(background=-10)
            ds9.mtv(mos.makeMosaic(bases), title="Basis functions", frame=1)

        inputs = []
        for i in range(numInputs):
            im = afwImage.ImageF(afwGeom.Extent2I(width, height))
            im.set(0)
            for b in bases:
                im.scaledPlus(random.random(), b)

            inputs.append(im)
            self.ImageSet.addImage(im, 1.0)

        if display:
            mos = displayUtils.Mosaic(background=-10)
            ds9.mtv(mos.makeMosaic(inputs), title="Inputs", frame=2)

        self.ImageSet.analyze()

        eImages = []
        for img in self.ImageSet.getEigenImages():
            eImages.append(img)

        if display:
            mos = displayUtils.Mosaic(background=-10)
            ds9.mtv(mos.makeMosaic(eImages), title="Eigenimages", frame=3)

        self.assertEqual(len(eImages), numInputs)

        # Test for orthogonality
        for i1, i2 in itertools.combinations(list(range(len(eImages))), 2):
            inner = afwImage.innerProduct(eImages[i1], eImages[i2])
            norm1 = eImages[i1].getArray().sum()
            norm2 = eImages[i2].getArray().sum()
            inner /= norm1*norm2
            self.assertAlmostEqual(inner, 0)

    def testPcaNaN(self):
        """Test calculating PCA when the images can contain NaNs"""

        width, height = 20, 10

        values = (100, 200, 300)
        for i, val in enumerate(values):
            im = afwImage.ImageF(afwGeom.Extent2I(width, height))
            im.set(val)

            if i == 1:
                im.set(width//2, height//2, np.nan)

            self.ImageSet.addImage(im, 1.0)

        self.ImageSet.analyze()

        eImages = []
        for img in self.ImageSet.getEigenImages():
            eImages.append(img)

        if display:
            mos = displayUtils.Mosaic(background=-10)
            ds9.mtv(mos.makeMosaic(eImages), frame=1)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass

def setup_module(module):
    lsst.utils.tests.init()

if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
