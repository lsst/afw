#!/usr/bin/env python
"""
Tests for PCA on Images

Run with:
   python ImagePca.py
or
   python
   >>> import ImagePca; ImagePca.run()
"""

import pdb  # we may want to say pdb.set_trace()
import unittest

import lsst.utils.tests as utilsTests
import lsst.pex.exceptions as pexExcept
import lsst.daf.base
import lsst.afw.image.imageLib as afwImage
import lsst.afw.display.utils as displayUtils
import lsst.afw.display.ds9 as ds9

try:
    type(display)
except NameError:
    display = False

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class ImagePcaTestCase(unittest.TestCase):
    """A test case for ImagePca"""

    def setUp(self):
        self.ImageSet = afwImage.ImagePcaF()

    def tearDown(self):
        del self.ImageSet
    
    def testInnerProducts(self):
        """Test inner products"""
        
        width, height = 10, 20
        im1 = afwImage.ImageF(width, height)
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
            im = afwImage.ImageF(21, 21)
            val = 1
            im.set(val)

            self.ImageSet.addImage(im, 1.0)

        vec = self.ImageSet.getImageList()
        self.assertEqual(len(vec), nImage)
        self.assertEqual(vec[nImage - 1].get(0, 0), val)

        def tst():
            """Try adding an image with no flux"""
            self.ImageSet.addImage(im, 0.0)

        utilsTests.assertRaisesLsstCpp(self, pexExcept.OutOfRangeException, tst)
        
    def testMean(self):
        """Test calculating mean image"""

        width, height = 10, 20

        values = (100, 200, 300)
        meanVal = 0
        for val in values:
            im = afwImage.ImageF(width, height)
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

        width, height = 20, 10

        values = (100, 200, 300)
        for val in values:
            im = afwImage.ImageF(width, height)
            im.set(val)

            self.ImageSet.addImage(im, 1.0)
        
        self.ImageSet.analyze()

        eImages = []
        for img in self.ImageSet.getEigenImages():
            eImages.append(img)

        if display:
            mos = displayUtils.Mosaic(background=-10)
            ds9.mtv(mos.makeMosaic(eImages), frame=1)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

def suite():
    """Returns a suite containing all the test cases in this module."""

    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(ImagePcaTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(exit=False):
    """Run the tests"""
    utilsTests.run(suite(), exit)

if __name__ == "__main__":
    run(True)
