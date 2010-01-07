#!/usr/bin/env python
"""Demonstrate a segmentation fault
"""
import lsst.afw.image as afwImage

testMaskedImage = afwImage.MaskedImageD(100, 100)
testImage  = testMaskedImage.getImage().get() # no segfault if .get() omitted
addImage = afwImage.ImageD(testMaskedImage.getCols(), testMaskedImage.getRows())
testImage += addImage # no segfault if this step omitted
print "Done"
