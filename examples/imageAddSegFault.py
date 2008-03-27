#!/usr/bin/env python
"""Demonstrate a segmentation fault
"""
import lsst.afw as afw

testMaskedImage = afw.image.MaskedImageD(100, 100)
testImage  = testMaskedImage.getImage().get() # no segfault if .get() omitted
addImage = afw.image.ImageD(testMaskedImage.getCols(), testMaskedImage.getRows())
testImage += addImage # no segfault if this step omitted
print "Done"
