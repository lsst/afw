#!/usr/bin/env python
"""Demonstrate a segmentation fault
"""
import lsst.fw.Core.fwLib as fw

testMaskedImage = fw.MaskedImageD(100, 100)
testImage  = testMaskedImage.getImage().get() # no segfault if .get() omitted
addImage = fw.ImageD(testMaskedImage.getCols(), testMaskedImage.getRows())
testImage += addImage # no segfault if this step omitted
print "Done"
