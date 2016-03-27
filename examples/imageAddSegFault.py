#!/usr/bin/env python

#
# LSST Data Management System
# See the COPYRIGHT and LICENSE files in the top-level directory of this
# package for notices and licensing terms.
#

"""Demonstrate a segmentation fault
"""
import lsst.afw.image as afwImage
import lsst.afw.geom as afwGeom

testMaskedImage = afwImage.MaskedImageD(afwGeom.Extent2I(100, 100))
testImage  = testMaskedImage.getImage().get() # no segfault if .get() omitted
addImage = afwImage.ImageD(testMaskedImage.getCols(), testMaskedImage.getRows())
testImage += addImage # no segfault if this step omitted
print "Done"
