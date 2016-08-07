#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

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
import math
import unittest
import re

import numpy as np

import lsst.utils.tests as utilsTests
import lsst.daf.base as dafBase
import lsst.afw.coord as afwCoord
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
from lsst.afw.image.testUtils import imagesDiffer
from lsst.afw.image.basicUtils import _compareWcsOverBBox

class TestTestUtils(utilsTests.TestCase):
    """Test test methods added to lsst.utils.tests.TestCase
    """
    def testAssertAnglesNearlyEqual(self):
        """Test assertAnglesNearlyEqual"""
        for angDeg in (0, 45, -75):
            ang0 = angDeg*afwGeom.degrees
            self.assertAnglesNearlyEqual(
                ang0,
                ang0 + 0.01*afwGeom.arcseconds,
                maxDiff = 0.010001*afwGeom.arcseconds,
            )
            self.assertRaises(AssertionError,
                self.assertAnglesNearlyEqual,
                    ang0,
                    ang0 + 0.01*afwGeom.arcseconds,
                    maxDiff = 0.009999*afwGeom.arcseconds,
            )

            self.assertAnglesNearlyEqual(
                ang0,
                ang0 - 0.01*afwGeom.arcseconds,
                maxDiff = 0.010001*afwGeom.arcseconds,
            )
            self.assertRaises(AssertionError,
                self.assertAnglesNearlyEqual,
                    ang0,
                    ang0 - 0.01*afwGeom.arcseconds,
                    maxDiff = 0.009999*afwGeom.arcseconds,
            )

            self.assertAnglesNearlyEqual(
                ang0 - 720*afwGeom.degrees,
                ang0 + 0.01*afwGeom.arcseconds,
                maxDiff = 0.010001*afwGeom.arcseconds,
            )
            self.assertRaises(AssertionError,
                self.assertAnglesNearlyEqual,
                ang0 - 720*afwGeom.degrees,
                ang0 + 0.01*afwGeom.arcseconds,
                ignoreWrap = False,
                maxDiff = 0.010001*afwGeom.arcseconds,
            )
            self.assertRaises(AssertionError,
                self.assertAnglesNearlyEqual,
                    ang0 - 720*afwGeom.degrees,
                    ang0 + 0.01*afwGeom.arcseconds,
                    maxDiff = 0.009999*afwGeom.arcseconds,
            )

            self.assertAnglesNearlyEqual(
                ang0,
                ang0 + 360*afwGeom.degrees + 0.01*afwGeom.arcseconds,
                maxDiff = 0.010001*afwGeom.arcseconds,
            )
            self.assertRaises(AssertionError,
                self.assertAnglesNearlyEqual,
                    ang0,
                    ang0 + 360*afwGeom.degrees + 0.01*afwGeom.arcseconds,
                    ignoreWrap = False,
                    maxDiff = 0.010001*afwGeom.arcseconds,
            )
            self.assertRaises(AssertionError,
                self.assertAnglesNearlyEqual,
                    ang0,
                    ang0 + 360*afwGeom.degrees + 0.01*afwGeom.arcseconds,
                    maxDiff = 0.009999*afwGeom.arcseconds,
            )

    def testAssertBoxesNearlyEqual(self):
        """Test assertBoxesNearlyEqual"""
        for min0 in ((0, 0), (-1000.5, 5000.1)):
            min0 = afwGeom.Point2D(*min0)
            for extent0 in ((2.01, 3.01), (5432, 2342)):
                extent0 = afwGeom.Extent2D(*extent0)
                box0 = afwGeom.Box2D(min0, extent0)
                self.assertBoxesNearlyEqual(box0, box0, maxDiff=1e-7)
                for deltaExtent in ((0.001, -0.001), (2, -3)):
                    deltaExtent = afwGeom.Extent2D(*deltaExtent)
                    box1 = afwGeom.Box2D(box0.getMin() + deltaExtent, box0.getMax())
                    radDiff = math.hypot(*deltaExtent)
                    self.assertBoxesNearlyEqual(box0, box1, maxDiff=radDiff*1.00001)
                    self.assertRaises(AssertionError, self.assertBoxesNearlyEqual,
                        box0, box1, maxDiff=radDiff*0.99999)

                    box2 = afwGeom.Box2D(box0.getMin() - deltaExtent, box0.getMax())
                    self.assertBoxesNearlyEqual(box0, box2, maxDiff=radDiff*1.00001)
                    self.assertRaises(AssertionError, self.assertBoxesNearlyEqual,
                        box0, box2, maxDiff=radDiff*0.999999)

                    box3 = afwGeom.Box2D(box0.getMin(), box0.getMax() + deltaExtent)
                    self.assertBoxesNearlyEqual(box0, box3, maxDiff=radDiff*1.00001)
                    self.assertRaises(AssertionError, self.assertBoxesNearlyEqual,
                        box0, box3, maxDiff=radDiff*0.999999)

    def testAssertCoordsNearlyEqual(self):
        """Test assertCoordsNearlyEqual"""
        for raDecDeg in ((45, 45), (-70, 89), (130, -89.5)):
            raDecDeg = [val*afwGeom.degrees for val in raDecDeg]
            coord0 = afwCoord.IcrsCoord(*raDecDeg)
            self.assertCoordsNearlyEqual(coord0, coord0, maxDiff=1e-7*afwGeom.arcseconds)

            for offAng in (0, 45, 90):
                offAng = offAng*afwGeom.degrees
                for offDist in (0.001, 0.1):
                    offDist = offDist*afwGeom.arcseconds
                    coord1 = coord0.toGalactic()
                    coord1.offset(offAng, offDist)
                    self.assertCoordsNearlyEqual(coord0, coord1, maxDiff=offDist*1.00001)
                    self.assertRaises(AssertionError,
                        self.assertCoordsNearlyEqual, coord0, coord1, maxDiff=offDist*0.99999)

            # test wraparound in RA
            coord2 = afwCoord.IcrsCoord(raDecDeg[0] + 360*afwGeom.degrees, raDecDeg[1])
            self.assertCoordsNearlyEqual(coord0, coord2, maxDiff=1e-7*afwGeom.arcseconds)

    def testAssertPairsNearlyEqual(self):
        """Test assertPairsNearlyEqual"""
        for pair0 in ((-5, 4), (-5, 0.001), (0, 0), (49, 0.1)):
            self.assertPairsNearlyEqual(pair0, pair0, maxDiff=1e-7)
            self.assertPairsNearlyEqual(afwGeom.Point2D(*pair0), afwGeom.Extent2D(*pair0), maxDiff=1e-7)
            for diff in ((0.001, 0), (-0.01, 0.03)):
                pair1 = [pair0[i] + diff[i] for i in range(2)]
                radialDiff = math.hypot(*diff)
                self.assertPairsNearlyEqual(pair0, pair1, maxDiff=radialDiff+1e-7)
                self.assertRaises(AssertionError,
                    self.assertPairsNearlyEqual,
                        pair0, pair1, maxDiff=radialDiff-1e-7)

    def testAssertWcssNearlyEqualOverBBox(self):
        """Test assertWcsNearlyEqualOverBBox and wcsNearlyEqualOverBBox"""
        bbox = afwGeom.Box2I(afwGeom.Point2I(0, 0), afwGeom.Extent2I(3001, 3001))
        ctrPix = afwGeom.Point2I(1500, 1500)
        metadata = dafBase.PropertySet()
        metadata.set("RADECSYS", "FK5")
        metadata.set("EQUINOX", 2000.0)
        metadata.set("CTYPE1", "RA---TAN")
        metadata.set("CTYPE2", "DEC--TAN")
        metadata.set("CUNIT1", "deg")
        metadata.set("CUNIT2", "deg")
        metadata.set("CRVAL1", 215.5)
        metadata.set("CRVAL2",  53.0)
        metadata.set("CRPIX1", ctrPix[0] + 1)
        metadata.set("CRPIX2", ctrPix[1] + 1)
        metadata.set("CD1_1",  5.1e-05)
        metadata.set("CD1_2",  0.0)
        metadata.set("CD2_2", -5.1e-05)
        metadata.set("CD2_1",  0.0)
        wcs0 = afwImage.cast_TanWcs(afwImage.makeWcs(metadata))
        metadata.set("CRVAL2",  53.000001) # tweak CRVAL2 for wcs1
        wcs1 = afwImage.cast_TanWcs(afwImage.makeWcs(metadata))

        self.assertWcsNearlyEqualOverBBox(wcs0, wcs0, bbox,
            maxDiffSky=1e-7*afwGeom.arcseconds, maxDiffPix=1e-7)
        self.assertTrue(afwImage.wcsNearlyEqualOverBBox(wcs0, wcs0, bbox,
            maxDiffSky=1e-7*afwGeom.arcseconds, maxDiffPix=1e-7))

        self.assertWcsNearlyEqualOverBBox(wcs0, wcs1, bbox,
            maxDiffSky=0.04*afwGeom.arcseconds, maxDiffPix=0.02)
        self.assertTrue(afwImage.wcsNearlyEqualOverBBox(wcs0, wcs1, bbox,
            maxDiffSky=0.04*afwGeom.arcseconds, maxDiffPix=0.02))

        self.assertRaises(AssertionError, self.assertWcsNearlyEqualOverBBox, wcs0, wcs1, bbox,
            maxDiffSky=0.001*afwGeom.arcseconds, maxDiffPix=0.02)
        self.assertFalse(afwImage.wcsNearlyEqualOverBBox(wcs0, wcs1, bbox,
            maxDiffSky=0.001*afwGeom.arcseconds, maxDiffPix=0.02))

        self.assertRaises(AssertionError, self.assertWcsNearlyEqualOverBBox, wcs0, wcs1, bbox,
            maxDiffSky=0.04*afwGeom.arcseconds, maxDiffPix=0.001)
        self.assertFalse(afwImage.wcsNearlyEqualOverBBox(wcs0, wcs1, bbox,
            maxDiffSky=0.04*afwGeom.arcseconds, maxDiffPix=0.001))

        # check that doShortCircuit works in the private implementation
        errStr1 = _compareWcsOverBBox(wcs0, wcs1, bbox,
            maxDiffSky=0.001*afwGeom.arcseconds, maxDiffPix=0.001, doShortCircuit=False)
        errStr2 = _compareWcsOverBBox(wcs0, wcs1, bbox,
            maxDiffSky=0.001*afwGeom.arcseconds, maxDiffPix=0.001, doShortCircuit=True)
        self.assertNotEqual(errStr1, errStr2)

    def checkMaskedImage(self, mi):
        """Run assertImage-like function tests on a masked image

        Compare the masked image to itself, then alter copies and check that the altered copy
        is or is not nearly equal the original, depending on the amount of change, rtol and atol
        """
        epsilon = 1e-5 # margin to avoid roundoff error

        mi0 = mi.Factory(mi, True) # deep copy
        mi1 = mi.Factory(mi, True)

        # a masked image should be exactly equal to itself
        self.assertMaskedImagesEqual(mi0, mi1)
        self.assertMaskedImagesEqual(mi1, mi0)
        self.assertMaskedImagesNearlyEqual(mi0, mi1, atol=0, rtol=0)
        self.assertMaskedImagesNearlyEqual(mi1, mi0, atol=0, rtol=0)
        self.assertMaskedImagesNearlyEqual(mi0.getArrays(), mi1, atol=0, rtol=0)
        self.assertMaskedImagesNearlyEqual(mi0, mi1.getArrays(), atol=0, rtol=0)
        self.assertMaskedImagesNearlyEqual(mi0.getArrays(), mi1.getArrays(), atol=0, rtol=0)
        for getName in ("getImage", "getVariance"):
            plane0 = getattr(mi0, getName)()
            plane1 = getattr(mi1, getName)()
            self.assertImagesEqual(plane0, plane1)
            self.assertImagesEqual(plane1, plane0)
            self.assertImagesNearlyEqual(plane0, plane1, atol=0, rtol=0)
            self.assertImagesNearlyEqual(plane1, plane0, atol=0, rtol=0)
            self.assertImagesNearlyEqual(plane0.getArray(), plane1, atol=0, rtol=0)
            self.assertImagesNearlyEqual(plane0, plane1.getArray(), atol=0, rtol=0)
            self.assertImagesNearlyEqual(plane0.getArray(), plane1.getArray(), atol=0, rtol=0)
            self.assertMasksEqual(plane0, plane1)
            self.assertMasksEqual(plane1, plane0)
            self.assertMasksEqual(plane0.getArray(), plane1)
            self.assertMasksEqual(plane0, plane1.getArray())
            self.assertMasksEqual(plane0.getArray(), plane1.getArray())
        self.assertMasksEqual(mi0.getMask(), mi1.getMask())
        self.assertMasksEqual(mi1.getMask(), mi0.getMask())

        # alter image and variance planes and check the results
        for getName in ("getImage", "getVariance"):
            isFloat = getattr(mi, getName)().getArray().dtype.kind == "f"
            if isFloat:
                for errVal in (np.nan, np.inf, -np.inf):
                    mi0 = mi.Factory(mi, True)
                    mi1 = mi.Factory(mi, True)
                    plane0 = getattr(mi0, getName)()
                    plane1 = getattr(mi1, getName)()
                    plane1[2, 2] = errVal
                    with self.assertRaises(Exception):
                        self.assertImagesNearlyEqual(plane0, plane1)
                    with self.assertRaises(Exception):
                        self.assertImagesNearlyEqual(plane0.getArray(), plane1)
                    with self.assertRaises(Exception):
                        self.assertImagesNearlyEqual(plane1, plane0)
                    with self.assertRaises(Exception):
                        self.assertMaskedImagesNearlyEqual(mi0, mi1)
                    with self.assertRaises(Exception):
                        self.assertMaskedImagesNearlyEqual(mi0, mi1.getArrays())
                    with self.assertRaises(Exception):
                        self.assertMaskedImagesNearlyEqual(mi1, mi0)

                    skipMask = mi.getMask().Factory(mi.getMask(), True)
                    skipMaskArr = skipMask.getArray()
                    skipMaskArr[:] = 0
                    skipMaskArr[2, 2] = 1
                    self.assertImagesNearlyEqual(plane0, plane1, skipMask=skipMaskArr, atol=0, rtol=0)
                    self.assertImagesNearlyEqual(plane0, plane1, skipMask=skipMask, atol=0, rtol=0)
                    self.assertMaskedImagesNearlyEqual(mi0, mi1, skipMask=skipMaskArr, atol=0, rtol=0)
                    self.assertMaskedImagesNearlyEqual(mi0, mi1, skipMask=skipMask, atol=0, rtol=0)

                for dval in (0.001, 0.03):
                    mi0 = mi.Factory(mi, True)
                    mi1 = mi.Factory(mi, True)
                    plane0 = getattr(mi0, getName)()
                    plane1 = getattr(mi1, getName)()
                    plane1[2, 2] += dval
                    val1 = plane1.get(2, 2)
                    self.assertImagesNearlyEqual(plane0, plane1, rtol=0, atol=dval + epsilon)
                    self.assertImagesNearlyEqual(plane0, plane1, rtol=dval/val1 + epsilon, atol=0)
                    self.assertMaskedImagesNearlyEqual(mi0, mi1, rtol=0, atol=dval + epsilon)
                    self.assertMaskedImagesNearlyEqual(mi1, mi0, rtol=0, atol=dval + epsilon)
                    with self.assertRaises(Exception):
                        self.assertImagesNearlyEqual(plane0, plane1, rtol=0, atol=dval - epsilon)
                    with self.assertRaises(Exception):
                        self.assertImagesNearlyEqual(plane0, plane1, rtol=dval/val1 - epsilon, atol=0)
                    with self.assertRaises(Exception):
                        self.assertMaskedImagesNearlyEqual(mi0, mi1, rtol=0, atol=dval - epsilon)
                    with self.assertRaises(Exception):
                        self.assertMaskedImagesNearlyEqual(mi0, mi1, rtol=dval/val1 - epsilon, atol=0)
            else:
                # plane is an integer of some type
                for dval in (1, 3):
                    mi0 = mi.Factory(mi, True)
                    mi1 = mi.Factory(mi, True)
                    plane0 = getattr(mi0, getName)()
                    plane1 = getattr(mi1, getName)()
                    plane1[2, 2] += dval
                    val1 = plane1.get(2, 2)
                    # int value and test is <= so epsilon not required for atol
                    # but rtol is a fraction, so epsilon is still safest for the rtol test
                    self.assertImagesNearlyEqual(plane0, plane1, rtol=0, atol=dval)
                    self.assertImagesNearlyEqual(plane0, plane1, rtol=dval/val1 + epsilon, atol=0)
                    with self.assertRaises(Exception):
                        self.assertImagesNearlyEqual(plane0, plane1, rtol=0, atol=dval - epsilon)
                    with self.assertRaises(Exception):
                        self.assertImagesNearlyEqual(plane0, plane1, rtol=dval/val1 - epsilon, atol=0)

        # alter mask and check the results
        mi0 = mi.Factory(mi, True)
        mi1 = mi.Factory(mi, True)
        mask0 = mi0.getMask()
        mask1 = mi1.getMask()
        for dval in (1, 3):
            mask1.getArray()[2, 2] += 1 # getArray avoids "unsupported operand type" failure
            with self.assertRaises(Exception):
                self.assertMasksEqual(mask0, mask1)
            with self.assertRaises(Exception):
                self.assertMasksEqual(mask1, mask0)
            with self.assertRaises(Exception):
                self.assertMaskedImagesEqual(mi0, mi1)
            with self.assertRaises(Exception):
                self.assertMaskedImagesEqual(mi1, mi0)

        skipMask = mi.getMask().Factory(mi.getMask(), True)
        skipMaskArr = skipMask.getArray()
        skipMaskArr[:] = 0
        skipMaskArr[2, 2] = 1
        self.assertMasksEqual(mask0, mask1, skipMask=skipMaskArr)
        self.assertMasksEqual(mask0, mask1, skipMask=skipMask)
        self.assertMaskedImagesNearlyEqual(mi0, mi1, skipMask=skipMaskArr, atol=0, rtol=0)
        self.assertMaskedImagesNearlyEqual(mi0, mi1, skipMask=skipMask, atol=0, rtol=0)

    def testAssertImagesNearlyEqual(self):
        """Test assertImagesNearlyEqual, assertMasksNearlyEqual and assertMaskedImagesNearlyEqual
        """
        width = 10
        height = 9

        for miType in (afwImage.MaskedImageF, afwImage.MaskedImageD, afwImage.MaskedImageI,
            afwImage.MaskedImageU):
            mi = makeRampMaskedImageWithNans(width, height, miType)
            self.checkMaskedImage(mi)

        for invalidType in (np.zeros([width+1, height]), str, self.assertRaises):
            mi = makeRampMaskedImageWithNans(width, height, miType)
            with self.assertRaises(TypeError):
                self.assertMasksEqual(mi.getMask(), invalidType)
            with self.assertRaises(TypeError):
                self.assertMasksEqual(invalidType, mi.getMask())
            with self.assertRaises(TypeError):
                self.assertMasksEqual(mi.getMask(), mi.getMask(), skipMask=invalidType)

            with self.assertRaises(TypeError):
                self.assertImagesNearlyEqual(mi.getImage(), invalidType)
            with self.assertRaises(TypeError):
                self.assertImagesNearlyEqual(invalidType, mi.getImage())
            with self.assertRaises(TypeError):
                self.assertImagesNearlyEqual(mi.getImage(), mi.getImage(), skipMask=invalidType)

            with self.assertRaises(TypeError):
                self.assertMaskedImagesNearlyEqual(mi, invalidType)
            with self.assertRaises(TypeError):
                self.assertMaskedImagesNearlyEqual(invalidType, mi)
            with self.assertRaises(TypeError):
                self.assertMaskedImagesNearlyEqual(mi, mi, skipMask=invalidType)

            with self.assertRaises(TypeError):
                self.assertMaskedImagesNearlyEqual(mi.getImage(), mi.getImage())


    def testUnsignedImages(self):
        """Unsigned images can give incorrect differences unless the test code is careful
        """
        image0 = np.zeros([5, 5], dtype=np.uint8)
        image1 = np.zeros([5, 5], dtype=np.uint8)
        image0[0,0] = 1
        image1[0,1] = 2

        # arrays differ by a maximum of 2
        errMsg1 = imagesDiffer(image0, image1)
        match = re.match(r"maxDiff *= *(\d+)", errMsg1, re.IGNORECASE)
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), "2")

        # arrays are equal to within 5
        self.assertImagesNearlyEqual(image0, image1, atol=5)


def makeRampMaskedImageWithNans(width, height, imgClass=afwImage.MaskedImageF):
    """Make a masked image that is a ramp with additional non-finite values

    Make a masked image with the following additional non-finite values
    in the variance plane and (if image is of some floating type) image plane:
    - nan at [0, 0]
    - inf at [1, 0]
    - -inf at [0, 1]
    """
    mi = makeRampMaskedImage(width, height, imgClass)

    var = mi.getVariance()
    var[0, 0] = np.nan
    var[1, 0] = np.inf
    var[0, 1] = -np.inf

    im = mi.getImage()
    try:
        np.array([np.nan], dtype=im.getArray().dtype)
    except Exception:
        # image plane does not support nan, etc. (presumably an int of some variety)
        pass
    else:
        # image plane does support nan, etc.
        im[0, 0] = np.nan
        im[1, 0] = np.inf
        im[0, 1] = -np.inf
    return mi

def makeRampMaskedImage(width, height, imgClass=afwImage.MaskedImageF):
    """Make a ramp image of the specified size and image class

    Image values start from 0 at the lower left corner and increase by 1 along rows
    Variance values equal image values + 100
    Mask values equal image values modulo 8 bits (leaving plenty of unused values)
    """
    mi = imgClass(width, height)
    image = mi.getImage()
    mask = mi.getMask()
    variance = mi.getVariance()
    val = 0
    for yInd in range(height):
        for xInd in range(width):
            image.set(xInd, yInd, val)
            variance.set(xInd, yInd, val + 100)
            mask.set(xInd, yInd, val % 0x100)
            val += 1
    return mi


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(TestTestUtils)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)

    return unittest.TestSuite(suites)

def run(exit=False):
    """Run the tests"""
    utilsTests.run(suite(), exit)

if __name__ == "__main__":
    run(True)
