#pybind11##!/usr/bin/env python
#pybind11#from __future__ import absolute_import, division, print_function
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
#pybind11#import math
#pybind11#import unittest
#pybind11#import re
#pybind11#
#pybind11#import numpy as np
#pybind11#
#pybind11#import lsst.utils.tests
#pybind11#import lsst.daf.base as dafBase
#pybind11#import lsst.afw.coord as afwCoord
#pybind11#import lsst.afw.geom as afwGeom
#pybind11#import lsst.afw.image as afwImage
#pybind11#from lsst.afw.image.testUtils import imagesDiffer
#pybind11#from lsst.afw.image.basicUtils import _compareWcsOverBBox
#pybind11#
#pybind11#
#pybind11#class TestTestUtils(lsst.utils.tests.TestCase):
#pybind11#    """Test test methods added to lsst.utils.tests.TestCase
#pybind11#    """
#pybind11#
#pybind11#    def testAssertAnglesNearlyEqual(self):
#pybind11#        """Test assertAnglesNearlyEqual"""
#pybind11#        for angDeg in (0, 45, -75):
#pybind11#            ang0 = angDeg*afwGeom.degrees
#pybind11#            self.assertAnglesNearlyEqual(
#pybind11#                ang0,
#pybind11#                ang0 + 0.01*afwGeom.arcseconds,
#pybind11#                maxDiff=0.010001*afwGeom.arcseconds,
#pybind11#            )
#pybind11#            with self.assertRaises(AssertionError):
#pybind11#                self.assertAnglesNearlyEqual(
#pybind11#                    ang0,
#pybind11#                    ang0 + 0.01*afwGeom.arcseconds,
#pybind11#                    maxDiff=0.009999*afwGeom.arcseconds,
#pybind11#                    )
#pybind11#
#pybind11#            self.assertAnglesNearlyEqual(
#pybind11#                ang0,
#pybind11#                ang0 - 0.01*afwGeom.arcseconds,
#pybind11#                maxDiff=0.010001*afwGeom.arcseconds,
#pybind11#            )
#pybind11#            with self.assertRaises(AssertionError):
#pybind11#                self.assertAnglesNearlyEqual(
#pybind11#                    ang0,
#pybind11#                    ang0 - 0.01*afwGeom.arcseconds,
#pybind11#                    maxDiff=0.009999*afwGeom.arcseconds,
#pybind11#                    )
#pybind11#
#pybind11#            self.assertAnglesNearlyEqual(
#pybind11#                ang0 - 720*afwGeom.degrees,
#pybind11#                ang0 + 0.01*afwGeom.arcseconds,
#pybind11#                maxDiff=0.010001*afwGeom.arcseconds,
#pybind11#            )
#pybind11#            with self.assertRaises(AssertionError):
#pybind11#                self.assertAnglesNearlyEqual(
#pybind11#                    ang0 - 720*afwGeom.degrees,
#pybind11#                    ang0 + 0.01*afwGeom.arcseconds,
#pybind11#                    ignoreWrap=False,
#pybind11#                    maxDiff=0.010001*afwGeom.arcseconds,
#pybind11#                    )
#pybind11#            with self.assertRaises(AssertionError):
#pybind11#                self.assertAnglesNearlyEqual(
#pybind11#                    ang0 - 720*afwGeom.degrees,
#pybind11#                    ang0 + 0.01*afwGeom.arcseconds,
#pybind11#                    maxDiff=0.009999*afwGeom.arcseconds,
#pybind11#                    )
#pybind11#
#pybind11#            self.assertAnglesNearlyEqual(
#pybind11#                ang0,
#pybind11#                ang0 + 360*afwGeom.degrees + 0.01*afwGeom.arcseconds,
#pybind11#                maxDiff=0.010001*afwGeom.arcseconds,
#pybind11#            )
#pybind11#            with self.assertRaises(AssertionError):
#pybind11#                self.assertAnglesNearlyEqual(
#pybind11#                    ang0,
#pybind11#                    ang0 + 360*afwGeom.degrees + 0.01*afwGeom.arcseconds,
#pybind11#                    ignoreWrap=False,
#pybind11#                    maxDiff=0.010001*afwGeom.arcseconds,
#pybind11#                    )
#pybind11#            with self.assertRaises(AssertionError):
#pybind11#                self.assertAnglesNearlyEqual(
#pybind11#                    ang0,
#pybind11#                    ang0 + 360*afwGeom.degrees + 0.01*afwGeom.arcseconds,
#pybind11#                    maxDiff=0.009999*afwGeom.arcseconds,
#pybind11#                    )
#pybind11#
#pybind11#    def testAssertBoxesNearlyEqual(self):
#pybind11#        """Test assertBoxesNearlyEqual"""
#pybind11#        for min0 in ((0, 0), (-1000.5, 5000.1)):
#pybind11#            min0 = afwGeom.Point2D(*min0)
#pybind11#            for extent0 in ((2.01, 3.01), (5432, 2342)):
#pybind11#                extent0 = afwGeom.Extent2D(*extent0)
#pybind11#                box0 = afwGeom.Box2D(min0, extent0)
#pybind11#                self.assertBoxesNearlyEqual(box0, box0, maxDiff=1e-7)
#pybind11#                for deltaExtent in ((0.001, -0.001), (2, -3)):
#pybind11#                    deltaExtent = afwGeom.Extent2D(*deltaExtent)
#pybind11#                    box1 = afwGeom.Box2D(box0.getMin() + deltaExtent, box0.getMax())
#pybind11#                    radDiff = math.hypot(*deltaExtent)
#pybind11#                    self.assertBoxesNearlyEqual(box0, box1, maxDiff=radDiff*1.00001)
#pybind11#                    with self.assertRaises(AssertionError):
#pybind11#                        self.assertBoxesNearlyEqual(
#pybind11#                            box0, box1, maxDiff=radDiff*0.99999)
#pybind11#
#pybind11#                    box2 = afwGeom.Box2D(box0.getMin() - deltaExtent, box0.getMax())
#pybind11#                    self.assertBoxesNearlyEqual(box0, box2, maxDiff=radDiff*1.00001)
#pybind11#                    with self.assertRaises(AssertionError):
#pybind11#                        self.assertBoxesNearlyEqual(
#pybind11#                            box0, box2, maxDiff=radDiff*0.999999)
#pybind11#
#pybind11#                    box3 = afwGeom.Box2D(box0.getMin(), box0.getMax() + deltaExtent)
#pybind11#                    self.assertBoxesNearlyEqual(box0, box3, maxDiff=radDiff*1.00001)
#pybind11#                    with self.assertRaises(AssertionError):
#pybind11#                        self.assertBoxesNearlyEqual(
#pybind11#                            box0, box3, maxDiff=radDiff*0.999999)
#pybind11#
#pybind11#    def testAssertCoordsNearlyEqual(self):
#pybind11#        """Test assertCoordsNearlyEqual"""
#pybind11#        for raDecDeg in ((45, 45), (-70, 89), (130, -89.5)):
#pybind11#            raDecDeg = [val*afwGeom.degrees for val in raDecDeg]
#pybind11#            coord0 = afwCoord.IcrsCoord(*raDecDeg)
#pybind11#            self.assertCoordsNearlyEqual(coord0, coord0, maxDiff=1e-7*afwGeom.arcseconds)
#pybind11#
#pybind11#            for offAng in (0, 45, 90):
#pybind11#                offAng = offAng*afwGeom.degrees
#pybind11#                for offDist in (0.001, 0.1):
#pybind11#                    offDist = offDist*afwGeom.arcseconds
#pybind11#                    coord1 = coord0.toGalactic()
#pybind11#                    coord1.offset(offAng, offDist)
#pybind11#                    self.assertCoordsNearlyEqual(coord0, coord1, maxDiff=offDist*1.00001)
#pybind11#                    with self.assertRaises(AssertionError):
#pybind11#                        self.assertCoordsNearlyEqual(coord0, coord1, maxDiff=offDist*0.99999)
#pybind11#
#pybind11#            # test wraparound in RA
#pybind11#            coord2 = afwCoord.IcrsCoord(raDecDeg[0] + 360*afwGeom.degrees, raDecDeg[1])
#pybind11#            self.assertCoordsNearlyEqual(coord0, coord2, maxDiff=1e-7*afwGeom.arcseconds)
#pybind11#
#pybind11#    def testAssertPairsNearlyEqual(self):
#pybind11#        """Test assertPairsNearlyEqual"""
#pybind11#        for pair0 in ((-5, 4), (-5, 0.001), (0, 0), (49, 0.1)):
#pybind11#            self.assertPairsNearlyEqual(pair0, pair0, maxDiff=1e-7)
#pybind11#            self.assertPairsNearlyEqual(afwGeom.Point2D(*pair0), afwGeom.Extent2D(*pair0), maxDiff=1e-7)
#pybind11#            for diff in ((0.001, 0), (-0.01, 0.03)):
#pybind11#                pair1 = [pair0[i] + diff[i] for i in range(2)]
#pybind11#                radialDiff = math.hypot(*diff)
#pybind11#                self.assertPairsNearlyEqual(pair0, pair1, maxDiff=radialDiff+1e-7)
#pybind11#                with self.assertRaises(AssertionError):
#pybind11#                    self.assertPairsNearlyEqual(pair0, pair1, maxDiff=radialDiff-1e-7)
#pybind11#
#pybind11#    def testAssertWcssNearlyEqualOverBBox(self):
#pybind11#        """Test assertWcsNearlyEqualOverBBox and wcsNearlyEqualOverBBox"""
#pybind11#        bbox = afwGeom.Box2I(afwGeom.Point2I(0, 0), afwGeom.Extent2I(3001, 3001))
#pybind11#        ctrPix = afwGeom.Point2I(1500, 1500)
#pybind11#        metadata = dafBase.PropertySet()
#pybind11#        metadata.set("RADECSYS", "FK5")
#pybind11#        metadata.set("EQUINOX", 2000.0)
#pybind11#        metadata.set("CTYPE1", "RA---TAN")
#pybind11#        metadata.set("CTYPE2", "DEC--TAN")
#pybind11#        metadata.set("CUNIT1", "deg")
#pybind11#        metadata.set("CUNIT2", "deg")
#pybind11#        metadata.set("CRVAL1", 215.5)
#pybind11#        metadata.set("CRVAL2", 53.0)
#pybind11#        metadata.set("CRPIX1", ctrPix[0] + 1)
#pybind11#        metadata.set("CRPIX2", ctrPix[1] + 1)
#pybind11#        metadata.set("CD1_1", 5.1e-05)
#pybind11#        metadata.set("CD1_2", 0.0)
#pybind11#        metadata.set("CD2_2", -5.1e-05)
#pybind11#        metadata.set("CD2_1", 0.0)
#pybind11#        wcs0 = afwImage.cast_TanWcs(afwImage.makeWcs(metadata))
#pybind11#        metadata.set("CRVAL2", 53.000001)  # tweak CRVAL2 for wcs1
#pybind11#        wcs1 = afwImage.cast_TanWcs(afwImage.makeWcs(metadata))
#pybind11#
#pybind11#        self.assertWcsNearlyEqualOverBBox(wcs0, wcs0, bbox,
#pybind11#                                          maxDiffSky=1e-7*afwGeom.arcseconds, maxDiffPix=1e-7)
#pybind11#        self.assertTrue(afwImage.wcsNearlyEqualOverBBox(wcs0, wcs0, bbox,
#pybind11#                                                        maxDiffSky=1e-7*afwGeom.arcseconds, maxDiffPix=1e-7))
#pybind11#
#pybind11#        self.assertWcsNearlyEqualOverBBox(wcs0, wcs1, bbox,
#pybind11#                                          maxDiffSky=0.04*afwGeom.arcseconds, maxDiffPix=0.02)
#pybind11#        self.assertTrue(afwImage.wcsNearlyEqualOverBBox(wcs0, wcs1, bbox,
#pybind11#                                                        maxDiffSky=0.04*afwGeom.arcseconds, maxDiffPix=0.02))
#pybind11#
#pybind11#        with self.assertRaises(AssertionError):
#pybind11#            self.assertWcsNearlyEqualOverBBox(wcs0, wcs1, bbox,
#pybind11#                          maxDiffSky=0.001*afwGeom.arcseconds, maxDiffPix=0.02)
#pybind11#        self.assertFalse(afwImage.wcsNearlyEqualOverBBox(wcs0, wcs1, bbox,
#pybind11#                                                         maxDiffSky=0.001*afwGeom.arcseconds, maxDiffPix=0.02))
#pybind11#
#pybind11#        with self.assertRaises(AssertionError):
#pybind11#            self.assertWcsNearlyEqualOverBBox(wcs0, wcs1, bbox,
#pybind11#                maxDiffSky=0.04*afwGeom.arcseconds, maxDiffPix=0.001)
#pybind11#        self.assertFalse(afwImage.wcsNearlyEqualOverBBox(wcs0, wcs1, bbox,
#pybind11#                                                         maxDiffSky=0.04*afwGeom.arcseconds, maxDiffPix=0.001))
#pybind11#
#pybind11#        # check that doShortCircuit works in the private implementation
#pybind11#        errStr1 = _compareWcsOverBBox(wcs0, wcs1, bbox,
#pybind11#                                      maxDiffSky=0.001*afwGeom.arcseconds, maxDiffPix=0.001, doShortCircuit=False)
#pybind11#        errStr2 = _compareWcsOverBBox(wcs0, wcs1, bbox,
#pybind11#                                      maxDiffSky=0.001*afwGeom.arcseconds, maxDiffPix=0.001, doShortCircuit=True)
#pybind11#        self.assertNotEqual(errStr1, errStr2)
#pybind11#
#pybind11#    def checkMaskedImage(self, mi):
#pybind11#        """Run assertImage-like function tests on a masked image
#pybind11#
#pybind11#        Compare the masked image to itself, then alter copies and check that the altered copy
#pybind11#        is or is not nearly equal the original, depending on the amount of change, rtol and atol
#pybind11#        """
#pybind11#        epsilon = 1e-5  # margin to avoid roundoff error
#pybind11#
#pybind11#        mi0 = mi.Factory(mi, True)  # deep copy
#pybind11#        mi1 = mi.Factory(mi, True)
#pybind11#
#pybind11#        # a masked image should be exactly equal to itself
#pybind11#        self.assertMaskedImagesEqual(mi0, mi1)
#pybind11#        self.assertMaskedImagesEqual(mi1, mi0)
#pybind11#        self.assertMaskedImagesNearlyEqual(mi0, mi1, atol=0, rtol=0)
#pybind11#        self.assertMaskedImagesNearlyEqual(mi1, mi0, atol=0, rtol=0)
#pybind11#        self.assertMaskedImagesNearlyEqual(mi0.getArrays(), mi1, atol=0, rtol=0)
#pybind11#        self.assertMaskedImagesNearlyEqual(mi0, mi1.getArrays(), atol=0, rtol=0)
#pybind11#        self.assertMaskedImagesNearlyEqual(mi0.getArrays(), mi1.getArrays(), atol=0, rtol=0)
#pybind11#        for getName in ("getImage", "getVariance"):
#pybind11#            plane0 = getattr(mi0, getName)()
#pybind11#            plane1 = getattr(mi1, getName)()
#pybind11#            self.assertImagesEqual(plane0, plane1)
#pybind11#            self.assertImagesEqual(plane1, plane0)
#pybind11#            self.assertImagesNearlyEqual(plane0, plane1, atol=0, rtol=0)
#pybind11#            self.assertImagesNearlyEqual(plane1, plane0, atol=0, rtol=0)
#pybind11#            self.assertImagesNearlyEqual(plane0.getArray(), plane1, atol=0, rtol=0)
#pybind11#            self.assertImagesNearlyEqual(plane0, plane1.getArray(), atol=0, rtol=0)
#pybind11#            self.assertImagesNearlyEqual(plane0.getArray(), plane1.getArray(), atol=0, rtol=0)
#pybind11#            self.assertMasksEqual(plane0, plane1)
#pybind11#            self.assertMasksEqual(plane1, plane0)
#pybind11#            self.assertMasksEqual(plane0.getArray(), plane1)
#pybind11#            self.assertMasksEqual(plane0, plane1.getArray())
#pybind11#            self.assertMasksEqual(plane0.getArray(), plane1.getArray())
#pybind11#        self.assertMasksEqual(mi0.getMask(), mi1.getMask())
#pybind11#        self.assertMasksEqual(mi1.getMask(), mi0.getMask())
#pybind11#
#pybind11#        # alter image and variance planes and check the results
#pybind11#        for getName in ("getImage", "getVariance"):
#pybind11#            isFloat = getattr(mi, getName)().getArray().dtype.kind == "f"
#pybind11#            if isFloat:
#pybind11#                for errVal in (np.nan, np.inf, -np.inf):
#pybind11#                    mi0 = mi.Factory(mi, True)
#pybind11#                    mi1 = mi.Factory(mi, True)
#pybind11#                    plane0 = getattr(mi0, getName)()
#pybind11#                    plane1 = getattr(mi1, getName)()
#pybind11#                    plane1[2, 2] = errVal
#pybind11#                    with self.assertRaises(Exception):
#pybind11#                        self.assertImagesNearlyEqual(plane0, plane1)
#pybind11#                    with self.assertRaises(Exception):
#pybind11#                        self.assertImagesNearlyEqual(plane0.getArray(), plane1)
#pybind11#                    with self.assertRaises(Exception):
#pybind11#                        self.assertImagesNearlyEqual(plane1, plane0)
#pybind11#                    with self.assertRaises(Exception):
#pybind11#                        self.assertMaskedImagesNearlyEqual(mi0, mi1)
#pybind11#                    with self.assertRaises(Exception):
#pybind11#                        self.assertMaskedImagesNearlyEqual(mi0, mi1.getArrays())
#pybind11#                    with self.assertRaises(Exception):
#pybind11#                        self.assertMaskedImagesNearlyEqual(mi1, mi0)
#pybind11#
#pybind11#                    skipMask = mi.getMask().Factory(mi.getMask(), True)
#pybind11#                    skipMaskArr = skipMask.getArray()
#pybind11#                    skipMaskArr[:] = 0
#pybind11#                    skipMaskArr[2, 2] = 1
#pybind11#                    self.assertImagesNearlyEqual(plane0, plane1, skipMask=skipMaskArr, atol=0, rtol=0)
#pybind11#                    self.assertImagesNearlyEqual(plane0, plane1, skipMask=skipMask, atol=0, rtol=0)
#pybind11#                    self.assertMaskedImagesNearlyEqual(mi0, mi1, skipMask=skipMaskArr, atol=0, rtol=0)
#pybind11#                    self.assertMaskedImagesNearlyEqual(mi0, mi1, skipMask=skipMask, atol=0, rtol=0)
#pybind11#
#pybind11#                for dval in (0.001, 0.03):
#pybind11#                    mi0 = mi.Factory(mi, True)
#pybind11#                    mi1 = mi.Factory(mi, True)
#pybind11#                    plane0 = getattr(mi0, getName)()
#pybind11#                    plane1 = getattr(mi1, getName)()
#pybind11#                    plane1[2, 2] += dval
#pybind11#                    val1 = plane1.get(2, 2)
#pybind11#                    self.assertImagesNearlyEqual(plane0, plane1, rtol=0, atol=dval + epsilon)
#pybind11#                    self.assertImagesNearlyEqual(plane0, plane1, rtol=dval/val1 + epsilon, atol=0)
#pybind11#                    self.assertMaskedImagesNearlyEqual(mi0, mi1, rtol=0, atol=dval + epsilon)
#pybind11#                    self.assertMaskedImagesNearlyEqual(mi1, mi0, rtol=0, atol=dval + epsilon)
#pybind11#                    with self.assertRaises(Exception):
#pybind11#                        self.assertImagesNearlyEqual(plane0, plane1, rtol=0, atol=dval - epsilon)
#pybind11#                    with self.assertRaises(Exception):
#pybind11#                        self.assertImagesNearlyEqual(plane0, plane1, rtol=dval/val1 - epsilon, atol=0)
#pybind11#                    with self.assertRaises(Exception):
#pybind11#                        self.assertMaskedImagesNearlyEqual(mi0, mi1, rtol=0, atol=dval - epsilon)
#pybind11#                    with self.assertRaises(Exception):
#pybind11#                        self.assertMaskedImagesNearlyEqual(mi0, mi1, rtol=dval/val1 - epsilon, atol=0)
#pybind11#            else:
#pybind11#                # plane is an integer of some type
#pybind11#                for dval in (1, 3):
#pybind11#                    mi0 = mi.Factory(mi, True)
#pybind11#                    mi1 = mi.Factory(mi, True)
#pybind11#                    plane0 = getattr(mi0, getName)()
#pybind11#                    plane1 = getattr(mi1, getName)()
#pybind11#                    plane1[2, 2] += dval
#pybind11#                    val1 = plane1.get(2, 2)
#pybind11#                    # int value and test is <= so epsilon not required for atol
#pybind11#                    # but rtol is a fraction, so epsilon is still safest for the rtol test
#pybind11#                    self.assertImagesNearlyEqual(plane0, plane1, rtol=0, atol=dval)
#pybind11#                    self.assertImagesNearlyEqual(plane0, plane1, rtol=dval/val1 + epsilon, atol=0)
#pybind11#                    with self.assertRaises(Exception):
#pybind11#                        self.assertImagesNearlyEqual(plane0, plane1, rtol=0, atol=dval - epsilon)
#pybind11#                    with self.assertRaises(Exception):
#pybind11#                        self.assertImagesNearlyEqual(plane0, plane1, rtol=dval/val1 - epsilon, atol=0)
#pybind11#
#pybind11#        # alter mask and check the results
#pybind11#        mi0 = mi.Factory(mi, True)
#pybind11#        mi1 = mi.Factory(mi, True)
#pybind11#        mask0 = mi0.getMask()
#pybind11#        mask1 = mi1.getMask()
#pybind11#        for dval in (1, 3):
#pybind11#            mask1.getArray()[2, 2] += 1  # getArray avoids "unsupported operand type" failure
#pybind11#            with self.assertRaises(Exception):
#pybind11#                self.assertMasksEqual(mask0, mask1)
#pybind11#            with self.assertRaises(Exception):
#pybind11#                self.assertMasksEqual(mask1, mask0)
#pybind11#            with self.assertRaises(Exception):
#pybind11#                self.assertMaskedImagesEqual(mi0, mi1)
#pybind11#            with self.assertRaises(Exception):
#pybind11#                self.assertMaskedImagesEqual(mi1, mi0)
#pybind11#
#pybind11#        skipMask = mi.getMask().Factory(mi.getMask(), True)
#pybind11#        skipMaskArr = skipMask.getArray()
#pybind11#        skipMaskArr[:] = 0
#pybind11#        skipMaskArr[2, 2] = 1
#pybind11#        self.assertMasksEqual(mask0, mask1, skipMask=skipMaskArr)
#pybind11#        self.assertMasksEqual(mask0, mask1, skipMask=skipMask)
#pybind11#        self.assertMaskedImagesNearlyEqual(mi0, mi1, skipMask=skipMaskArr, atol=0, rtol=0)
#pybind11#        self.assertMaskedImagesNearlyEqual(mi0, mi1, skipMask=skipMask, atol=0, rtol=0)
#pybind11#
#pybind11#    def testAssertImagesNearlyEqual(self):
#pybind11#        """Test assertImagesNearlyEqual, assertMasksNearlyEqual and assertMaskedImagesNearlyEqual
#pybind11#        """
#pybind11#        width = 10
#pybind11#        height = 9
#pybind11#
#pybind11#        for miType in (afwImage.MaskedImageF, afwImage.MaskedImageD, afwImage.MaskedImageI,
#pybind11#                       afwImage.MaskedImageU):
#pybind11#            mi = makeRampMaskedImageWithNans(width, height, miType)
#pybind11#            self.checkMaskedImage(mi)
#pybind11#
#pybind11#        for invalidType in (np.zeros([width+1, height]), str, self.assertRaises):
#pybind11#            mi = makeRampMaskedImageWithNans(width, height, miType)
#pybind11#            with self.assertRaises(TypeError):
#pybind11#                self.assertMasksEqual(mi.getMask(), invalidType)
#pybind11#            with self.assertRaises(TypeError):
#pybind11#                self.assertMasksEqual(invalidType, mi.getMask())
#pybind11#            with self.assertRaises(TypeError):
#pybind11#                self.assertMasksEqual(mi.getMask(), mi.getMask(), skipMask=invalidType)
#pybind11#
#pybind11#            with self.assertRaises(TypeError):
#pybind11#                self.assertImagesNearlyEqual(mi.getImage(), invalidType)
#pybind11#            with self.assertRaises(TypeError):
#pybind11#                self.assertImagesNearlyEqual(invalidType, mi.getImage())
#pybind11#            with self.assertRaises(TypeError):
#pybind11#                self.assertImagesNearlyEqual(mi.getImage(), mi.getImage(), skipMask=invalidType)
#pybind11#
#pybind11#            with self.assertRaises(TypeError):
#pybind11#                self.assertMaskedImagesNearlyEqual(mi, invalidType)
#pybind11#            with self.assertRaises(TypeError):
#pybind11#                self.assertMaskedImagesNearlyEqual(invalidType, mi)
#pybind11#            with self.assertRaises(TypeError):
#pybind11#                self.assertMaskedImagesNearlyEqual(mi, mi, skipMask=invalidType)
#pybind11#
#pybind11#            with self.assertRaises(TypeError):
#pybind11#                self.assertMaskedImagesNearlyEqual(mi.getImage(), mi.getImage())
#pybind11#
#pybind11#    def testUnsignedImages(self):
#pybind11#        """Unsigned images can give incorrect differences unless the test code is careful
#pybind11#        """
#pybind11#        image0 = np.zeros([5, 5], dtype=np.uint8)
#pybind11#        image1 = np.zeros([5, 5], dtype=np.uint8)
#pybind11#        image0[0, 0] = 1
#pybind11#        image1[0, 1] = 2
#pybind11#
#pybind11#        # arrays differ by a maximum of 2
#pybind11#        errMsg1 = imagesDiffer(image0, image1)
#pybind11#        match = re.match(r"maxDiff *= *(\d+)", errMsg1, re.IGNORECASE)
#pybind11#        self.assertIsNotNone(match)
#pybind11#        self.assertEqual(match.group(1), "2")
#pybind11#
#pybind11#        # arrays are equal to within 5
#pybind11#        self.assertImagesNearlyEqual(image0, image1, atol=5)
#pybind11#
#pybind11#
#pybind11#def makeRampMaskedImageWithNans(width, height, imgClass=afwImage.MaskedImageF):
#pybind11#    """Make a masked image that is a ramp with additional non-finite values
#pybind11#
#pybind11#    Make a masked image with the following additional non-finite values
#pybind11#    in the variance plane and (if image is of some floating type) image plane:
#pybind11#    - nan at [0, 0]
#pybind11#    - inf at [1, 0]
#pybind11#    - -inf at [0, 1]
#pybind11#    """
#pybind11#    mi = makeRampMaskedImage(width, height, imgClass)
#pybind11#
#pybind11#    var = mi.getVariance()
#pybind11#    var[0, 0] = np.nan
#pybind11#    var[1, 0] = np.inf
#pybind11#    var[0, 1] = -np.inf
#pybind11#
#pybind11#    im = mi.getImage()
#pybind11#    try:
#pybind11#        np.array([np.nan], dtype=im.getArray().dtype)
#pybind11#    except Exception:
#pybind11#        # image plane does not support nan, etc. (presumably an int of some variety)
#pybind11#        pass
#pybind11#    else:
#pybind11#        # image plane does support nan, etc.
#pybind11#        im[0, 0] = np.nan
#pybind11#        im[1, 0] = np.inf
#pybind11#        im[0, 1] = -np.inf
#pybind11#    return mi
#pybind11#
#pybind11#
#pybind11#def makeRampMaskedImage(width, height, imgClass=afwImage.MaskedImageF):
#pybind11#    """Make a ramp image of the specified size and image class
#pybind11#
#pybind11#    Image values start from 0 at the lower left corner and increase by 1 along rows
#pybind11#    Variance values equal image values + 100
#pybind11#    Mask values equal image values modulo 8 bits (leaving plenty of unused values)
#pybind11#    """
#pybind11#    mi = imgClass(width, height)
#pybind11#    image = mi.getImage()
#pybind11#    mask = mi.getMask()
#pybind11#    variance = mi.getVariance()
#pybind11#    val = 0
#pybind11#    for yInd in range(height):
#pybind11#        for xInd in range(width):
#pybind11#            image.set(xInd, yInd, val)
#pybind11#            variance.set(xInd, yInd, val + 100)
#pybind11#            mask.set(xInd, yInd, val % 0x100)
#pybind11#            val += 1
#pybind11#    return mi
#pybind11#
#pybind11#
#pybind11#class MemoryTester(lsst.utils.tests.MemoryTestCase):
#pybind11#    pass
#pybind11#
#pybind11#
#pybind11#def setup_module(module):
#pybind11#    lsst.utils.tests.init()
#pybind11#
#pybind11#
#pybind11#if __name__ == "__main__":
#pybind11#    lsst.utils.tests.init()
#pybind11#    unittest.main()
