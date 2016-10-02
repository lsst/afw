#pybind11##!/usr/bin/env python
#pybind11#from __future__ import absolute_import, division, print_function
#pybind11#from builtins import zip
#pybind11#from builtins import range
#pybind11#from functools import reduce
#pybind11#
#pybind11##
#pybind11## LSST Data Management System
#pybind11## Copyright 2008-2015 AURA/LSST.
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
#pybind11## see <https://www.lsstcorp.org/LegalNotices/>.
#pybind11##
#pybind11#
#pybind11#import math
#pybind11#import os.path
#pybind11#import unittest
#pybind11#import numpy as np
#pybind11#import pickle
#pybind11#
#pybind11#import lsst.utils
#pybind11#import lsst.utils.tests
#pybind11#import lsst.pex.exceptions
#pybind11#from lsst.daf.base import PropertySet
#pybind11#import lsst.afw.image.imageLib as afwImage
#pybind11#import lsst.afw.math as afwMath
#pybind11#import lsst.afw.geom as afwGeom
#pybind11#import lsst.afw.display.ds9 as ds9
#pybind11#import lsst.pex.exceptions as pexExcept
#pybind11#
#pybind11## Set to True to display debug messages and images in ds9.
#pybind11#debugMode = False
#pybind11#
#pybind11#try:
#pybind11#    AfwdataDir = lsst.utils.getPackageDir("afwdata")
#pybind11#except pexExcept.NotFoundError:
#pybind11#    AfwdataDir = None
#pybind11#
#pybind11#
#pybind11#class BackgroundTestCase(lsst.utils.tests.TestCase):
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        np.random.seed(1)
#pybind11#        self.val = 10
#pybind11#        self.image = afwImage.ImageF(afwGeom.Box2I(afwGeom.Point2I(1000, 500), afwGeom.Extent2I(100, 200)))
#pybind11#        self.image.set(self.val)
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        del self.image
#pybind11#
#pybind11#    def testOddSize(self):
#pybind11#        """Test for ticket #1781 -- without it, in oddly-sized images
#pybind11#        there is a chunk of pixels on the right/bottom that do not go
#pybind11#        into the fit and are extrapolated.  After this ticket, the
#pybind11#        subimage boundaries are spread more evenly so the last pixels
#pybind11#        get fit as well.  This slightly strange test case checks that
#pybind11#        the interpolant is close to the function at the end.  I could
#pybind11#        not think of an interpolant that would fit exactly, so this
#pybind11#        just puts a limit on the errors.
#pybind11#        """
#pybind11#        W, H = 2, 99
#pybind11#        image = afwImage.ImageF(afwGeom.Extent2I(W, H))
#pybind11#        bgCtrl = afwMath.BackgroundControl(afwMath.Interpolate.LINEAR)
#pybind11#        bgCtrl.setNxSample(2)
#pybind11#        NY = 10
#pybind11#        bgCtrl.setNySample(NY)
#pybind11#        for y in range(H):
#pybind11#            for x in range(W):
#pybind11#                B = 89
#pybind11#                if y < B:
#pybind11#                    image.set(x, y, y)
#pybind11#                else:
#pybind11#                    image.set(x, y, B + (y-B)*-1.)
#pybind11#        bobj = afwMath.makeBackground(image, bgCtrl)
#pybind11#        back = bobj.getImageF()
#pybind11#
#pybind11#        for iy, by in zip([image.get(0, y) for y in range(H)],
#pybind11#                          [back.get(0, y) for y in range(H)]):
#pybind11#            self.assertLess(abs(iy - by), 5)
#pybind11#
#pybind11#    def testgetPixel(self):
#pybind11#        """Tests basic functionality of getPixel() method (floats)"""
#pybind11#        xcen, ycen = 50, 100
#pybind11#        bgCtrl = afwMath.BackgroundControl(10, 10)
#pybind11#        bgCtrl.setNxSample(5)
#pybind11#        bgCtrl.setNySample(5)
#pybind11#        bgCtrl.getStatisticsControl().setNumIter(3)
#pybind11#        bgCtrl.getStatisticsControl().setNumSigmaClip(3)
#pybind11#        back = afwMath.makeBackground(self.image, bgCtrl)
#pybind11#
#pybind11#        self.assertEqual(afwMath.cast_BackgroundMI(back).getPixel(xcen, ycen), self.val)
#pybind11#
#pybind11#    @unittest.skipIf(AfwdataDir is None, "afwdata not setup")
#pybind11#    def testBackgroundTestImages(self):
#pybind11#        """Tests Laher's afwdata/Statistics/*.fits images (doubles)"""
#pybind11#        imginfolist = []
#pybind11#        imginfolist.append(["v1_i1_g_m400_s20_f.fits", 399.9912966583894])  # cooked to known value
#pybind11#
#pybind11#        for imginfo in imginfolist:
#pybind11#            imgfile, centerValue = imginfo
#pybind11#            imgPath = os.path.join(AfwdataDir, "Statistics", imgfile)
#pybind11#
#pybind11#            # get the image and header
#pybind11#            dimg = afwImage.DecoratedImageF(imgPath)
#pybind11#            img = dimg.getImage()
#pybind11#            fitsHdr = dimg.getMetadata()  # the FITS header
#pybind11#
#pybind11#            # get the True values of the mean and stdev
#pybind11#            reqMean = fitsHdr.getAsDouble("MEANREQ")
#pybind11#            reqStdev = fitsHdr.getAsDouble("SIGREQ")
#pybind11#            naxis1 = img.getWidth()
#pybind11#            naxis2 = img.getHeight()
#pybind11#
#pybind11#            # create a background control object
#pybind11#            bctrl = afwMath.BackgroundControl(afwMath.Interpolate.AKIMA_SPLINE)
#pybind11#            bctrl.setNxSample(5)
#pybind11#            bctrl.setNySample(5)
#pybind11#
#pybind11#            # run the background constructor and call the getPixel() and getImage() functions.
#pybind11#            backobj = afwMath.makeBackground(img, bctrl)
#pybind11#
#pybind11#            pixPerSubimage = img.getWidth()*img.getHeight()/(bctrl.getNxSample()*bctrl.getNySample())
#pybind11#            stdevInterp = reqStdev/math.sqrt(pixPerSubimage)
#pybind11#
#pybind11#            # test getPixel()
#pybind11#            testval = afwMath.cast_BackgroundMI(backobj).getPixel(naxis1//2, naxis2//2)
#pybind11#            self.assertAlmostEqual(testval/centerValue, 1, places=7)
#pybind11#            self.assertLess(abs(testval - reqMean), 2*stdevInterp)
#pybind11#
#pybind11#            # test getImage() by checking the center pixel
#pybind11#            bimg = backobj.getImageF()
#pybind11#            testImgval = bimg.get(naxis1//2, naxis2//2)
#pybind11#            self.assertLess(abs(testImgval - reqMean), 2*stdevInterp)
#pybind11#
#pybind11#    def testRamp(self):
#pybind11#        """tests Laher's afwdata/Statistics/*.fits images (doubles)"""
#pybind11#        # make a ramping image (spline should be exact for linear increasing image
#pybind11#        nx = 512
#pybind11#        ny = 512
#pybind11#        x0, y0 = 9876, 54321
#pybind11#        box = afwGeom.Box2I(afwGeom.Point2I(x0, y0), afwGeom.Extent2I(nx, ny))
#pybind11#        rampimg = afwImage.ImageF(box)
#pybind11#        dzdx, dzdy, z0 = 0.1, 0.2, 10000.0
#pybind11#        for x in range(nx):
#pybind11#            for y in range(ny):
#pybind11#                rampimg.set(x, y, dzdx*x + dzdy*y + z0)
#pybind11#
#pybind11#        # check corner, edge, and center pixels
#pybind11#        bctrl = afwMath.BackgroundControl(10, 10)
#pybind11#        bctrl.setInterpStyle(afwMath.Interpolate.CUBIC_SPLINE)
#pybind11#        bctrl.setNxSample(6)
#pybind11#        bctrl.setNySample(6)
#pybind11#        bctrl.getStatisticsControl().setNumSigmaClip(20.0) # something large enough to avoid clipping entirely
#pybind11#        bctrl.getStatisticsControl().setNumIter(1)
#pybind11#        backobj = afwMath.cast_BackgroundMI(afwMath.makeBackground(rampimg, bctrl))
#pybind11#
#pybind11#        if debugMode:
#pybind11#            print(rampimg.getArray())
#pybind11#
#pybind11#        frame = 1
#pybind11#        for interp in ("CONSTANT", "LINEAR", "NATURAL_SPLINE", "AKIMA_SPLINE"):
#pybind11#            diff = backobj.getImageF(interp)
#pybind11#            if debugMode:
#pybind11#                ds9.mtv(diff, frame=frame)
#pybind11#                frame += 1
#pybind11#            diff -= rampimg
#pybind11#            if debugMode:
#pybind11#                print(interp, diff.getArray().mean(), diff.getArray().std())
#pybind11#            if debugMode:
#pybind11#                ds9.mtv(diff, frame=frame)
#pybind11#                frame += 1
#pybind11#        if debugMode:
#pybind11#            ds9.mtv(rampimg, frame=frame)
#pybind11#            frame += 1
#pybind11#            ds9.mtv(backobj.getStatsImage(), frame=frame)
#pybind11#            frame += 1
#pybind11#
#pybind11#        xpixels = [0, nx//2, nx - 1]
#pybind11#        ypixels = [0, ny//2, ny - 1]
#pybind11#        for xpix in xpixels:
#pybind11#            for ypix in ypixels:
#pybind11#                testval = backobj.getPixel(xpix, ypix)
#pybind11#                self.assertAlmostEqual(testval/rampimg.get(xpix, ypix), 1, 6)
#pybind11#
#pybind11#        # Test pickle
#pybind11#        new = pickle.loads(pickle.dumps(backobj))
#pybind11#        self.assertBackgroundEqual(backobj, new)
#pybind11#
#pybind11#        # Check creation of sub-image
#pybind11#        box = afwGeom.Box2I(afwGeom.Point2I(123, 45), afwGeom.Extent2I(45, 123))
#pybind11#        box.shift(afwGeom.Extent2I(x0, y0))
#pybind11#        bgImage = backobj.getImageF("AKIMA_SPLINE")
#pybind11#        bgSubImage = afwImage.ImageF(bgImage, box)
#pybind11#        testImage = backobj.getImageF(box, "AKIMA_SPLINE")
#pybind11#        self.assertEqual(testImage.getXY0(), bgSubImage.getXY0())
#pybind11#        self.assertEqual(testImage.getDimensions(), bgSubImage.getDimensions())
#pybind11#        self.assertImagesEqual(testImage, bgSubImage)
#pybind11#
#pybind11#    def getParabolaImage(self, nx, ny, pars=(1.0e-4, 1.0e-4, 0.1, 0.2, 10.0)):
#pybind11#        """Make sure a quadratic map is *well* reproduced by the spline model"""
#pybind11#        parabimg = afwImage.ImageF(afwGeom.Extent2I(nx, ny))
#pybind11#        d2zdx2, d2zdy2, dzdx, dzdy, z0 = pars  # no cross-terms
#pybind11#        for x in range(nx):
#pybind11#            for y in range(ny):
#pybind11#                parabimg.set(x, y, d2zdx2*x*x + d2zdy2*y*y + dzdx*x + dzdy*y + z0)
#pybind11#        return parabimg
#pybind11#
#pybind11#    @unittest.skipIf(AfwdataDir is None, "afwdata not setup")
#pybind11#    def testTicket987(self):
#pybind11#        """This code used to abort; so the test is that it doesn't"""
#pybind11#        imagePath = os.path.join(AfwdataDir, "DC3a-Sim", "sci", "v5-e0", "v5-e0-c011-a00.sci.fits")
#pybind11#        mimg = afwImage.MaskedImageF(imagePath)
#pybind11#        binsize = 512
#pybind11#        bctrl = afwMath.BackgroundControl("NATURAL_SPLINE")
#pybind11#
#pybind11#        # note: by default undersampleStyle is THROW_EXCEPTION
#pybind11#        bctrl.setUndersampleStyle(afwMath.REDUCE_INTERP_ORDER)
#pybind11#
#pybind11#        nx = int(mimg.getWidth()/binsize) + 1
#pybind11#        ny = int(mimg.getHeight()/binsize) + 1
#pybind11#
#pybind11#        bctrl.setNxSample(nx)
#pybind11#        bctrl.setNySample(ny)
#pybind11#        image = mimg.getImage()
#pybind11#        backobj = afwMath.makeBackground(image, bctrl)
#pybind11#        image -= backobj.getImageF()
#pybind11#
#pybind11#    def testTicket1781(self):
#pybind11#        """Test an unusual-sized image"""
#pybind11#        nx = 526
#pybind11#        ny = 154
#pybind11#
#pybind11#        parabimg = self.getParabolaImage(nx, ny)
#pybind11#
#pybind11#        bctrl = afwMath.BackgroundControl(afwMath.Interpolate.CUBIC_SPLINE)
#pybind11#        bctrl.setNxSample(16)
#pybind11#        bctrl.setNySample(4)
#pybind11#        bctrl.getStatisticsControl().setNumSigmaClip(10.0)
#pybind11#        bctrl.getStatisticsControl().setNumIter(1)
#pybind11#        backobj = afwMath.makeBackground(parabimg, bctrl)
#pybind11#
#pybind11#    def testParabola(self):
#pybind11#        """Test an image which varies parabolicly (spline should be exact for 2rd order polynomial)"""
#pybind11#        nx = 512
#pybind11#        ny = 512
#pybind11#
#pybind11#        parabimg = self.getParabolaImage(nx, ny)
#pybind11#
#pybind11#        # check corner, edge, and center pixels
#pybind11#        bctrl = afwMath.BackgroundControl(afwMath.Interpolate.CUBIC_SPLINE)
#pybind11#        bctrl.setNxSample(24)
#pybind11#        bctrl.setNySample(24)
#pybind11#        bctrl.getStatisticsControl().setNumSigmaClip(10.0)
#pybind11#        bctrl.getStatisticsControl().setNumIter(1)
#pybind11#        backobj = afwMath.makeBackground(parabimg, bctrl)
#pybind11#
#pybind11#        segmentCenter = int(0.5*nx/bctrl.getNxSample())
#pybind11#        xpixels = [segmentCenter, nx//2, nx - segmentCenter]
#pybind11#        ypixels = [segmentCenter, ny//2, ny - segmentCenter]
#pybind11#        for xpix in xpixels:
#pybind11#            for ypix in ypixels:
#pybind11#                testval = afwMath.cast_BackgroundMI(backobj).getPixel(bctrl.getInterpStyle(), xpix, ypix)
#pybind11#                realval = parabimg.get(xpix, ypix)
#pybind11#                # quadratic terms skew the averages of the subimages and the clipped mean for
#pybind11#                # a subimage != value of center pixel.  1/20 counts on a 10000 count sky
#pybind11#                # is a fair (if arbitrary) test.
#pybind11#                self.assertLess(abs(testval - realval), 0.5)
#pybind11#
#pybind11#    @unittest.skipIf(AfwdataDir is None, "afwdata not setup")
#pybind11#    def testCFHT_oldAPI(self):
#pybind11#        """Test background subtraction on some real CFHT data"""
#pybind11#        mi = afwImage.MaskedImageF(os.path.join(AfwdataDir,
#pybind11#                                                "CFHT", "D4", "cal-53535-i-797722_1.fits"))
#pybind11#        mi = mi.Factory(mi, afwGeom.Box2I(afwGeom.Point2I(32, 2),
#pybind11#                                          afwGeom.Point2I(2079, 4609)), afwImage.LOCAL)
#pybind11#
#pybind11#        bctrl = afwMath.BackgroundControl(afwMath.Interpolate.AKIMA_SPLINE)
#pybind11#        bctrl.setNxSample(16)
#pybind11#        bctrl.setNySample(16)
#pybind11#        bctrl.getStatisticsControl().setNumSigmaClip(3.0)
#pybind11#        bctrl.getStatisticsControl().setNumIter(2)
#pybind11#        backobj = afwMath.makeBackground(mi.getImage(), bctrl)
#pybind11#
#pybind11#        if debugMode:
#pybind11#            ds9.mtv(mi, frame=0)
#pybind11#
#pybind11#        im = mi.getImage()
#pybind11#        im -= backobj.getImageF()
#pybind11#
#pybind11#        if debugMode:
#pybind11#            ds9.mtv(mi, frame=1)
#pybind11#
#pybind11#    def getCfhtImage(self):
#pybind11#        """Get a portion of a CFHT image as a MaskedImageF"""
#pybind11#        bbox = afwGeom.Box2I(afwGeom.Point2I(500, 2000), afwGeom.Point2I(2079, 4609))
#pybind11#        imagePath = os.path.join(AfwdataDir, "CFHT", "D4", "cal-53535-i-797722_1.fits")
#pybind11#        return afwImage.MaskedImageF(imagePath, PropertySet(), bbox)
#pybind11#
#pybind11#    @unittest.skipIf(AfwdataDir is None, "afwdata not setup")
#pybind11#    def testXY0(self):
#pybind11#        """Test fitting the background to an image with nonzero xy0
#pybind11#
#pybind11#        The statsImage and background image should not vary with xy0
#pybind11#        """
#pybind11#        bgImageList = []  # list of background images, one per xy0
#pybind11#        statsImageList = []  # list of stats images, one per xy0
#pybind11#        for xy0 in (afwGeom.Point2I(0, 0), afwGeom.Point2I(-100, -999), afwGeom.Point2I(1000, 500)):
#pybind11#            mi = self.getCfhtImage()
#pybind11#            mi.setXY0(xy0)
#pybind11#
#pybind11#            bctrl = afwMath.BackgroundControl(mi.getWidth()//128, mi.getHeight()//128)
#pybind11#            backobj = afwMath.makeBackground(mi.getImage(), bctrl)
#pybind11#            bgImage = backobj.getImageF()
#pybind11#            self.assertEqual(bgImage.getBBox(), mi.getBBox())
#pybind11#            bgImageList.append(bgImage)
#pybind11#
#pybind11#            statsImage = afwMath.cast_BackgroundMI(backobj).getStatsImage()
#pybind11#            statsImageList.append(statsImage)
#pybind11#
#pybind11#        # changing the bounding box should make no difference to the pixel values,
#pybind11#        # so compare pixels using exact equality
#pybind11#        for bgImage in bgImageList[1:]:
#pybind11#            self.assertImagesEqual(bgImage, bgImageList[0])
#pybind11#        for statsImage in statsImageList[1:]:
#pybind11#            self.assertMaskedImagesEqual(statsImage, statsImageList[0])
#pybind11#
#pybind11#    @unittest.skipIf(AfwdataDir is None, "afwdata not setup")
#pybind11#    def testSubImage(self):
#pybind11#        """Test getImage on a subregion of the full background image
#pybind11#
#pybind11#        Using real image data is a cheap way to get a variable background
#pybind11#        """
#pybind11#        mi = self.getCfhtImage()
#pybind11#
#pybind11#        bctrl = afwMath.BackgroundControl(mi.getWidth()//128, mi.getHeight()//128)
#pybind11#        backobj = afwMath.makeBackground(mi.getImage(), bctrl)
#pybind11#        subBBox = afwGeom.Box2I(afwGeom.Point2I(1000, 3000), afwGeom.Extent2I(100, 100))
#pybind11#
#pybind11#        bgFullImage = backobj.getImageF()
#pybind11#        self.assertEqual(bgFullImage.getBBox(), mi.getBBox())
#pybind11#
#pybind11#        subFullArr = afwImage.ImageF(bgFullImage, subBBox).getArray()
#pybind11#
#pybind11#        bgSubImage = backobj.getImageF(subBBox, bctrl.getInterpStyle())
#pybind11#        subArr = bgSubImage.getArray()
#pybind11#
#pybind11#        # the pixels happen to be identical but it is safer not to rely on that; close is good enough
#pybind11#        self.assertFloatsEqual(subArr, subFullArr)
#pybind11#
#pybind11#    @unittest.skipIf(AfwdataDir is None, "afwdata not setup")
#pybind11#    def testCFHT(self):
#pybind11#        """Test background subtraction on some real CFHT data"""
#pybind11#        mi = self.getCfhtImage()
#pybind11#
#pybind11#        bctrl = afwMath.BackgroundControl(mi.getWidth()//128, mi.getHeight()//128)
#pybind11#        bctrl.getStatisticsControl().setNumSigmaClip(3.0)
#pybind11#        bctrl.getStatisticsControl().setNumIter(2)
#pybind11#        backobj = afwMath.makeBackground(mi.getImage(), bctrl)
#pybind11#
#pybind11#        if debugMode:
#pybind11#            ds9.mtv(mi, frame=0)
#pybind11#
#pybind11#        im = mi.getImage()
#pybind11#        im -= backobj.getImageF("AKIMA_SPLINE")
#pybind11#
#pybind11#        if debugMode:
#pybind11#            ds9.mtv(mi, frame=1)
#pybind11#
#pybind11#        statsImage = afwMath.cast_BackgroundMI(backobj).getStatsImage()
#pybind11#
#pybind11#        if debugMode:
#pybind11#            ds9.mtv(statsImage, frame=2)
#pybind11#            ds9.mtv(statsImage.getVariance(), frame=3)
#pybind11#
#pybind11#    def testUndersample(self):
#pybind11#        """Test how the program handles nx,ny being too small for requested interp style."""
#pybind11#        nx = 64
#pybind11#        ny = 64
#pybind11#        img = afwImage.ImageF(afwGeom.Extent2I(nx, ny))
#pybind11#
#pybind11#        # make a background control object
#pybind11#        bctrl = afwMath.BackgroundControl(10, 10)
#pybind11#        bctrl.setInterpStyle(afwMath.Interpolate.CUBIC_SPLINE)
#pybind11#        bctrl.setNxSample(3)
#pybind11#        bctrl.setNySample(3)
#pybind11#
#pybind11#        # put nx,ny back to 2 and see if it adjusts the interp style down to linear
#pybind11#        bctrl.setNxSample(2)
#pybind11#        bctrl.setNySample(2)
#pybind11#        bctrl.setUndersampleStyle("REDUCE_INTERP_ORDER")
#pybind11#        backobj = afwMath.makeBackground(img, bctrl)
#pybind11#        backobj.getImageF()             # Need to interpolate background to discover what we actually needed
#pybind11#        self.assertEqual(backobj.getAsUsedInterpStyle(), afwMath.Interpolate.LINEAR)
#pybind11#
#pybind11#        # put interp style back up to cspline and see if it throws an exception
#pybind11#        bctrl.setUndersampleStyle("THROW_EXCEPTION")
#pybind11#
#pybind11#        def tst(img, bctrl):
#pybind11#            backobj = afwMath.makeBackground(img, bctrl)
#pybind11#            backobj.getImageF("CUBIC_SPLINE")  # only now do we see that we have too few points
#pybind11#        self.assertRaises(lsst.pex.exceptions.InvalidParameterError,
#pybind11#                          tst, img, bctrl)
#pybind11#
#pybind11#    def testOnlyOneGridCell(self):
#pybind11#        """Test how the program handles nxSample,nySample being 1x1."""
#pybind11#        # try a ramping image ... has an easy analytic solution
#pybind11#        nx = 64
#pybind11#        ny = 64
#pybind11#        img = afwImage.ImageF(afwGeom.Extent2I(nx, ny), 10)
#pybind11#
#pybind11#        dzdx, dzdy, z0 = 0.1, 0.2, 10000.0
#pybind11#        mean = z0 + dzdx*(nx - 1)/2 + dzdy*(ny - 1)/2  # the analytic solution
#pybind11#        for x in range(nx):
#pybind11#            for y in range(ny):
#pybind11#                img.set(x, y, dzdx*x + dzdy*y + z0)
#pybind11#
#pybind11#        # make a background control object
#pybind11#        bctrl = afwMath.BackgroundControl(10, 10)
#pybind11#        bctrl.setInterpStyle(afwMath.Interpolate.CONSTANT)
#pybind11#        bctrl.setNxSample(1)
#pybind11#        bctrl.setNySample(1)
#pybind11#        bctrl.setUndersampleStyle(afwMath.THROW_EXCEPTION)
#pybind11#        backobj = afwMath.makeBackground(img, bctrl)
#pybind11#
#pybind11#        xpixels = [0, nx//2, nx - 1]
#pybind11#        ypixels = [0, ny//2, ny - 1]
#pybind11#        for xpix in xpixels:
#pybind11#            for ypix in ypixels:
#pybind11#                testval = afwMath.cast_BackgroundMI(backobj).getPixel(bctrl.getInterpStyle(), xpix, ypix)
#pybind11#                self.assertAlmostEqual(testval/mean, 1)
#pybind11#
#pybind11#    def testAdjustLevel(self):
#pybind11#        """Test that we can adjust a background level"""
#pybind11#        sky = 100
#pybind11#        im = afwImage.ImageF(40, 40)
#pybind11#        im.set(sky)
#pybind11#        nx, ny = im.getWidth()//2, im.getHeight()//2
#pybind11#        bctrl = afwMath.BackgroundControl("LINEAR", nx, ny)
#pybind11#        bkd = afwMath.makeBackground(im, bctrl)
#pybind11#
#pybind11#        self.assertEqual(afwMath.makeStatistics(bkd.getImageF(), afwMath.MEAN).getValue(), sky)
#pybind11#
#pybind11#        delta = 123
#pybind11#        bkd += delta
#pybind11#        self.assertEqual(afwMath.makeStatistics(bkd.getImageF(), afwMath.MEAN).getValue(), sky + delta)
#pybind11#        bkd -= delta
#pybind11#        self.assertEqual(afwMath.makeStatistics(bkd.getImageF(), afwMath.MEAN).getValue(), sky)
#pybind11#
#pybind11#    def testNaNFromMaskedImage(self):
#pybind11#        """Check that an extensively masked image doesn't lead to NaNs in the background estimation"""
#pybind11#        image = afwImage.MaskedImageF(800, 800)
#pybind11#        msk = image.getMask()
#pybind11#        bbox = afwGeom.BoxI(afwGeom.PointI(560, 0), afwGeom.PointI(799, 335))
#pybind11#        smsk = msk.Factory(msk, bbox)
#pybind11#        smsk.set(msk.getPlaneBitMask("DETECTED"))
#pybind11#
#pybind11#        binSize = 256
#pybind11#        nx = image.getWidth()//binSize + 1
#pybind11#        ny = image.getHeight()//binSize + 1
#pybind11#
#pybind11#        sctrl = afwMath.StatisticsControl()
#pybind11#        sctrl.setAndMask(reduce(lambda x, y: x | image.getMask().getPlaneBitMask(y),
#pybind11#                                ['EDGE', 'DETECTED', 'DETECTED_NEGATIVE'], 0x0))
#pybind11#
#pybind11#        bctrl = afwMath.BackgroundControl(nx, ny, sctrl, "MEANCLIP")
#pybind11#
#pybind11#        bkgd = afwMath.makeBackground(image, bctrl)
#pybind11#        bkgdImage = bkgd.getImageF("NATURAL_SPLINE", "THROW_EXCEPTION")
#pybind11#        if debugMode:
#pybind11#            ds9.mtv(image)
#pybind11#            ds9.mtv(bkgdImage, frame=1)
#pybind11#
#pybind11#        self.assertFalse(np.isnan(bkgdImage.get(0, 0)))
#pybind11#
#pybind11#        # Check that the non-string API works too
#pybind11#        bkgdImage = bkgd.getImageF(afwMath.Interpolate.NATURAL_SPLINE, afwMath.THROW_EXCEPTION)
#pybind11#
#pybind11#    def testBadAreaFailsSpline(self):
#pybind11#        """Check that a NaN in the stats image doesn't cause spline interpolation to fail (#2734)"""
#pybind11#        image = afwImage.ImageF(15, 9)
#pybind11#        for y in range(image.getHeight()):
#pybind11#            for x in range(image.getWidth()):
#pybind11#                image.set(x, y, 1 + 2*y)  # n.b. linear, which is what the interpolation will fall back to
#pybind11#
#pybind11#        # Set the right corner to NaN.  This will mean that we have too few points for a spline interpolator
#pybind11#        binSize = 3
#pybind11#        image[-binSize:, -binSize:] = np.nan
#pybind11#
#pybind11#        nx = image.getWidth()//binSize
#pybind11#        ny = image.getHeight()//binSize
#pybind11#
#pybind11#        sctrl = afwMath.StatisticsControl()
#pybind11#        bctrl = afwMath.BackgroundControl(nx, ny, sctrl, afwMath.MEANCLIP)
#pybind11#
#pybind11#        bkgd = afwMath.makeBackground(image, bctrl)
#pybind11#        if debugMode:
#pybind11#            ds9.mtv(image)
#pybind11#            ds9.mtv(afwMath.cast_BackgroundMI(bkgd).getStatsImage(), frame=1)
#pybind11#        # Should throw if we don't permit REDUCE_INTERP_ORDER
#pybind11#        self.assertRaises(lsst.pex.exceptions.OutOfRangeError,
#pybind11#                          bkgd.getImageF, afwMath.Interpolate.NATURAL_SPLINE)
#pybind11#        # The interpolation should fall back to linear for the right part of the image
#pybind11#        # where the NaNs don't permit spline interpolation (n.b. this happens to be exact)
#pybind11#        bkgdImage = bkgd.getImageF(afwMath.Interpolate.NATURAL_SPLINE, afwMath.REDUCE_INTERP_ORDER)
#pybind11#
#pybind11#        if debugMode:
#pybind11#            ds9.mtv(bkgdImage, frame=2)
#pybind11#
#pybind11#        image -= bkgdImage
#pybind11#        self.assertEqual(afwMath.makeStatistics(image, afwMath.MEAN).getValue(), 0.0)
#pybind11#
#pybind11#    def testBadPatch(self):
#pybind11#        """Test that a large bad patch of an image doesn't cause an absolute failure"""
#pybind11#        initialValue = 20
#pybind11#        mi = afwImage.MaskedImageF(500, 200)
#pybind11#        mi.set((initialValue, 0x0, 1.0))
#pybind11#        im = mi.getImage()
#pybind11#        im[0:200, :] = np.nan
#pybind11#        del im
#pybind11#        msk = mi.getMask()
#pybind11#        badBits = msk.getPlaneBitMask(['EDGE', 'DETECTED', 'DETECTED_NEGATIVE'])
#pybind11#        msk[0:400, :] |= badBits
#pybind11#        del msk
#pybind11#
#pybind11#        if debugMode:
#pybind11#            ds9.mtv(mi, frame=0)
#pybind11#
#pybind11#        sctrl = afwMath.StatisticsControl()
#pybind11#        sctrl.setAndMask(badBits)
#pybind11#        nx, ny = 17, 17
#pybind11#        bctrl = afwMath.BackgroundControl(nx, ny, sctrl, afwMath.MEANCLIP)
#pybind11#
#pybind11#        bkgd = afwMath.makeBackground(mi, bctrl)
#pybind11#        statsImage = afwMath.cast_BackgroundMI(bkgd).getStatsImage()
#pybind11#        if debugMode:
#pybind11#            ds9.mtv(statsImage, frame=1)
#pybind11#
#pybind11#        # the test is that this doesn't fail if the bug (#2297) is fixed
#pybind11#        bkgdImage = bkgd.getImageF(afwMath.Interpolate.NATURAL_SPLINE, afwMath.REDUCE_INTERP_ORDER)
#pybind11#        self.assertEqual(np.mean(bkgdImage[0:100, 0:100].getArray()), initialValue)
#pybind11#        if debugMode:
#pybind11#            ds9.mtv(bkgdImage, frame=2)
#pybind11#        # Check that we can fix the NaNs in the statsImage
#pybind11#        sim = statsImage.getImage().getArray()
#pybind11#        sim[np.isnan(sim)] = initialValue  # replace NaN by initialValue
#pybind11#        bkgdImage = bkgd.getImageF(afwMath.Interpolate.NATURAL_SPLINE, afwMath.REDUCE_INTERP_ORDER)
#pybind11#
#pybind11#        self.assertAlmostEqual(np.mean(bkgdImage[0:100, 0:100].getArray(), dtype=np.float64), initialValue)
#pybind11#
#pybind11#    def testBadRows(self):
#pybind11#        """Test that a bad set of rows in an image doesn't cause a failure"""
#pybind11#        initialValue = 20
#pybind11#        mi = afwImage.MaskedImageF(500, 200)
#pybind11#        mi.set((initialValue, 0x0, 1.0))
#pybind11#        im = mi.getImage()
#pybind11#        im[:, 0:100] = np.nan
#pybind11#        del im
#pybind11#        msk = mi.getMask()
#pybind11#        badBits = msk.getPlaneBitMask(['EDGE', 'DETECTED', 'DETECTED_NEGATIVE'])
#pybind11#        msk[0:400, :] |= badBits
#pybind11#        del msk
#pybind11#
#pybind11#        if debugMode:
#pybind11#            ds9.mtv(mi, frame=0)
#pybind11#
#pybind11#        sctrl = afwMath.StatisticsControl()
#pybind11#        sctrl.setAndMask(badBits)
#pybind11#        nx, ny = 17, 17
#pybind11#        bctrl = afwMath.BackgroundControl(nx, ny, sctrl, afwMath.MEANCLIP)
#pybind11#
#pybind11#        bkgd = afwMath.makeBackground(mi, bctrl)
#pybind11#        statsImage = afwMath.cast_BackgroundMI(bkgd).getStatsImage()
#pybind11#        if debugMode:
#pybind11#            ds9.mtv(statsImage, frame=1)
#pybind11#
#pybind11#        # the test is that this doesn't fail if the bug (#2297) is fixed
#pybind11#        for frame, interpStyle in enumerate([afwMath.Interpolate.CONSTANT, afwMath.Interpolate.LINEAR,
#pybind11#                                             afwMath.Interpolate.NATURAL_SPLINE,
#pybind11#                                             afwMath.Interpolate.AKIMA_SPLINE], 2):
#pybind11#            bkgdImage = bkgd.getImageF(interpStyle, afwMath.REDUCE_INTERP_ORDER)
#pybind11#            self.assertEqual(np.mean(bkgdImage[0:100, 0:100].getArray()), initialValue)
#pybind11#            if debugMode:
#pybind11#                ds9.mtv(bkgdImage, frame=frame)
#pybind11#
#pybind11#    def testBadImage(self):
#pybind11#        """Test that an entirely bad image doesn't cause an absolute failure"""
#pybind11#        initialValue = 20
#pybind11#        mi = afwImage.MaskedImageF(500, 200)
#pybind11#        # Check that no good values don't crash (they return NaN), and that a single good value
#pybind11#        # is enough to redeem the entire image
#pybind11#        for pix00 in [np.nan, initialValue]:
#pybind11#            mi.getImage()[:] = np.nan
#pybind11#            mi.getImage()[0, 0] = pix00
#pybind11#
#pybind11#            sctrl = afwMath.StatisticsControl()
#pybind11#            nx, ny = 17, 17
#pybind11#            bctrl = afwMath.BackgroundControl(nx, ny, sctrl, afwMath.MEANCLIP)
#pybind11#
#pybind11#            bkgd = afwMath.makeBackground(mi, bctrl)
#pybind11#
#pybind11#            for interpStyle in [afwMath.Interpolate.CONSTANT, afwMath.Interpolate.LINEAR,
#pybind11#                                afwMath.Interpolate.NATURAL_SPLINE, afwMath.Interpolate.AKIMA_SPLINE]:
#pybind11#                # the test is that this doesn't fail if the bug (#2297) is fixed
#pybind11#                bkgdImage = bkgd.getImageF(interpStyle, afwMath.REDUCE_INTERP_ORDER)
#pybind11#                val = np.mean(bkgdImage[0:100, 0:100].getArray())
#pybind11#
#pybind11#                if np.isfinite(pix00):
#pybind11#                    self.assertEqual(val, pix00)
#pybind11#                else:
#pybind11#                    self.assertTrue(np.isnan(val))
#pybind11#
#pybind11#    def testBackgroundFromStatsImage(self):
#pybind11#        """Check that we can rebuild a Background from a BackgroundMI.getStatsImage()"""
#pybind11#        bgCtrl = afwMath.BackgroundControl(10, 10)
#pybind11#        bkgd = afwMath.cast_BackgroundMI(afwMath.makeBackground(self.image, bgCtrl))
#pybind11#
#pybind11#        interpStyle = afwMath.Interpolate.AKIMA_SPLINE
#pybind11#        undersampleStyle = afwMath.REDUCE_INTERP_ORDER
#pybind11#        bkgdImage = bkgd.getImageF(interpStyle, undersampleStyle)
#pybind11#        self.assertEqual(np.mean(bkgdImage.getArray()), self.val)
#pybind11#        self.assertEqual(interpStyle, bkgd.getAsUsedInterpStyle())
#pybind11#        self.assertEqual(undersampleStyle, bkgd.getAsUsedUndersampleStyle())
#pybind11#
#pybind11#        # OK, we have our background.  Make a copy
#pybind11#        bkgd2 = afwMath.BackgroundMI(self.image.getBBox(), bkgd.getStatsImage())
#pybind11#        del bkgd           # we should be handling the memory correctly, but let's check
#pybind11#        bkgdImage2 = bkgd2.getImageF(interpStyle)
#pybind11#
#pybind11#        self.assertEqual(np.mean(bkgdImage2.getArray()), self.val)
#pybind11#
#pybind11#    def testBackgroundList(self):
#pybind11#        """Test that a BackgroundLists behaves like a list"""
#pybind11#        bgCtrl = afwMath.BackgroundControl(10, 10)
#pybind11#        interpStyle = afwMath.Interpolate.AKIMA_SPLINE
#pybind11#        undersampleStyle = afwMath.REDUCE_INTERP_ORDER
#pybind11#        approxStyle = afwMath.ApproximateControl.UNKNOWN
#pybind11#        approxOrderX = 0
#pybind11#        approxOrderY = 0
#pybind11#        approxWeighting = False
#pybind11#
#pybind11#        backgroundList = afwMath.BackgroundList()
#pybind11#
#pybind11#        for i in range(2):
#pybind11#            bkgd = afwMath.makeBackground(self.image, bgCtrl)
#pybind11#            if i == 0:
#pybind11#                # no need to call getImage
#pybind11#                backgroundList.append((bkgd, interpStyle, undersampleStyle,
#pybind11#                                       approxStyle, approxOrderX, approxOrderY, approxWeighting))
#pybind11#            else:
#pybind11#                backgroundList.append(bkgd)  # Relies on having called getImage; deprecated
#pybind11#
#pybind11#        def assertBackgroundList(bgl):
#pybind11#            self.assertEqual(len(bgl), 2)  # check that len() works
#pybind11#            for a in bgl:                 # check that we can iterate
#pybind11#                pass
#pybind11#            self.assertEqual(len(bgl[0]), 7)  # check that we can index
#pybind11#            # check that we always have a tuple (bkgd, interp, under, approxStyle, orderX, orderY, weighting)
#pybind11#            self.assertEqual(len(bgl[1]), 7)
#pybind11#
#pybind11#        assertBackgroundList(backgroundList)
#pybind11#
#pybind11#        # Check pickling
#pybind11#        new = pickle.loads(pickle.dumps(backgroundList))
#pybind11#        assertBackgroundList(new)
#pybind11#        self.assertEqual(len(new), len(backgroundList))
#pybind11#        for i, j in zip(new, backgroundList):
#pybind11#            self.assertBackgroundEqual(i[0], j[0])
#pybind11#            self.assertEqual(i[1:], j[1:])
#pybind11#
#pybind11#    def assertBackgroundEqual(self, lhs, rhs):
#pybind11#        lhsStats, rhsStats = lhs.getStatsImage(), rhs.getStatsImage()
#pybind11#        self.assertEqual(lhs.getImageBBox(), rhs.getImageBBox())
#pybind11#        self.assertMaskedImagesEqual(lhsStats, rhsStats)
#pybind11#        lhsImage, rhsImage = lhs.getImageF("LINEAR"), rhs.getImageF("LINEAR")
#pybind11#        self.assertImagesEqual(lhsImage, rhsImage)
#pybind11#
#pybind11#    def testApproximate(self):
#pybind11#        """Test I/O for BackgroundLists with Approximate"""
#pybind11#        # approx and interp should be very close, but not the same
#pybind11#        img = self.getParabolaImage(256, 256)
#pybind11#
#pybind11#        # try regular interpolated image (the default)
#pybind11#        bgCtrl = afwMath.BackgroundControl(6, 6)
#pybind11#        bgCtrl.setInterpStyle(afwMath.Interpolate.AKIMA_SPLINE)
#pybind11#        bgCtrl.setUndersampleStyle(afwMath.REDUCE_INTERP_ORDER)
#pybind11#        bkgd = afwMath.makeBackground(img, bgCtrl)
#pybind11#        interpImage = bkgd.getImageF()
#pybind11#
#pybind11#        with lsst.utils.tests.getTempFilePath("_bgi.fits") as bgiFile, \
#pybind11#                lsst.utils.tests.getTempFilePath("_bga.fits") as bgaFile:
#pybind11#            bglInterp = afwMath.BackgroundList()
#pybind11#            bglInterp.append(bkgd)
#pybind11#            bglInterp.writeFits(bgiFile)
#pybind11#
#pybind11#            # try an approx background
#pybind11#            approxStyle = afwMath.ApproximateControl.CHEBYSHEV
#pybind11#            approxOrder = 2
#pybind11#            actrl = afwMath.ApproximateControl(approxStyle, approxOrder)
#pybind11#            bkgd.getBackgroundControl().setApproximateControl(actrl)
#pybind11#            approxImage = bkgd.getImageF()
#pybind11#            bglApprox = afwMath.BackgroundList()
#pybind11#            bglApprox.append(bkgd)
#pybind11#            bglApprox.writeFits(bgaFile)
#pybind11#
#pybind11#            # take a difference and make sure the two are very similar
#pybind11#            interpNp = interpImage.getArray()
#pybind11#            diff = np.abs(interpNp - approxImage.getArray())/interpNp
#pybind11#
#pybind11#            # the image and interp/approx parameters are chosen so these limits
#pybind11#            # will be greater than machine precision for float.  The two methods
#pybind11#            # should be measurably different (so we know we're not just getting the
#pybind11#            # same thing from the getImage() method.  But they should be very close
#pybind11#            # since they're both doing the same sort of thing.
#pybind11#            tolSame = 1.0e-3  # should be the same to this order
#pybind11#            tolDiff = 1.0e-4  # should be different here
#pybind11#            self.assertLess(diff.max(), tolSame)
#pybind11#            self.assertGreater(diff.max(), tolDiff)
#pybind11#
#pybind11#            # now see if we can reload them from files and get the same images we wrote
#pybind11#            interpImage2 = afwMath.BackgroundList().readFits(bgiFile).getImage()
#pybind11#            approxImage2 = afwMath.BackgroundList().readFits(bgaFile).getImage()
#pybind11#
#pybind11#            idiff = interpImage.getArray() - interpImage2.getArray()
#pybind11#            adiff = approxImage.getArray() - approxImage2.getArray()
#pybind11#
#pybind11#            self.assertEqual(idiff.max(), 0.0)
#pybind11#            self.assertEqual(adiff.max(), 0.0)
#pybind11#
#pybind11#    def testBackgroundListIO(self):
#pybind11#        """Test I/O for BackgroundLists"""
#pybind11#        bgCtrl = afwMath.BackgroundControl(10, 10)
#pybind11#        interpStyle = afwMath.Interpolate.AKIMA_SPLINE
#pybind11#        undersampleStyle = afwMath.REDUCE_INTERP_ORDER
#pybind11#        approxOrderX = 6
#pybind11#        approxOrderY = 6
#pybind11#        approxWeighting = True
#pybind11#
#pybind11#        im = self.image.Factory(self.image, self.image.getBBox(afwImage.PARENT))
#pybind11#        arr = im.getArray()
#pybind11#        arr += np.random.normal(size=(im.getHeight(), im.getWidth()))
#pybind11#
#pybind11#        for astyle in afwMath.ApproximateControl.UNKNOWN, afwMath.ApproximateControl.CHEBYSHEV:
#pybind11#            actrl = afwMath.ApproximateControl(astyle, approxOrderX)
#pybind11#            bgCtrl.setApproximateControl(actrl)
#pybind11#
#pybind11#            backgroundList = afwMath.BackgroundList()
#pybind11#            backImage = afwImage.ImageF(im.getDimensions())
#pybind11#            for i in range(2):
#pybind11#                bkgd = afwMath.makeBackground(im, bgCtrl)
#pybind11#                if i == 0:
#pybind11#                    # no need to call getImage
#pybind11#                    backgroundList.append((bkgd, interpStyle, undersampleStyle,
#pybind11#                                           astyle, approxOrderX, approxOrderY, approxWeighting))
#pybind11#                else:
#pybind11#                    backgroundList.append(bkgd)  # Relies on having called getImage; deprecated
#pybind11#
#pybind11#                backImage += bkgd.getImageF(interpStyle, undersampleStyle)
#pybind11#
#pybind11#            with lsst.utils.tests.getTempFilePath(".fits") as fileName:
#pybind11#                backgroundList.writeFits(fileName)
#pybind11#
#pybind11#                backgrounds = afwMath.BackgroundList.readFits(fileName)
#pybind11#
#pybind11#                img = backgrounds.getImage()
#pybind11#                # Check that the read-back image is identical to that generated from the backgroundList
#pybind11#                # round-tripped to disk
#pybind11#                backImage -= img
#pybind11#
#pybind11#                self.assertEqual(np.min(backImage.getArray()), 0.0)
#pybind11#                self.assertEqual(np.max(backImage.getArray()), 0.0)
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
