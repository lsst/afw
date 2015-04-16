#!/usr/bin/env python

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
Tests for Background

Run with:
   ./Background.py
or
   python
   >>> import Background; Background.run()
"""

import math
import os
import sys
import unittest
import numpy as np
import pickle
import numpy

import lsst.utils.tests as utilsTests
import lsst.pex.exceptions
import lsst.afw.image.imageLib as afwImage
import lsst.afw.math as afwMath
import lsst.afw.geom as afwGeom
import lsst.afw.display.ds9 as ds9
import eups

try:
    type(display)
except NameError:
    display = False

numpy.random.seed(1)

# ==== summary to currently implemented tests ====
# getPixel: tests basic functionality of getPixel() method (floats)
# BackgroundTestImages: tests Laher's afwdata/Statistics/*.fits images (doubles)
# testRamp: make sure a constant slope is *exactly* reproduced by the spline model
# testParabola: make sure a quadratic map is *well* reproduced by the spline model

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class BackgroundTestCase(unittest.TestCase):
    
    """A test case for Background"""
    def setUp(self):
        self.val = 10
        self.image = afwImage.ImageF(afwGeom.Extent2I(100, 200))
        self.image.set(self.val)

    def tearDown(self):
        del self.image


        #self.assertAlmostEqual(mean[1], sd/math.sqrt(image2.getWidth()*image2.getHeight()), 10)

    def testOddSize(self):
        '''
        Test for ticket #1781 -- without it, in oddly-sized images
        there is a chunk of pixels on the right/bottom that do not go
        into the fit and are extrapolated.  After this ticket, the
        subimage boundaries are spread more evenly so the last pixels
        get fit as well.  This slightly strange test case checks that
        the interpolant is close to the function at the end.  I could
        not think of an interpolant that would fit exactly, so this
        just puts a limit on the errors.
        '''
        W,H = 2,99
        image = afwImage.ImageF(afwGeom.Extent2I(W,H))
        bgCtrl = afwMath.BackgroundControl(afwMath.Interpolate.LINEAR)
        bgCtrl.setNxSample(2)
        NY = 10
        bgCtrl.setNySample(NY)
        for y in range(H):
            for x in range(W):
                B = 89
                if y < B:
                    image.set(x,y,y)
                else:
                    image.set(x,y,B+(y-B)*-1.) #0.5)
        bobj = afwMath.makeBackground(image, bgCtrl)
        back = bobj.getImageF()
        
        for iy,by in zip([image.get(0,y) for y in range(H)],
                         [ back.get(0,y) for y in range(H)]):
            self.assertTrue( abs(iy - by) < 5 )

        if False:
            import matplotlib
            matplotlib.use('Agg')
            import pylab as plt
            plt.clf()
            IY = [image.get(0,y) for y in range(H)]
            BY = [ back.get(0,y) for y in range(H)]
            for iy,by in zip(IY,BY):
                print 'diff', iy-by
            b = np.linspace(0, H-1, NY+1)
            plt.plot(IY, 'b-', lw=3, alpha=0.5)
            plt.plot(BY, 'r-')
            for y in b:
                plt.axvline(y)
            plt.savefig('bg.png')


        

    def testgetPixel(self):
        """Test the getPixel() function"""


        xcen, ycen = 50, 100
        bgCtrl = afwMath.BackgroundControl(10, 10)
        bgCtrl.setNxSample(5)
        bgCtrl.setNySample(5)
        bgCtrl.getStatisticsControl().setNumIter(3)
        bgCtrl.getStatisticsControl().setNumSigmaClip(3)
        back = afwMath.makeBackground(self.image, bgCtrl)
        
        self.assertEqual(afwMath.cast_BackgroundMI(back).getPixel(xcen, ycen), self.val)


    def testBackgroundTestImages(self):

        imginfolist = []
        #imginfolist.append( ["v1_i1_g_m400_s20_f.fits", 400.05551471441612] ) # cooked to known value
        #imginfolist.append( ["v1_i1_g_m400_s20_f.fits", 400.00295902395123] ) # cooked to known value
        #imginfolist.append( ["v1_i1_g_m400_s20_f.fits", 400.08468385712251] ) # cooked to known value
        #imginfolist.append( ["v1_i1_g_m400_s20_f.fits", 400.00305806663295] ) # cooked to known value
        #imginfolist.append( ["v1_i1_g_m400_s20_f.fits", 400.0035102188698] ) # cooked to known value
        imginfolist.append( ["v1_i1_g_m400_s20_f.fits", 399.9912966583894] ) # cooked to known value
        #imgfiles.append("v1_i1_g_m400_s20_u16.fits")
        #imgfiles.append("v1_i2_g_m400_s20_f.fits"
        #imgfiles.append("v1_i2_g_m400_s20_u16.fits")
        #imgfiles.append("v2_i1_p_m9_f.fits")
        #imgfiles.append("v2_i1_p_m9_u16.fits")
        #imgfiles.append("v2_i2_p_m9_f.fits")
        #imgfiles.append("v2_i2_p_m9_u16.fits")
        
        afwdataDir = eups.productDir("afwdata")
        if not afwdataDir:
            print >> sys.stderr, "Skipping testBackgroundTestImages as afwdata is not setup"
            return
        
        for imginfo in imginfolist:

            imgfile, centerValue = imginfo

            imgPath = afwdataDir + "/Statistics/" + imgfile

            # get the image and header
            dimg = afwImage.DecoratedImageF(imgPath)
            img = dimg.getImage()
            fitsHdr = dimg.getMetadata() # the FITS header

            # get the True values of the mean and stdev
            reqMean  = fitsHdr.getAsDouble("MEANREQ")
            reqStdev = fitsHdr.getAsDouble("SIGREQ")
            naxis1 = img.getWidth()
            naxis2 = img.getHeight()
            
            # create a background control object
            bctrl = afwMath.BackgroundControl(afwMath.Interpolate.AKIMA_SPLINE)
            bctrl.setNxSample(5)
            bctrl.setNySample(5)
            
            # run the background constructor and call the getPixel() and getImage() functions.
            backobj = afwMath.makeBackground(img, bctrl)

            pixPerSubimage = img.getWidth()*img.getHeight()/(bctrl.getNxSample()*bctrl.getNySample())
            stdevInterp = reqStdev/math.sqrt(pixPerSubimage)
            
            # test getPixel()
            testval = afwMath.cast_BackgroundMI(backobj).getPixel(naxis1/2, naxis2/2)
            self.assertAlmostEqual(testval/centerValue, 1, places=7)
            self.assertTrue( abs(testval - reqMean) < 2*stdevInterp )

            # test getImage() by checking the center pixel
            bimg = backobj.getImageF()
            testImgval = bimg.get(naxis1/2, naxis2/2)
            self.assertTrue( abs(testImgval - reqMean) < 2*stdevInterp )
            

    def testRamp(self):

        # make a ramping image (spline should be exact for linear increasing image
        nx = 512
        ny = 512
        rampimg = afwImage.ImageF(afwGeom.Extent2I(nx, ny))
        dzdx, dzdy, z0 = 0.1, 0.2, 10000.0

        for x in range(nx):
            for y in range(ny):
                rampimg.set(x, y, dzdx*x + dzdy*y + z0)
        
        # check corner, edge, and center pixels
        bctrl = afwMath.BackgroundControl(10, 10)
        bctrl.setInterpStyle(afwMath.Interpolate.CUBIC_SPLINE)
        bctrl.setNxSample(6)
        bctrl.setNySample(6)
        bctrl.getStatisticsControl().setNumSigmaClip(20.0)  # something large enough to avoid clipping entirely
        bctrl.getStatisticsControl().setNumIter(1)
        backobj = afwMath.makeBackground(rampimg, bctrl)

        xpixels = [0, nx/2, nx - 1]
        ypixels = [0, ny/2, ny - 1]
        for xpix in xpixels:
            for ypix in ypixels:
                testval = afwMath.cast_BackgroundMI(backobj).getPixel(xpix, ypix)
                self.assertAlmostEqual(testval/rampimg.get(xpix, ypix), 1, 6)

        # Test pickle
        bg = afwMath.cast_BackgroundMI(backobj)
        new = pickle.loads(pickle.dumps(bg))
        self.assertBackgroundEqual(bg, new)

        # Check creation of sub-image
        box = afwGeom.Box2I(afwGeom.Point2I(123, 45), afwGeom.Extent2I(45, 123))
        bgImage = bg.getImageF("AKIMA_SPLINE")
        bgSubImage = afwImage.ImageF(bgImage, box)
        testImage = bg.getImageF(box, "AKIMA_SPLINE")
        self.assertEqual(testImage.getXY0(), bgSubImage.getXY0())
        self.assertEqual(testImage.getDimensions(), bgSubImage.getDimensions())
        self.assertTrue(np.all(testImage.getArray() == bgSubImage.getArray()))

    def getParabolaImage(self, nx, ny, pars=(1.0e-4, 1.0e-4, 0.1, 0.2, 10.0)):
        parabimg = afwImage.ImageF(afwGeom.Extent2I(nx, ny))
        d2zdx2, d2zdy2, dzdx, dzdy, z0 = pars  # no cross-terms
        for x in range(nx):
            for y in range(ny):
                parabimg.set(x, y, d2zdx2*x*x + d2zdy2*y*y + dzdx*x + dzdy*y + z0)
        return parabimg

    def testTicket987(self):
        """This code used to abort; so the test is that it doesn't"""
        afwdataDir = eups.productDir("afwdata")
        if not afwdataDir:
            print >> sys.stderr, "Skipping testTicket987 as afwdata is not setup"
            return

        imagePath = os.path.join(afwdataDir, "DC3a-Sim", "sci", "v5-e0", "v5-e0-c011-a00.sci.fits")
        mimg      = afwImage.MaskedImageF(imagePath)
        binsize   = 512
        bctrl     = afwMath.BackgroundControl("NATURAL_SPLINE")

        ###  Adding this line solves the problem  ###
        # note: by default undersampleStyle is THROW_EXCEPTION 
        bctrl.setUndersampleStyle(afwMath.REDUCE_INTERP_ORDER)
        ################################################

        nx = int(mimg.getWidth()/binsize) + 1
        ny = int(mimg.getHeight()/binsize) + 1

        #print 'Binning', nx, ny
        bctrl.setNxSample(nx)
        bctrl.setNySample(ny)
        image   = mimg.getImage()
        backobj = afwMath.makeBackground(image, bctrl)
        image  -= backobj.getImageF()

    def testTicket1781(self):
        # make an unusual-sized image
        nx = 526
        ny = 154

        parabimg = self.getParabolaImage(nx, ny)

        bctrl = afwMath.BackgroundControl(afwMath.Interpolate.CUBIC_SPLINE)
        bctrl.setNxSample(16)
        bctrl.setNySample(4)
        bctrl.getStatisticsControl().setNumSigmaClip(10.0)  
        bctrl.getStatisticsControl().setNumIter(1)
        backobj = afwMath.makeBackground(parabimg, bctrl)

        #parabimg.writeFits('in.fits')
        #backobj.getImageF().writeFits('out.fits')


    def testParabola(self):

        # make an image which varies parabolicly (spline should be exact for 2rd order polynomial)
        nx = 512
        ny = 512

        parabimg = self.getParabolaImage(nx, ny)
        
        # check corner, edge, and center pixels
        bctrl = afwMath.BackgroundControl(afwMath.Interpolate.CUBIC_SPLINE)
        bctrl.setNxSample(24)
        bctrl.setNySample(24)
        bctrl.getStatisticsControl().setNumSigmaClip(10.0)  
        bctrl.getStatisticsControl().setNumIter(1)
        backobj = afwMath.makeBackground(parabimg, bctrl)

        # debug
        #bimg = backobj.getImageF()
        #ds9.mtv(parabimg)
        #ds9.mtv(bimg, frame=1)
        #parabimg.writeFits("a.fits")
        #bimg.writeFits("b.fits")

        segmentCenter = int(0.5*nx/bctrl.getNxSample())
        xpixels = [segmentCenter, nx/2, nx - segmentCenter]
        ypixels = [segmentCenter, ny/2, ny - segmentCenter]
        for xpix in xpixels:
            for ypix in ypixels:
                testval = afwMath.cast_BackgroundMI(backobj).getPixel(bctrl.getInterpStyle(), xpix, ypix)
                realval = parabimg.get(xpix, ypix)
                #print "Parab: ", xpix, ypix, realval, -(testval - realval)
                # quadratic terms skew the averages of the subimages and the clipped mean for
                # a subimage != value of center pixel.  1/20 counts on a 10000 count sky
                #  is a fair (if arbitrary) test.
                self.assertTrue( abs(testval - realval) < 0.5 )

    def testCFHT_oldAPI(self):
        """Test background subtraction on some real CFHT data"""

        afwdataDir = eups.productDir("afwdata")
        if not afwdataDir:
            print >> sys.stderr, "Skipping testCFHT as afwdata is not setup"
            return

        mi = afwImage.MaskedImageF(os.path.join(afwdataDir,
                                                "CFHT", "D4", "cal-53535-i-797722_1.fits"))
        mi = mi.Factory(mi, afwGeom.Box2I(afwGeom.Point2I(32, 2), afwGeom.Point2I(2079, 4609)), afwImage.LOCAL)

        bctrl = afwMath.BackgroundControl(afwMath.Interpolate.AKIMA_SPLINE)
        bctrl.setNxSample(16)
        bctrl.setNySample(16)
        bctrl.getStatisticsControl().setNumSigmaClip(3.0)  
        bctrl.getStatisticsControl().setNumIter(2)
        backobj = afwMath.makeBackground(mi.getImage(), bctrl)

        if display:
            ds9.mtv(mi, frame = 0)

        im = mi.getImage()
        im -= backobj.getImageF()

        if display:
            ds9.mtv(mi, frame = 1)

            
    def testCFHT(self):
        """Test background subtraction on some real CFHT data"""

        afwdataDir = eups.productDir("afwdata")
        if not afwdataDir:
            print >> sys.stderr, "Skipping testCFHT as afwdata is not setup"
            return

        mi = afwImage.MaskedImageF(os.path.join(afwdataDir,
                                                "CFHT", "D4", "cal-53535-i-797722_1.fits"))
        mi = mi.Factory(mi, afwGeom.Box2I(afwGeom.Point2I(32, 2), afwGeom.Point2I(2079, 4609)), afwImage.LOCAL)

        bctrl = afwMath.BackgroundControl(mi.getWidth()//128, mi.getHeight()//128)
        bctrl.getStatisticsControl().setNumSigmaClip(3.0)  
        bctrl.getStatisticsControl().setNumIter(2)
        backobj = afwMath.makeBackground(mi.getImage(), bctrl)

        if display:
            ds9.mtv(mi, frame = 0)

        im = mi.getImage()
        im -= backobj.getImageF("AKIMA_SPLINE")

        if display:
            ds9.mtv(mi, frame = 1)

        statsImage = afwMath.cast_BackgroundMI(backobj).getStatsImage()

        if display:
            ds9.mtv(backobj.getStatsImage(), frame=2)
            ds9.mtv(backobj.getStatsImage().getVariance(), frame=3)
            
    def testUndersample(self):
        """Test how the program handles nx,ny being too small for requested interp style."""

        # make an image
        nx = 64
        ny = 64
        img = afwImage.ImageF(afwGeom.Extent2I(nx, ny))
        
        # make a background control object
        bctrl = afwMath.BackgroundControl(10, 10)
        bctrl.setInterpStyle(afwMath.Interpolate.CUBIC_SPLINE)
        bctrl.setNxSample(3)
        bctrl.setNySample(3)

        if False:                       # INCREASE_NXNYSAMPLE is no longer supported post #2074
            bctrl.setNxSample(2)
            bctrl.setNySample(2)
            # see if it adjusts the nx,ny values up to 3x3
            bctrl.setUndersampleStyle(afwMath.INCREASE_NXNYSAMPLE)
            backobj = afwMath.makeBackground(img, bctrl)
            self.assertEqual(backobj.getBackgroundControl().getNxSample(), 3)
            self.assertEqual(backobj.getBackgroundControl().getNySample(), 3)
            
        # put nx,ny back to 2 and see if it adjusts the interp style down to linear
        bctrl.setNxSample(2)
        bctrl.setNySample(2)
        bctrl.setUndersampleStyle("REDUCE_INTERP_ORDER")
        backobj = afwMath.makeBackground(img, bctrl)
        backobj.getImageF()             # Need to interpolate background to discover what we actually needed
        self.assertEqual(backobj.getAsUsedInterpStyle(), afwMath.Interpolate.LINEAR)

        # put interp style back up to cspline and see if it throws an exception
        bctrl.setUndersampleStyle("THROW_EXCEPTION")
        def tst(img, bctrl):
            backobj = afwMath.makeBackground(img, bctrl)
            backobj.getImageF("CUBIC_SPLINE") # only now do we see that we have too few points
        utilsTests.assertRaisesLsstCpp(self, lsst.pex.exceptions.InvalidParameterException,
                                       tst, img, bctrl)

        
    def testOnlyOneGridCell(self):
        """Test how the program handles nxSample,nySample being 1x1."""
        
        # try a ramping image ... has an easy analytic solution
        nx = 64
        ny = 64
        img = afwImage.ImageF(afwGeom.Extent2I(nx, ny), 10)
        
        dzdx, dzdy, z0 = 0.1, 0.2, 10000.0
        mean = z0 + dzdx*(nx - 1)/2 + dzdy*(ny - 1)/2  # the analytic solution
        for x in range(nx):
            for y in range(ny):
                img.set(x, y, dzdx*x + dzdy*y + z0)
        
        # make a background control object
        bctrl = afwMath.BackgroundControl(10, 10)
        bctrl.setInterpStyle(afwMath.Interpolate.CONSTANT)
        bctrl.setNxSample(1)
        bctrl.setNySample(1)
        bctrl.setUndersampleStyle(afwMath.THROW_EXCEPTION)
        backobj = afwMath.makeBackground(img, bctrl)
        
        xpixels = [0, nx/2, nx - 1]
        ypixels = [0, ny/2, ny - 1]
        for xpix in xpixels:
            for ypix in ypixels:
                testval = afwMath.cast_BackgroundMI(backobj).getPixel(bctrl.getInterpStyle(), xpix, ypix)
                self.assertAlmostEqual(testval/mean, 1)
        
    def testTicket1681OffByOne(self):
        if False:                       # doesn't seem to actually test anything, and writes b?im.fits
            im = afwImage.ImageF(40, 40); im.set(5, 6, 100);
            nx, ny = im.getWidth()//2, im.getHeight()//2
            print nx, ny
            bctrl = afwMath.BackgroundControl("LINEAR", nx, ny)
            bctrl.setStatisticsProperty(afwMath.MEAN)
            bkd = afwMath.makeBackground(im, bctrl)
            bim = bkd.getImageF()
            im.writeFits("im.fits")
            bim.writeFits("bim.fits")

    def testAdjustLevel(self):
        """Test that we can adjust a background level"""
        sky = 100
        im = afwImage.ImageF(40, 40); im.set(sky);
        nx, ny = im.getWidth()//2, im.getHeight()//2
        bctrl = afwMath.BackgroundControl("LINEAR", nx, ny)
        bkd = afwMath.makeBackground(im, bctrl)
       
        self.assertEqual(afwMath.makeStatistics(bkd.getImageF(), afwMath.MEAN).getValue(), sky)
            
        delta = 123
        bkd += delta
        self.assertEqual(afwMath.makeStatistics(bkd.getImageF(), afwMath.MEAN).getValue(), sky + delta)
        bkd -= delta
        self.assertEqual(afwMath.makeStatistics(bkd.getImageF(), afwMath.MEAN).getValue(), sky)

    def testNaNFromMaskedImage(self):
        """Check that an extensively masked image doesn't lead to NaNs in the background estimation"""
        
        image = afwImage.MaskedImageF(800, 800)
        msk = image.getMask()
        bbox = afwGeom.BoxI(afwGeom.PointI(560, 0), afwGeom.PointI(799, 335))
        smsk = msk.Factory(msk, bbox)
        smsk.set(msk.getPlaneBitMask("DETECTED"))
        
        binSize = 256
        nx = image.getWidth()//binSize + 1
        ny = image.getHeight()//binSize + 1

        sctrl = afwMath.StatisticsControl()
        sctrl.setAndMask(reduce(lambda x, y: x | image.getMask().getPlaneBitMask(y),
                                ['EDGE', 'DETECTED', 'DETECTED_NEGATIVE'], 0x0))

        bctrl = afwMath.BackgroundControl(nx, ny, sctrl, "MEANCLIP")

        bkgd = afwMath.makeBackground(image, bctrl)
        bkgdImage = bkgd.getImageF("NATURAL_SPLINE", "THROW_EXCEPTION")
        if display:
            ds9.mtv(image)
            ds9.mtv(bkgdImage, frame=1)

        self.assertFalse(np.isnan(bkgdImage.get(0,0)))

        # Check that the non-string API works too
        bkgdImage = bkgd.getImageF(afwMath.Interpolate.NATURAL_SPLINE, afwMath.THROW_EXCEPTION)

    def testBadAreaFailsSpline(self):
        """Check that a NaN in the stats image doesn't cause spline interpolation to fail (#2734)"""
        
        image = afwImage.ImageF(15, 9)
        for y in range(image.getHeight()):
            for x in range(image.getWidth()):
                image.set(x, y, 1 + 2*y) # n.b. linear, which is what the interpolation will fall back to

        # Set the right corner to NaN.  This will mean that we have too few points for a spline interpolator
        binSize = 3
        image[-binSize:, -binSize:] = np.nan

        nx = image.getWidth()//binSize
        ny = image.getHeight()//binSize

        sctrl = afwMath.StatisticsControl()
        bctrl = afwMath.BackgroundControl(nx, ny, sctrl, afwMath.MEANCLIP)

        bkgd = afwMath.makeBackground(image, bctrl)
        if display:
            ds9.mtv(image)
            ds9.mtv(afwMath.cast_BackgroundMI(bkgd).getStatsImage(), frame=1)
        #
        # Should throw if we don't permit REDUCE_INTERP_ORDER
        #
        utilsTests.assertRaisesLsstCpp(self, lsst.pex.exceptions.OutOfRangeException,
                                       bkgd.getImageF, afwMath.Interpolate.NATURAL_SPLINE)
        #
        # The interpolation should fall back to linear for the right part of the image
        # where the NaNs don't permit spline interpolation (n.b. this happens to be exact)
        #
        bkgdImage = bkgd.getImageF(afwMath.Interpolate.NATURAL_SPLINE, afwMath.REDUCE_INTERP_ORDER)
            
        if display:
            ds9.mtv(bkgdImage, frame=2)

        image -= bkgdImage
        self.assertEqual(afwMath.makeStatistics(image, afwMath.MEAN).getValue(), 0.0)

    def testBadPatch(self):
        """Test that a large bad patch of an image doesn't cause an absolute failure"""

        initialValue = 20
        mi = afwImage.MaskedImageF(500, 200)
        mi.set((initialValue, 0x0, 1.0))
        im = mi.getImage()
        im[0:200, :] = np.nan
        del im
        msk = mi.getMask()
        badBits = msk.getPlaneBitMask(['EDGE', 'DETECTED', 'DETECTED_NEGATIVE'])
        msk[0:400, :] |= badBits
        del msk
        
        if display:
            ds9.mtv(mi, frame=0)

        sctrl = afwMath.StatisticsControl()
        sctrl.setAndMask(badBits)
        nx, ny = 17, 17
        bctrl = afwMath.BackgroundControl(nx, ny, sctrl, afwMath.MEANCLIP)

        bkgd = afwMath.makeBackground(mi, bctrl)
        statsImage = afwMath.cast_BackgroundMI(bkgd).getStatsImage()
        if display:
            ds9.mtv(statsImage, frame=1)

        # the test is that this doesn't fail if the bug (#2297) is fixed
        bkgdImage = bkgd.getImageF(afwMath.Interpolate.NATURAL_SPLINE, afwMath.REDUCE_INTERP_ORDER)
        self.assertEqual(np.mean(bkgdImage[0:100, 0:100].getArray()), initialValue)
        if display:
            ds9.mtv(bkgdImage, frame=2)
        #
        # Check that we can fix the NaNs in the statsImage
        #
        defaultValue = 10
        sim = statsImage.getImage().getArray()
        sim[np.isnan(sim)] = defaultValue # replace NaN by defaultValue
        bkgdImage = bkgd.getImageF(afwMath.Interpolate.NATURAL_SPLINE, afwMath.REDUCE_INTERP_ORDER)

        self.assertEqual(np.mean(bkgdImage[0:100, 0:100].getArray()), defaultValue)
            
    def testBadRows(self):
        """Test that a bad set of rows in an image doesn't cause a failure"""

        initialValue = 20
        mi = afwImage.MaskedImageF(500, 200)
        mi.set((initialValue, 0x0, 1.0))
        im = mi.getImage()
        im[:, 0:100] = np.nan
        del im
        msk = mi.getMask()
        badBits = msk.getPlaneBitMask(['EDGE', 'DETECTED', 'DETECTED_NEGATIVE'])
        msk[0:400, :] |= badBits
        del msk
        
        if display:
            ds9.mtv(mi, frame=0)

        sctrl = afwMath.StatisticsControl()
        sctrl.setAndMask(badBits)
        nx, ny = 17, 17
        bctrl = afwMath.BackgroundControl(nx, ny, sctrl, afwMath.MEANCLIP)

        bkgd = afwMath.makeBackground(mi, bctrl)
        statsImage = afwMath.cast_BackgroundMI(bkgd).getStatsImage()
        if display:
            ds9.mtv(statsImage, frame=1)

        # the test is that this doesn't fail if the bug (#2297) is fixed
        bkgdImage = bkgd.getImageF(afwMath.Interpolate.NATURAL_SPLINE, afwMath.REDUCE_INTERP_ORDER)
        self.assertEqual(np.mean(bkgdImage[0:100, 0:100].getArray()), initialValue)
        if display:
            ds9.mtv(bkgdImage, frame=2)

    def testBadImage(self):
        """Test that an entirely bad image doesn't cause an absolute failure"""

        initialValue = 20
        mi = afwImage.MaskedImageF(500, 200)
        #
        # Check that no good values don't crash (they return NaN), and that a single good value
        # is enough to redeem the entire image
        #
        for pix00 in [np.nan, initialValue]:
            mi.getImage()[:] = np.nan
            mi.getImage()[0, 0] = pix00

            sctrl = afwMath.StatisticsControl()
            nx, ny = 17, 17
            bctrl = afwMath.BackgroundControl(nx, ny, sctrl, afwMath.MEANCLIP)

            bkgd = afwMath.makeBackground(mi, bctrl)
            statsImage = afwMath.cast_BackgroundMI(bkgd).getStatsImage()

            # the test is that this doesn't fail if the bug (#2297) is fixed
            bkgdImage = bkgd.getImageF(afwMath.Interpolate.NATURAL_SPLINE, afwMath.REDUCE_INTERP_ORDER)
            val = np.mean(bkgdImage[0:100, 0:100].getArray())

            if np.isfinite(pix00):
                self.assertEqual(val, pix00)
            else:
                self.assertTrue(np.isnan(val))
            
    def testBackgroundFromStatsImage(self):
        """Check that we can rebuild a Background from a BackgroundMI.getStatsImage()"""

        bgCtrl = afwMath.BackgroundControl(10, 10)
        bkgd = afwMath.cast_BackgroundMI(afwMath.makeBackground(self.image, bgCtrl))

        interpStyle = afwMath.Interpolate.AKIMA_SPLINE
        undersampleStyle = afwMath.REDUCE_INTERP_ORDER
        bkgdImage = bkgd.getImageF(interpStyle, undersampleStyle)
        self.assertEqual(np.mean(bkgdImage.getArray()), self.val)
        self.assertEqual(interpStyle, bkgd.getAsUsedInterpStyle())
        self.assertEqual(undersampleStyle, bkgd.getAsUsedUndersampleStyle())
        #
        # OK, we have our background.  Make a copy
        #
        bkgd2 = afwMath.BackgroundMI(self.image.getBBox(), bkgd.getStatsImage())
        del bkgd; bkgd = None           # we should be handling the memory correctly, but let's check
        bkgdImage2 = bkgd2.getImageF(interpStyle)

        self.assertEqual(np.mean(bkgdImage2.getArray()), self.val)
        
    def testBackgroundList(self):
        """Test that a BackgroundLists behaves like a list"""
        bgCtrl = afwMath.BackgroundControl(10, 10)
        interpStyle = afwMath.Interpolate.AKIMA_SPLINE
        undersampleStyle = afwMath.REDUCE_INTERP_ORDER
        approxStyle = afwMath.ApproximateControl.UNKNOWN
        approxOrderX = 0
        approxOrderY = 0

        backgroundList = afwMath.BackgroundList()
        backImage = afwImage.ImageF(self.image.getDimensions())
        for i in range(2):
            bkgd = afwMath.makeBackground(self.image, bgCtrl)
            if i == 0:
                # no need to call getImage
                backgroundList.append((bkgd, interpStyle, undersampleStyle,
                                       approxStyle, approxOrderX, approxOrderY))
            else:
                backgroundList.append(bkgd) # Relies on having called getImage; deprecated

        def assertBackgroundList(bgl):
            self.assertEqual(len(bgl), 2) # check that len() works
            for a in bgl:                 # check that we can iterate
                pass
            self.assertEqual(len(bgl[0]), 6) # check that we can index
            # check that we always have a tuple (bkgd, interp, under, approxStyle, orderX, orderY)
            self.assertEqual(len(bgl[1]), 6) 

        assertBackgroundList(backgroundList)

        # Check pickling
        new = pickle.loads(pickle.dumps(backgroundList))
        assertBackgroundList(new)
        self.assertEqual(len(new), len(backgroundList))
        for i, j in zip(new, backgroundList):
            self.assertBackgroundEqual(i[0], j[0])
            self.assertEqual(i[1:], j[1:])

    def assertBackgroundEqual(self, lhs, rhs):
        lhsStats, rhsStats = lhs.getStatsImage(), rhs.getStatsImage()
        self.assertEqual(lhs.getImageBBox(), rhs.getImageBBox())
        self.assertTrue(np.all(lhsStats.getImage().getArray() == rhsStats.getImage().getArray()))
        self.assertTrue(np.all(lhsStats.getMask().getArray() == rhsStats.getMask().getArray()))
        self.assertTrue(np.all(lhsStats.getVariance().getArray() == rhsStats.getVariance().getArray()))


    def testApproximate(self):
        """Test I/O for BackgroundLists with Approximate"""

        # approx and interp should be very close, but not the same
        img = self.getParabolaImage(256, 256)

        # try regular interpolated image (the default)
        bgCtrl = afwMath.BackgroundControl(6, 6)
        bgCtrl.setInterpStyle(afwMath.Interpolate.AKIMA_SPLINE)
        bgCtrl.setUndersampleStyle(afwMath.REDUCE_INTERP_ORDER)
        bkgd = afwMath.makeBackground(img, bgCtrl)
        interpImage = bkgd.getImageF()
        bglInterp = afwMath.BackgroundList()
        bglInterp.append(bkgd)
        bgiFile = "tests/bgInterp.fits"
        bglInterp.writeFits(bgiFile)

        # try an approx background
        approxStyle = afwMath.ApproximateControl.CHEBYSHEV
        approxOrder = 2
        actrl = afwMath.ApproximateControl(approxStyle, approxOrder)
        bkgd.getBackgroundControl().setApproximateControl(actrl)
        approxImage = bkgd.getImageF()
        bglApprox = afwMath.BackgroundList()
        bglApprox.append(bkgd)
        bgaFile = "tests/bgApprox.fits"
        bglApprox.writeFits(bgaFile)

        # take a difference and make sure the two are very similar
        interpNp = interpImage.getArray()
        diff = numpy.abs(interpNp - approxImage.getArray())/interpNp

        # the image and interp/approx parameters are chosen so these limits
        # will be greater than machine precision for float.  The two methods
        # should be measurably different (so we know we're not just getting the
        # same thing from the getImage() method.  But they should be very close
        # since they're both doing the same sort of thing.
        tolSame = 1.0e-3  # should be the same to this order
        tolDiff = 1.0e-4  # should be different here
        self.assertLess(diff.max(), tolSame)
        self.assertGreater(diff.max(), tolDiff)
        #print "Interp/Approx diff min and max:", diff.min(), diff.max()

        # now see if we can reload them from files and get the same images we wrote
        interpImage2 = afwMath.BackgroundList().readFits(bgiFile).getImage()
        approxImage2 = afwMath.BackgroundList().readFits(bgaFile).getImage()

        idiff = interpImage.getArray() - interpImage2.getArray()
        adiff = approxImage.getArray() - approxImage2.getArray()

        self.assertEqual(idiff.max(), 0.0)
        self.assertEqual(adiff.max(), 0.0)
        #print "Reloaded interp and approx diff max:", idiff.max(), adiff.max()

    def testBackgroundListIO(self):
        """Test I/O for BackgroundLists"""
        bgCtrl = afwMath.BackgroundControl(10, 10)
        interpStyle = afwMath.Interpolate.AKIMA_SPLINE
        undersampleStyle = afwMath.REDUCE_INTERP_ORDER
        approxOrderX = 6
        approxOrderY = 6

        im = self.image.Factory(self.image, self.image.getBBox(afwImage.PARENT))
        arr = im.getArray()
        arr += numpy.random.normal(size=(im.getHeight(),im.getWidth()))

        for astyle in afwMath.ApproximateControl.UNKNOWN, afwMath.ApproximateControl.CHEBYSHEV:

            actrl = afwMath.ApproximateControl(astyle, approxOrderX)
            bgCtrl.setApproximateControl(actrl)

            backgroundList = afwMath.BackgroundList()
            backImage = afwImage.ImageF(im.getDimensions())
            for i in range(2):
                bkgd = afwMath.makeBackground(im, bgCtrl)
                if i == 0:
                    # no need to call getImage
                    backgroundList.append((bkgd, interpStyle, undersampleStyle,
                                           astyle, approxOrderX, approxOrderY))
                else:
                    backgroundList.append(bkgd) # Relies on having called getImage; deprecated

                backImage += bkgd.getImageF(interpStyle, undersampleStyle)

            fileName = "backgroundList.fits"
            try:
                backgroundList.writeFits(fileName)

                backgrounds = afwMath.BackgroundList.readFits(fileName)
            finally:
                if os.path.exists(fileName):
                    os.unlink(fileName)

            img = backgrounds.getImage()
            #
            # Check that the read-back image is identical to that generated from the backgroundList
            # round-tripped to disk
            #
            backImage -= img

            self.assertEqual(np.min(backImage.getArray()), 0.0)
            self.assertEqual(np.max(backImage.getArray()), 0.0)


def suite():
    """Returns a suite containing all the test cases in this module."""

    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(BackgroundTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit = False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)
