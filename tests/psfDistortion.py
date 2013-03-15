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

# todo:
# - growth curves
# - 

import math
import pdb                          # we may want to say pdb.set_trace()
import unittest

import numpy
import eups

import lsst.pex.policy          as pexPolicy
import lsst.afw.image           as afwImage
import lsst.afw.math            as afwMath
import lsst.afw.detection       as afwDet
import lsst.afw.geom            as afwGeom
import lsst.afw.geom.ellipses   as geomEllip
import lsst.afw.cameraGeom      as cameraGeom
import lsst.afw.cameraGeom.utils as cameraUtils

import lsst.utils.tests         as utilsTests

import lsst.afw.display.ds9     as ds9

try:
    type(verbose)
except NameError:
    verbose = 0

try:
    display
except NameError:
    display = False



# a helper function to plant a source with known shape    
def plantEllipse(nx, ny, ixx, iyy, ixy):

    tmp = 0.25*(ixx-iyy)**2 + ixy**2
    a = math.sqrt(0.5*(ixx+iyy) + numpy.sqrt(tmp))
    b = math.sqrt(0.5*(ixx+iyy) - numpy.sqrt(tmp))
    theta = 0.5*numpy.arctan2(2.0*ixy, ixx-iyy)
    c, s = math.cos(theta), math.sin(theta)

    img = afwImage.ImageD(nx, ny, 0.0)
    for iy in range(ny):
        dy = iy - ny/2
        for ix in range(nx):
            dx = ix - nx/2
            u =  c*dx + s*dy
            v = -s*dx + c*dy
            val = math.exp(-0.5*((u/a)**2 + (v/b)**2))
            img.set(ix, iy, val)
                
    return img


# a helper function to measure the adaptive moments (naively)
# note: the measure code is in meas_algorithms, so repeat
# a minimal shape measurement here for testing purposes.
def quickAndDirtyShape(img, p):

    sumIx = sumIy = sumIxx = sumIyy = sumIxy = sumI = 0.0
    rmax = 0.5*img.getWidth()
    for iy in range(img.getHeight()):
        y = iy - (p.getY() - img.getY0())
        for ix in range(img.getWidth()):
            I = img.get(ix, iy)
            x = ix - (p.getX() - img.getX0())
            r = math.sqrt(x*x+ y*y)

            if r < rmax:
                sumIx += I*x
                sumIy += I*y
                sumIxx += I*x*x
                sumIyy += I*y*y
                sumIxy += I*x*y
                sumI += I

    # compute the adaptive moments
    xbar = sumIx/sumI
    ybar = sumIy/sumI
    xxbar = sumIxx/sumI - xbar**2
    yybar = sumIyy/sumI - ybar**2
    xybar = sumIxy/sumI - xbar*ybar
    
    # compute a, b, theta
    foo = 0.5*(xxbar + yybar)
    bar = math.sqrt(0.25*(xxbar - yybar)**2 + xybar**2)
    a = math.sqrt(foo + bar)
    b = math.sqrt(foo - bar)
    theta = 0.5*math.atan2(2.0*xybar, xxbar-yybar)
    
    return a, b, 180.0*theta/math.pi, xxbar, yybar, xybar




#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class PsfDistortionTestCase(unittest.TestCase):
    """Test the distortion of the PSF."""

    def setUp(self):
        self.nx, self.ny = 513, 513 #2048, 4096

        self.lanczosOrder = 5
	self.sCamCoeffs = [0.0, 1.0, 7.16417e-08, 3.03146e-10, 5.69338e-14, -6.61572e-18]
	self.sCamDistorter = cameraGeom.RadialPolyDistortion(self.sCamCoeffs, True, self.lanczosOrder)
	
    def tearDown(self):
	del self.sCamDistorter
        
    def testPsfDistortion(self):

	#distorter = cameraGeom.NullDistortion() #self.sCamDistorter #Exag
	distorter = self.sCamDistorter

	# set the psf
	kwid = 55
	psfSigma = 4.5
	psf = afwDet.DoubleGaussianPsf(kwid, kwid, psfSigma, psfSigma, 0.0)

        
	# create a detector which is offset from the boresight
        pixelSize = 0.01 # mm
        allPixels = afwGeom.BoxI(afwGeom.PointI(0, 0), afwGeom.ExtentI(self.nx, self.ny))
        detector = cameraUtils.makeDefaultCcd(allPixels, pixelSize=pixelSize)
        detector.setCenterPixel(afwGeom.Point2D(self.nx/2, self.ny/2))
        # try the upper right corner of chip 0 on suprimecam
        cenPixX, cenPixY = 5000.0, 4000.0
        detector.setCenter(cameraGeom.FpPoint(cenPixX*pixelSize, cenPixY*pixelSize))
        
        detector.setDistortion(distorter)
        psf.setDetector(detector)

        
	settings = {'scale': 'minmax', 'zoom':"to fit", 'mask':'transparency 80'}

        # use a point in the middle of the test image
        x = self.nx//2
        y = self.ny//2
        p = afwGeom.Point2D(x,y) # this is our **measured** coordinate
        pOrig = distorter.undistort(p, detector)  # this is where p would be without optical distortion

        ########################################################
        # check that the camera behaves as expected
        pos = detector.getPositionFromPixel(p)
        pix = detector.getPixelFromPosition(pos)
        print "posmm, pospix, pix", pos.getMm(), pos.getPixels(detector.getPixelSize()), pix
        posPix = pos.getPixels(detector.getPixelSize())
        # note that p is in the center of the ccd
        self.assertEqual(posPix.getX(), cenPixX)
        self.assertEqual(posPix.getY(), cenPixY)
        self.assertEqual(pix.getX(), x)
        self.assertEqual(pix.getY(), y)


        ########################################################
        # compare the measured shear in a psf image to the expected value
        
        # get the expected shear at p
        q = distorter.distort(pOrig, geomEllip.Quadrupole(), detector)
        ax = geomEllip.Axes(q)
        aKnown = ax.getA()
        bKnown = ax.getB()
        thetaKnown = ax.getTheta()*180.0/math.pi
        print "Shear at p: ", ax, thetaKnown


        # make a plain PSF
        doDistort = False # the default is True
        psfImg               = psf.computeImage(p, True, doDistort)

        # compute a PSF at p
        psfImgDistInternally = psf.computeImage(p)

        # make a plain one and distort it ourselves
        # --> note that we use the undistorted pOrig ... that's where p was before the optics
        #psfImgOrig = psf.computeImage(pOrig, True, doDistort)
        #psfImgDistByUs       = distorter.distort(pOrig, psfImgOrig, detector, 0.0)
        #shift = p - afwGeom.Extent2D(pOrig)
        #afwMath.offsetImage(psfImgDistByUs, shift.getX(), shift.getY(), "lanczos5", 5) 
        psfImgDistByUs       = distorter.distort(p, psfImg, detector, 0.0)

        # to display, we'll trim off the edge of the original so it's the same size as the distorted.
        wid2 = psfImgDistInternally.getWidth()
        edge = (psfImg.getWidth() - wid2)/2
        box = afwGeom.Box2I(afwGeom.Point2I(edge, edge), afwGeom.Extent2I(wid2,wid2))
        if display:
            ds9.mtv(afwImage.ImageD(psfImg, box), frame=1, title="psf", settings=settings)
            ds9.mtv(psfImgDistInternally, frame=2, title="psfDist", settings=settings)
            ds9.mtv(afwImage.ImageD(psfImgDistByUs, box), frame=3, title="psfDist2", settings=settings)

        # first make sure we can plant a known quantity and measure it
        # quickAndDirtyShape() must be tested to be used itself as a tester
        sigma = 1.0
        img = plantEllipse(kwid, kwid, sigma, sigma, 0.0)
        a, b, theta, ixx, iyy, ixy = quickAndDirtyShape(img, afwGeom.Point2D(kwid/2,kwid/2))
        print "planted:", a/sigma, b/sigma, theta, ixx/sigma**2, iyy/sigma**2, ixy/sigma**2
        prec = 6
        self.assertAlmostEqual(a, sigma, prec)
        self.assertAlmostEqual(b, sigma, prec)
        self.assertAlmostEqual(ixx, sigma**2, prec)
        self.assertAlmostEqual(iyy, sigma**2, prec)
        
        # try 4% shear along theta=45
        shear = 1.04
        q = geomEllip.Quadrupole(geomEllip.Axes(shear*sigma, sigma, math.pi/4.0))
        img = plantEllipse(kwid, kwid, q.getIxx(), q.getIyy(), q.getIxy())
        a, b, theta, ixx, iyy, ixy = quickAndDirtyShape(img, afwGeom.Point2D(kwid/2,kwid/2))
        print "sheared 4%:", a/sigma, b/sigma, theta, ixx/sigma**2, iyy/sigma**2, ixy/sigma**2
        self.assertAlmostEqual(a, shear*sigma, prec)
        self.assertAlmostEqual(b, sigma, prec)
        self.assertAlmostEqual(theta, 45.0, prec)
        

        # now use quickAndDirty to measure the PSFs we created
        a, b, theta, ixx, iyy, ixy = quickAndDirtyShape(psfImg, p)
        print "psfImg:", a/psfSigma, b/psfSigma, theta, ixx/psfSigma**2, iyy/psfSigma**2, ixy/psfSigma**2
        self.assertAlmostEqual(a, psfSigma, prec)
        self.assertAlmostEqual(b, psfSigma, prec)

        
        print "known Theta = ", thetaKnown
        a, b, theta, ixx, iyy, ixy = quickAndDirtyShape(psfImgDistInternally, p)
        print "warpIntern:", a/psfSigma, b/psfSigma, theta, ixx/psfSigma**2, iyy/psfSigma**2, ixy/psfSigma**2
        self.assertTrue(abs(a/psfSigma - aKnown) < 0.01)
        self.assertTrue(abs(b/psfSigma - bKnown) < 0.01)
        self.assertTrue(abs(theta - thetaKnown) < 0.5) # half a degree

        a, b, theta, ixx, iyy, ixy = quickAndDirtyShape(psfImgDistByUs, p)
        print "warpExtern:", a/psfSigma, b/psfSigma, theta, ixx/psfSigma**2, iyy/psfSigma**2, ixy/psfSigma**2
        self.assertTrue(abs(a/psfSigma - aKnown) < 0.01)
        self.assertTrue(abs(b/psfSigma - bKnown) < 0.01)
        self.assertTrue(abs(theta - thetaKnown) < 0.5)


        
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(PsfDistortionTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)

    return unittest.TestSuite(suites)

def run(exit=False):
    """Run the tests"""
    utilsTests.run(suite(), exit)
 
if __name__ == "__main__":
    run(True)


