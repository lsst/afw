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

# -*- python -*-
"""
Sundry tests on the camera distortion class

Run with:
   python distort.py
"""

import os, sys
import math
import unittest

import numpy

import eups

import lsst.pex.policy           as pexPolicy
import lsst.utils.tests          as utilsTests
import lsst.afw.geom             as afwGeom
import lsst.afw.geom.ellipses    as geomEllip
import lsst.afw.cameraGeom       as cameraGeom
import lsst.afw.cameraGeom.utils as cameraGeomUtils
from lsst.afw.cameraGeom.testUtils import DetectorWrapper

class DistortionTestCase(unittest.TestCase):

    def setUp(self):
	self.prynt = False
        nx, ny = 6001, 8001
        pixelSize = 1.0 # mm
        self.det = DetectorWrapper(pixelSize=pixelSize, ampExtent=afwGeom.Extent2I(nx,ny)).detector
        self.center = afwGeom.Point2D(int(0.5*nx), int(0.5*ny)))
        
	# try the suprimecam numbers
	self.coeffs = [0.0, 1.0, 7.16417e-08, 3.03146e-10, 5.69338e-14, -6.61572e-18]

	self.xs = [0.0, 1000.0, 5000.0]
	self.ys = [0.0, 1000.0, 4000.0]
	
    def tearDown(self):
        del self.det


        
    def roundTrip(self, dist, *args):

	if len(args) == 2:
	    x, y = args
	    p = afwGeom.Point2D(x, y)
	    pDist = dist.distort(p, self.det)
	    pp    = dist.undistort(pDist, self.det)

	    if self.prynt:
		print "p:     %.12f %.12f" % (p.getX(),     p.getY())
		print "pDist: %.12f %.12f" % (pDist.getX(), pDist.getY())
		print "pp:    %.12f %.12f" % (pp.getX(),    pp.getY())
		
	    self.assertAlmostEqual(pp.getX(), p.getX())
	    self.assertAlmostEqual(pp.getY(), p.getY())


	if len(args) == 3:
	    x, y, m = args
	    ixx, iyy, ixy = m

	    p = afwGeom.Point2D(x, y)
	    pDist = dist.distort(p, self.det)
	    m = geomEllip.Quadrupole(ixx, iyy, ixy)
            mDist = dist.distort(p, m, self.det)
	    mm    = dist.undistort(pDist, mDist, self.det)
	    r0 = math.sqrt(x*x + y*y)

	    theta = math.atan2(y,x)
	    #dx = 0.001*math.cos(theta)
	    #dy = 0.001*math.sin(theta)
	    #dr = math.sqrt(dx*dx+dy*dy)
	    scale = 1.0000001
	    p2 = afwGeom.Point2D(scale*x, scale*y)
	    p2Dist = dist.distort(p2, self.det)

	    r1 = math.sqrt(pDist.getX()*pDist.getX() + pDist.getY()*pDist.getY())
	    r2 = math.sqrt(p2Dist.getX()*p2Dist.getX() + p2Dist.getY()*p2Dist.getY())
	    if r0 > 0:
		drTest = (r2 - r1)/((scale-1.0)*r0)
	    else:
		drTest = 0.0
	    
	    if self.prynt:
		print "m:     %.12f %.12f %.12f" % (m.getIxx(), m.getIyy(), m.getIxy())
		print "mDist: %.12f %.12f %.12f" % (mDist.getIxx(), mDist.getIyy(), mDist.getIxy())
		print "mp:    %.12f %.12f %.12f" % (mm.getIxx(), mm.getIyy(), mm.getIxy())

		ixyTmp = m.getIxy() if m.getIxy() != 0.0 else 1.0
		print "err:   %.12f %.12f %.12f" % (
		    (m.getIxx()-mm.getIxx())/m.getIxx(),
		    (m.getIyy()-mm.getIyy())/m.getIyy(),
		    (m.getIxy()-mm.getIxy())/ixyTmp,
		    )
		
	    self.assertAlmostEqual(mm.getIxx(), m.getIxx()) 
	    self.assertAlmostEqual(mm.getIyy(), m.getIyy()) 
	    self.assertAlmostEqual(mm.getIxy(), m.getIxy()) 
	    


    def testRoundTrip(self):
	
	x, y = 1.0, 1.0

	# try NullDistortion
	ndist = cameraGeom.NullDistortion()
	self.roundTrip(ndist, x, y)
	
	# try RadialPolyDistortion
	rdist = cameraGeom.RadialPolyDistortion(self.coeffs)
	self.roundTrip(rdist, x, y)


    def testDistortionPointerInDetector(self):

	# no distortion object present by default
	dist = self.det.getDistortion()
        self.assertTrue(dist is None)

	# make sure we can set a radialpoly and round-trip it.
	x, y = 1.0, 1.0
	p = afwGeom.Point2D(x, y)
	self.det.setDistortion(cameraGeom.RadialPolyDistortion(self.coeffs))
	self.roundTrip(self.det.getDistortion(), x, y)


    def tryAFewCoords(self, dist, moment):
	for x in self.xs:
	    for y in self.ys:
		if self.prynt:
		    print x, y
		self.roundTrip(dist, x, y, moment)
	

    def testMomentDistortion(self):

	moments = [
	    [1.0, 1.0, 0.0],
	    [1.0, 1.0, 0.1],
	    ]
	
	nDist = cameraGeom.NullDistortion()
	rDist = cameraGeom.RadialPolyDistortion(self.coeffs)

	for moment in moments:
	    self.tryAFewCoords(rDist, moment)
	    self.tryAFewCoords(nDist, moment)


    def testDistortionInACamera(self):

	policyFile = cameraGeomUtils.getGeomPolicy(os.path.join(eups.productDir("afw"),
								"tests", "TestCameraGeom.paf"))
	pol = pexPolicy.Policy(policyFile)
	pol = cameraGeomUtils.getGeomPolicy(pol)
	cam = cameraGeomUtils.makeCamera(pol)

	# see if the distortion object made it into the camera object
	dist = cam.getDistortion()
	self.tryAFewCoords(dist, [1.0, 1.0, 0.1])

	# see if the distortion object is accessible in the ccds
	for raft in cam:
	    for ccd in cameraGeom.cast_Raft(raft):
		ccd = cameraGeom.cast_Ccd(ccd)
		if self.prynt:
		    print "CCD id: ", ccd.getId()
		ccdDist = ccd.getDistortion()
		self.tryAFewCoords(dist, [1.0, 1.0, 0.1])


    def testzAxisCases(self):

	r = 1000.0
	iqq = geomEllip.Quadrupole(1.0, 1.0, 0.0)
	
	px    = afwGeom.Point2D(r, 0.0)
	py    = afwGeom.Point2D(0.0, r)
	pxy   = afwGeom.Point2D(r/numpy.sqrt(2.0), r/numpy.sqrt(2.0))

	nDist = cameraGeom.NullDistortion()
	rDist = cameraGeom.RadialPolyDistortion(self.coeffs)

	rcoeffs = self.coeffs[:]
	rcoeffs.reverse()

	r2Known = numpy.polyval(rcoeffs, r)
	epsilon = 1.0e-6
	drKnown = -(r2Known - numpy.polyval(rcoeffs, r+epsilon))/epsilon

        points = [px, py, pxy]
        scalings = [drKnown**2, drKnown**2, 0.5*(drKnown**2-1.0)]
	for i in range(3):
            p = points[i]
            scale = scalings[i]

            # check the point
	    p2 = rDist.distort(p, self.det)
	    x, y = p2.getX(), p2.getY()
	    r2Calc = numpy.hypot(x, y)
	    if self.prynt:
		print "r2known,r2Calc: ", r2Known, r2Calc
	    self.assertAlmostEqual(r2Known, r2Calc)

            # check the moment
            iqq2 = rDist.distort(p, iqq, self.det)
            iqqList = [iqq2.getIxx(), iqq2.getIyy(), iqq2.getIxy()]
            if self.prynt:
                print "scale: ", scale, iqqList[i]
            self.assertAlmostEqual(scale, iqqList[i])

    def testNullDistortionDefaultCcd(self):
        """Test that we can use a NullDistortion even if the Detector is default-constructed"""
        ccd = cameraGeom.Ccd(cameraGeom.Id("dummy"))
        distorter = cameraGeom.NullDistortion()

        pin = afwGeom.PointD(0, 0)
        pout = distorter.undistort(pin, ccd)

        self.assertEqual(pin, pout)

        ein = geomEllip.Quadrupole(1, 0, 1)
        eout = distorter.distort(pin, ein, ccd)
        
        self.assertEqual(ein, eout)

#################################################################
# Test suite boiler plate
#################################################################
def suite():
    """Returns a suite containing all the test cases in this module."""

    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(DistortionTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(exit = False):
    """Run the tests"""
    utilsTests.run(suite(), exit)

if __name__ == "__main__":
    run(True)
