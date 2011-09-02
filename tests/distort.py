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

import lsst.utils.tests    as utilsTests
import lsst.afw.geom       as afwGeom
import lsst.afw.cameraGeom as cameraGeom

class DistortionTestCase(unittest.TestCase):

    def setUp(self):
	self.prynt = True #False
	#self.coeffs = [0.0, 1.0, 1.0e-3, 1.0e-6, 1.0e-9, 1.0e-12, 1.0e-15]
	self.coeffs = [0.0, 1.0, 7.16417e-08, 3.03146e-10, 5.69338e-14, -6.61572e-18]

	self.xs = [0.0, 1000.0, 5000.0]
	self.ys = [0.0, 1000.0, 4000.0]
	
    def tearDown(self):
        pass


        
    def roundTrip(self, dist, *args):

	if len(args) == 2:
	    x, y = args
	    p = afwGeom.Point2D(x, y)
	    pDist = dist.distort(p)
	    pp    = dist.undistort(pDist)

	    if self.prynt:
		print "p:     %.12f %.12f" % (p.getX(),     p.getY())
		print "pDist: %.12f %.12f" % (pDist.getX(), pDist.getY())
		print "pp:    %.12f %.12f" % (pp.getX(),    pp.getY())
		
	    self.assertAlmostEqual(pp.getX(), p.getX(), 6)
	    self.assertAlmostEqual(pp.getY(), p.getY(), 6)


	if len(args) == 3:
	    x, y, m = args
	    ixx, iyy, ixy = m

	    p = afwGeom.Point2D(x, y)
	    pDist = dist.distort(p)
	    m = cameraGeom.Moment(ixx, iyy, ixy)
	    mDist = dist.distort(p, m)
	    mm    = dist.undistort(pDist, mDist)
	    r0 = math.sqrt(x*x+y*y)

	    theta = math.atan2(y,x)
	    #dx = 0.001*math.cos(theta)
	    #dy = 0.001*math.sin(theta)
	    #dr = math.sqrt(dx*dx+dy*dy)
	    scale = 1.0000001
	    p2 = afwGeom.Point2D(scale*x, scale*y)
	    p2Dist = dist.distort(p2)

	    r1 = math.sqrt(pDist.getX()*pDist.getX() + pDist.getY()*pDist.getY())
	    r2 = math.sqrt(p2Dist.getX()*p2Dist.getX() + p2Dist.getY()*p2Dist.getY())
	    if r0 > 0:
		drTest = (r2 - r1)/((scale-1.0)*r0)
	    else:
		drTest = 0.0
	    
	    if hasattr(dist, 'transformR'):
		dr = dist.transformR(r0, dist.getDCoeffs())
		if r0 > 0:
		    ror = dist.transformR(r0, dist.getCoeffs())/r0
		else:
		    ror = 1.0
	    else:
		dr = 1.0
		ror = 1.0
	    
	    if self.prynt:
		print "r,dr:  %.12f,%.12f %.12f"       % (ror, dr, drTest)
		print "m:     %.12f %.12f %.12f" % (m.getIxx(), m.getIyy(), m.getIxy())
		print "mDist: %.12f %.12f %.12f" % (mDist.getIxx(), mDist.getIyy(), mDist.getIxy())
		print "mp:    %.12f %.12f %.12f" % (mm.getIxx(), mm.getIyy(), mm.getIxy())

		ixyTmp = m.getIxy() if m.getIxy() != 0.0 else 1.0
		print "err:   %.12f %.12f %.12f" % (
		    (m.getIxx()-mm.getIxx())/m.getIxx(),
		    (m.getIyy()-mm.getIyy())/m.getIyy(),
		    (m.getIxy()-mm.getIxy())/ixyTmp,
		    )
		
	    self.assertAlmostEqual(mm.getIxx(), m.getIxx(), 2)
	    self.assertAlmostEqual(mm.getIyy(), m.getIyy(), 2)
	    self.assertAlmostEqual(mm.getIxy(), m.getIxy(), 2)
	    


    def testRoundTrip(self):
	
	x, y = 1.0, 1.0

	# try NullDistortion
	ndist = cameraGeom.NullDistortion()
	self.roundTrip(ndist, x, y)
	
	# try RadialPolyDistortion
	rdist = cameraGeom.RadialPolyDistortion(self.coeffs)
	self.roundTrip(rdist, x, y)


    def testDistortionPointerInDetector(self):

	det = cameraGeom.Detector(cameraGeom.Id(1))

	# default to No distortion
	dist = det.getDistortion()
	x, y = 1.0, 1.0
	p = dist.distort(afwGeom.Point2D(x, y))

	if self.prynt:
	    print "%.12f %.12f" % (p.getX(), p.getY())
	
	self.assertEqual(p.getX(), x)
	self.assertEqual(p.getY(), y)


	# make sure we can set a radialpoly and round-trip it.
	det.setDistortion(cameraGeom.RadialPolyDistortion(self.coeffs))
	self.roundTrip(det.getDistortion(), x, y)


    def testMomentDistortion(self):

	moments = [
	    [1.0, 1.0, 0.0],
	    [1.0, 1.0, 0.1],
	    ]
	#self.xs = [5000.0]
	#self.ys = [0.0]
	
	nDist = cameraGeom.NullDistortion()
	rDist = cameraGeom.RadialPolyDistortion(self.coeffs)

	print rDist.getCoeffs()
	print rDist.getICoeffs()

	if False: #True:
	    import numpy
	    import matplotlib.figure as figure
	    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigCanvas
	    fig = figure.Figure()
	    canvas = FigCanvas(fig)
	    ax = fig.add_subplot(211)
	    dax = fig.add_subplot(212)
	    r = numpy.arange(0, 7500, 100)
	    c = rDist.getCoeffs()
	    c.reverse()
	    ic = rDist.getICoeffs()
	    ic.reverse()
	    dc = rDist.getDCoeffs()
	    dc.reverse()
	    idc = rDist.getIdCoeffs()
	    idc.reverse()
	    
	    rp  = numpy.polyval(c, r)
	    irp = numpy.polyval(ic, r)
	    r_round = numpy.polyval(ic, rp)
	    
	    drp = numpy.polyval(dc, r)
	    idrp = numpy.polyval(idc, r)
	    d_round = numpy.polyval(idc, rp)
				    
	    ax.plot(r, rp-r, '-b')
	    ax.plot(r, -(irp-r), '.b')
	    ax.semilogy(r, abs(r_round-r), 'r-')
	    ax.axhline(1.0)
	    ax.axhline(1.0e-3)
	    dax.plot(r, drp, '-g')
	    dax.plot(r, idrp, '.g')
	    dax.semilogy(r, numpy.abs(1.0-d_round*drp), 'r-')
	    dax.axhline(1.0e-3)

	    fig.savefig("foo.png")

	for moment in moments:
	    ixx, iyy, ixy = moment
	    for x in self.xs:
		for y in self.ys:
		    print x, y
		    self.roundTrip(nDist, x, y, [ixx, iyy, ixy])
		    self.roundTrip(rDist, x, y, [ixx, iyy, ixy])


	
        
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
