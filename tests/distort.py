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

	    if self.prynt:
		print "m:     %.12f %.12f %.12f" % (m.getIxx(), m.getIyy(), m.getIxy())
		print "mDist: %.12f %.12f %.12f" % (mDist.getIxx(), mDist.getIyy(), mDist.getIxy())
		print "mp:    %.12f %.12f %.12f" % (mm.getIxx(), mm.getIyy(), mm.getIxy())
		
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

	ixx, iyy, ixy = 1.0, 1.0, 0.0
	nDist = cameraGeom.NullDistortion()
	rDist = cameraGeom.RadialPolyDistortion(self.coeffs)

	print rDist.getCoeffs()
	print rDist.getICoeffs()

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
	drp = numpy.polyval(dc, r)
	idrp = numpy.polyval(idc, r)
	
	ax.plot(r, rp-r, '-b')
	ax.plot(r, irp-r, '.b')
	ax.plot(r, (rp-r)+(irp-r), 'r-')
	dax.plot(r, drp, '-g')
	dax.plot(r, idrp, '.g')
	dax.plot(r, drp*idrp, 'r-')
	
	fig.savefig("foo.png")
	
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
