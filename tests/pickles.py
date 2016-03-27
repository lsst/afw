#!/usr/bin/env python2
from __future__ import absolute_import, division

#
# LSST Data Management System
# See the COPYRIGHT and LICENSE files in the top-level directory of this
# package for notices and licensing terms.
#

"""
Tests for pickles of some afw types
"""

import unittest
import pickle

import lsst.daf.base as dafBase
import lsst.utils.tests as utilsTests
import lsst.afw.image as afwImage
import lsst.afw.geom  as afwGeom
import lsst.afw.geom.ellipses as geomEllip
import lsst.afw.coord as afwCoord

class PickleTestCase(unittest.TestCase):
    """A test case for pickles"""

    def setUp(self):
        raise NotImplementedError("Need to inherit and create the 'data' element.")

    def tearDown(self):
        del self.data

    def assertPickled(self, new):
        """Assert that the pickled data is the same as the original

        Subclasses should override this method if the particular data
        doesn't support the == operator.
        """
        self.assertTrue(new == self.data)

    def testPickle(self):
        """Test round-trip pickle"""
        pickled = pickle.dumps(self.data)
        newData = pickle.loads(pickled)
        self.assertPickled(newData)


class AngleTestCase(PickleTestCase):
    def setUp(self):
        self.data = 1.0*afwGeom.degrees

class CoordTestCase(PickleTestCase):
    def setUp(self):
        ra = 10.0*afwGeom.degrees
        dec = 1.0*afwGeom.degrees
        epoch = 2000.0
        self.data = [afwCoord.Coord(ra, dec, epoch),
                     afwCoord.Fk5Coord(ra, dec, epoch),
                     afwCoord.IcrsCoord(ra, dec),
                     afwCoord.GalacticCoord(ra, dec),
                     afwCoord.EclipticCoord(ra, dec),
                     # TopocentricCoord is not currently picklable
                     ]

class QuadrupoleTestCase(PickleTestCase):
    def setUp(self):
        ixx, iyy, ixy = 1.0, 1.0, 0.0
        self.data = geomEllip.Quadrupole(ixx, iyy, ixy)

class AxesTestCase(PickleTestCase):
    def setUp(self):
        a, b, theta = 1.0, 1.0, 0.0
        self.data = geomEllip.Axes(a, b, theta)
        
class Point2DTestCase(PickleTestCase):
    def setUp(self):
        x, y = 1.0, 1.0
        self.data = afwGeom.Point2D(x, y)
class Point2ITestCase(PickleTestCase):
    def setUp(self):
        x, y = 1, 1
        self.data = afwGeom.Point2I(x, y)
class Point3DTestCase(PickleTestCase):
    def setUp(self):
        x, y, z = 1.0, 1.0, 1.0
        self.data = afwGeom.Point3D(x, y, z)
class Point3ITestCase(PickleTestCase):
    def setUp(self):
        x, y, z = 1,1,1
        self.data = afwGeom.Point3I(x, y, z)
        
class Extent2DTestCase(PickleTestCase):
    def setUp(self):
        x, y = 1.0, 1.0
        self.data = afwGeom.Extent2D(x, y)
class Extent3DTestCase(PickleTestCase):
    def setUp(self):
        x, y, z = 1,1,1
        self.data = afwGeom.Extent3D(x, y, z)
class Extent2ITestCase(PickleTestCase):
    def setUp(self):
        x, y = 1, 1
        self.data = afwGeom.Extent2I(x, y)
class Extent3ITestCase(PickleTestCase):
    def setUp(self):
        x, y, z = 1,1,1
        self.data = afwGeom.Extent3I(x, y, z)
        
class Box2DTestCase(PickleTestCase):    
    def setUp(self):
        p, e = afwGeom.Point2D(1.0, 1.0), afwGeom.Extent2D(0.5, 0.5)
        self.data = afwGeom.Box2D(p, e)
class Box2ITestCase(PickleTestCase):    
    def setUp(self):
        p, e = afwGeom.Point2I(1, 2), afwGeom.Extent2I(1, 1)
        self.data = afwGeom.Box2I(p, e)
        
class AffineTransformTestCase(PickleTestCase):
    def setUp(self):
        scale = 2.2
        linear = afwGeom.LinearTransform().makeScaling(scale)
        dx, dy = 1.1, 3.3
        trans = afwGeom.Extent2D(dx, dy)
        self.data = afwGeom.AffineTransform(linear, trans)

    def assertPickled(self, new):
        self.assertTrue((new.getMatrix() == self.data.getMatrix()).all())
        
class LinearTransformTestCase(PickleTestCase):
    def setUp(self):
        scale = 2.0
        self.data = afwGeom.LinearTransform().makeScaling(scale)

    def assertPickled(self, new):
        self.assertTrue((new.getMatrix() == self.data.getMatrix()).all())

class WcsPickleTestCase(PickleTestCase):
    def setUp(self):
        hdr = dafBase.PropertyList()
        hdr.add("NAXIS", 2)
        hdr.add("EQUINOX", 2000.0000000000)
        hdr.add("RADESYS", "FK5")
        hdr.add("CRPIX1", 947.04531175212)
        hdr.add("CRPIX2", -305.70042176782)
        hdr.add("CD1_1", -5.6081060666063e-05)
        hdr.add("CD1_2", 1.1941349711530e-10)
        hdr.add("CD2_1", 1.1938226362497e-10)
        hdr.add("CD2_2", 5.6066392248206e-05)
        hdr.add("CRVAL1", 5.5350859380564)
        hdr.add("CRVAL2", -0.57805534748292)
        hdr.add("CUNIT1", "deg")
        hdr.add("CUNIT2", "deg")
        hdr.add("CTYPE1", "RA---TAN")
        hdr.add("CTYPE2", "DEC--TAN")
        self.data = afwImage.makeWcs(hdr)

class TanWcsPickleTestCase(PickleTestCase):
    def setUp(self):
        hdr = dafBase.PropertyList()
        hdr.add("NAXIS", 2)
        hdr.add("EQUINOX", 2000.0000000000)
        hdr.add("RADESYS", "FK5")
        hdr.add("CRPIX1", 947.04531175212)
        hdr.add("CRPIX2", -305.70042176782)
        hdr.add("CD1_1", -5.6081060666063e-05)
        hdr.add("CD1_2", 1.1941349711530e-10)
        hdr.add("CD2_1", 1.1938226362497e-10)
        hdr.add("CD2_2", 5.6066392248206e-05)
        hdr.add("CRVAL1", 5.5350859380564)
        hdr.add("CRVAL2", -0.57805534748292)
        hdr.add("CUNIT1", "deg")
        hdr.add("CUNIT2", "deg")
        hdr.add("A_ORDER", 3)
        hdr.add("A_0_0", -3.4299726900155e-05)
        hdr.add("A_0_2", 2.9999243742039e-08)
        hdr.add("A_0_3", 5.3160367322875e-12)
        hdr.add("A_1_0", -1.1102230246252e-16)
        hdr.add("A_1_1", 1.7804837804549e-07)
        hdr.add("A_1_2", -3.9117665277930e-10)
        hdr.add("A_2_0", 1.2614116305773e-07)
        hdr.add("A_2_1", 2.4753748298399e-12)
        hdr.add("A_3_0", -4.0559790823371e-10)
        hdr.add("B_ORDER", 3)
        hdr.add("B_0_0", -0.00040333633853922)
        hdr.add("B_0_2", 2.7329405108287e-07)
        hdr.add("B_0_3", -4.1945333823804e-10)
        hdr.add("B_1_1", 1.0211300606274e-07)
        hdr.add("B_1_2", -1.1907781112538e-12)
        hdr.add("B_2_0", 7.1256679698479e-08)
        hdr.add("B_2_1", -4.0026664120969e-10)
        hdr.add("B_3_0", 7.2509034631981e-14)
        hdr.add("AP_ORDER", 5)
        hdr.add("AP_0_0", 0.065169424373537)
        hdr.add("AP_0_1", 3.5323035231808e-05)
        hdr.add("AP_0_2", -2.4878457741060e-08)
        hdr.add("AP_0_3", -1.4288745247360e-11)
        hdr.add("AP_0_4", -2.0000000098183)
        hdr.add("AP_0_5", 4.3337569354109e-19)
        hdr.add("AP_1_0", 1.9993638555698)
        hdr.add("AP_1_1", -2.0722860000493e-07)
        hdr.add("AP_1_2", 4.7562056847339e-10)
        hdr.add("AP_1_3", -8.5172068319818e-06)
        hdr.add("AP_1_4", -1.3242986537057e-18)
        hdr.add("AP_2_0", -1.4594781790233e-07)
        hdr.add("AP_2_1", -2.9254828606617e-12)
        hdr.add("AP_2_2", -2.7203380713516e-11)
        hdr.add("AP_2_3", 1.5030517486646e-19)
        hdr.add("AP_3_0", 4.7856034999197e-10)
        hdr.add("AP_3_1", 1.5571061278960e-15)
        hdr.add("AP_3_2", -3.2422164667295e-18)
        hdr.add("AP_4_0", 5.8904402441647e-16)
        hdr.add("AP_4_1", -4.5488928339401e-20)
        hdr.add("AP_5_0", -1.3198044795585e-18)
        hdr.add("BP_ORDER", 5)
        hdr.add("BP_0_0", 0.00025729974056661)
        hdr.add("BP_0_1", -0.00060857907313083)
        hdr.add("BP_0_2", -3.1283728005742e-07)
        hdr.add("BP_0_3", 5.0413932972962e-10)
        hdr.add("BP_0_4", -0.0046142128142681)
        hdr.add("BP_0_5", -2.2359607268985e-18)
        hdr.add("BP_1_0", 0.0046783112625990)
        hdr.add("BP_1_1", -1.2304042740813e-07)
        hdr.add("BP_1_2", -2.3756827881344e-12)
        hdr.add("BP_1_3", -3.9300202582816e-08)
        hdr.add("BP_1_4", -9.7385290942256e-21)
        hdr.add("BP_2_0", -6.5238116398890e-08)
        hdr.add("BP_2_1", 4.7855579009100e-10)
        hdr.add("BP_2_2", -1.2297758131839e-13)
        hdr.add("BP_2_3", -3.0849793267035e-18)
        hdr.add("BP_3_0", -9.3923321275113e-12)
        hdr.add("BP_3_1", -1.3193479628568e-17)
        hdr.add("BP_3_2", 2.1762350028059e-19)
        hdr.add("BP_4_0", -5.9687252632035e-16)
        hdr.add("BP_4_1", -1.4096893423344e-18)
        hdr.add("BP_5_0", 2.8085458107813e-19)
        hdr.add("CTYPE1", "RA---TAN-SIP")
        hdr.add("CTYPE2", "DEC--TAN-SIP")
        self.data = afwImage.makeWcs(hdr)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(AngleTestCase)
    suites += unittest.makeSuite(CoordTestCase)
    suites += unittest.makeSuite(QuadrupoleTestCase)
    suites += unittest.makeSuite(AxesTestCase)
    suites += unittest.makeSuite(Point2DTestCase)
    suites += unittest.makeSuite(Point3DTestCase)
    suites += unittest.makeSuite(Point2ITestCase)
    suites += unittest.makeSuite(Point3ITestCase)
    suites += unittest.makeSuite(Extent2DTestCase)
    suites += unittest.makeSuite(Extent3DTestCase)
    suites += unittest.makeSuite(Box2DTestCase)
    suites += unittest.makeSuite(Box2ITestCase)
    suites += unittest.makeSuite(AffineTransformTestCase)
    suites += unittest.makeSuite(LinearTransformTestCase)
    suites += unittest.makeSuite(WcsPickleTestCase)
    suites += unittest.makeSuite(TanWcsPickleTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)

    return unittest.TestSuite(suites)

def run(exit=False):
    """Run the tests"""
    utilsTests.run(suite(), exit)

if __name__ == "__main__":
    run(True)
