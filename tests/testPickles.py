#pybind11##!/usr/bin/env python
#pybind11#from __future__ import absolute_import, division
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
#pybind11#
#pybind11#"""
#pybind11#Tests for pickles of some afw types
#pybind11#"""
#pybind11#
#pybind11#import unittest
#pybind11#import pickle
#pybind11#
#pybind11#import lsst.daf.base as dafBase
#pybind11#import lsst.utils.tests
#pybind11#import lsst.afw.image as afwImage
#pybind11#import lsst.afw.geom as afwGeom
#pybind11#import lsst.afw.geom.ellipses as geomEllip
#pybind11#import lsst.afw.coord as afwCoord
#pybind11#
#pybind11#
#pybind11#class PickleBase:
#pybind11#    """A test case for pickles"""
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        raise NotImplementedError("Need to inherit and create the 'data' element.")
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        del self.data
#pybind11#
#pybind11#    def assertPickled(self, new):
#pybind11#        """Assert that the pickled data is the same as the original
#pybind11#
#pybind11#        Subclasses should override this method if the particular data
#pybind11#        doesn't support the == operator.
#pybind11#        """
#pybind11#        self.assertEqual(new, self.data)
#pybind11#
#pybind11#    def testPickle(self):
#pybind11#        """Test round-trip pickle"""
#pybind11#        pickled = pickle.dumps(self.data)
#pybind11#        newData = pickle.loads(pickled)
#pybind11#        self.assertPickled(newData)
#pybind11#
#pybind11#
#pybind11#class AngleTestCase(PickleBase,unittest.TestCase):
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        self.data = 1.0*afwGeom.degrees
#pybind11#
#pybind11#
#pybind11#class CoordTestCase(PickleBase,unittest.TestCase):
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        ra = 10.0*afwGeom.degrees
#pybind11#        dec = 1.0*afwGeom.degrees
#pybind11#        epoch = 2000.0
#pybind11#        self.data = [afwCoord.Coord(ra, dec, epoch),
#pybind11#                     afwCoord.Fk5Coord(ra, dec, epoch),
#pybind11#                     afwCoord.IcrsCoord(ra, dec),
#pybind11#                     afwCoord.GalacticCoord(ra, dec),
#pybind11#                     afwCoord.EclipticCoord(ra, dec),
#pybind11#                     # TopocentricCoord is not currently picklable
#pybind11#                     ]
#pybind11#
#pybind11#
#pybind11#class QuadrupoleTestCase(PickleBase,unittest.TestCase):
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        ixx, iyy, ixy = 1.0, 1.0, 0.0
#pybind11#        self.data = geomEllip.Quadrupole(ixx, iyy, ixy)
#pybind11#
#pybind11#
#pybind11#class AxesTestCase(PickleBase,unittest.TestCase):
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        a, b, theta = 1.0, 1.0, 0.0
#pybind11#        self.data = geomEllip.Axes(a, b, theta)
#pybind11#
#pybind11#
#pybind11#class Point2DTestCase(PickleBase,unittest.TestCase):
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        x, y = 1.0, 1.0
#pybind11#        self.data = afwGeom.Point2D(x, y)
#pybind11#
#pybind11#
#pybind11#class Point2ITestCase(PickleBase,unittest.TestCase):
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        x, y = 1, 1
#pybind11#        self.data = afwGeom.Point2I(x, y)
#pybind11#
#pybind11#
#pybind11#class Point3DTestCase(PickleBase,unittest.TestCase):
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        x, y, z = 1.0, 1.0, 1.0
#pybind11#        self.data = afwGeom.Point3D(x, y, z)
#pybind11#
#pybind11#
#pybind11#class Point3ITestCase(PickleBase,unittest.TestCase):
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        x, y, z = 1, 1, 1
#pybind11#        self.data = afwGeom.Point3I(x, y, z)
#pybind11#
#pybind11#
#pybind11#class Extent2DTestCase(PickleBase,unittest.TestCase):
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        x, y = 1.0, 1.0
#pybind11#        self.data = afwGeom.Extent2D(x, y)
#pybind11#
#pybind11#
#pybind11#class Extent3DTestCase(PickleBase,unittest.TestCase):
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        x, y, z = 1, 1, 1
#pybind11#        self.data = afwGeom.Extent3D(x, y, z)
#pybind11#
#pybind11#
#pybind11#class Extent2ITestCase(PickleBase,unittest.TestCase):
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        x, y = 1, 1
#pybind11#        self.data = afwGeom.Extent2I(x, y)
#pybind11#
#pybind11#
#pybind11#class Extent3ITestCase(PickleBase,unittest.TestCase):
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        x, y, z = 1, 1, 1
#pybind11#        self.data = afwGeom.Extent3I(x, y, z)
#pybind11#
#pybind11#
#pybind11#class Box2DTestCase(PickleBase,unittest.TestCase):
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        p, e = afwGeom.Point2D(1.0, 1.0), afwGeom.Extent2D(0.5, 0.5)
#pybind11#        self.data = afwGeom.Box2D(p, e)
#pybind11#
#pybind11#
#pybind11#class Box2ITestCase(PickleBase,unittest.TestCase):
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        p, e = afwGeom.Point2I(1, 2), afwGeom.Extent2I(1, 1)
#pybind11#        self.data = afwGeom.Box2I(p, e)
#pybind11#
#pybind11#
#pybind11#class AffineTransformTestCase(PickleBase,unittest.TestCase):
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        scale = 2.2
#pybind11#        linear = afwGeom.LinearTransform().makeScaling(scale)
#pybind11#        dx, dy = 1.1, 3.3
#pybind11#        trans = afwGeom.Extent2D(dx, dy)
#pybind11#        self.data = afwGeom.AffineTransform(linear, trans)
#pybind11#
#pybind11#    def assertPickled(self, new):
#pybind11#        self.assertListEqual(new.getMatrix().flatten().tolist(), self.data.getMatrix().flatten().tolist())
#pybind11#
#pybind11#
#pybind11#class LinearTransformTestCase(PickleBase,unittest.TestCase):
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        scale = 2.0
#pybind11#        self.data = afwGeom.LinearTransform().makeScaling(scale)
#pybind11#
#pybind11#    def assertPickled(self, new):
#pybind11#        self.assertListEqual(new.getMatrix().flatten().tolist(), self.data.getMatrix().flatten().tolist())
#pybind11#
#pybind11#
#pybind11#class WcsPickleBase(PickleBase,unittest.TestCase):
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        hdr = dafBase.PropertyList()
#pybind11#        hdr.add("NAXIS", 2)
#pybind11#        hdr.add("EQUINOX", 2000.0000000000)
#pybind11#        hdr.add("RADESYS", "FK5")
#pybind11#        hdr.add("CRPIX1", 947.04531175212)
#pybind11#        hdr.add("CRPIX2", -305.70042176782)
#pybind11#        hdr.add("CD1_1", -5.6081060666063e-05)
#pybind11#        hdr.add("CD1_2", 1.1941349711530e-10)
#pybind11#        hdr.add("CD2_1", 1.1938226362497e-10)
#pybind11#        hdr.add("CD2_2", 5.6066392248206e-05)
#pybind11#        hdr.add("CRVAL1", 5.5350859380564)
#pybind11#        hdr.add("CRVAL2", -0.57805534748292)
#pybind11#        hdr.add("CUNIT1", "deg")
#pybind11#        hdr.add("CUNIT2", "deg")
#pybind11#        hdr.add("CTYPE1", "RA---TAN")
#pybind11#        hdr.add("CTYPE2", "DEC--TAN")
#pybind11#        self.data = afwImage.makeWcs(hdr)
#pybind11#
#pybind11#
#pybind11#class TanWcsPickleBase(PickleBase,unittest.TestCase):
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        hdr = dafBase.PropertyList()
#pybind11#        hdr.add("NAXIS", 2)
#pybind11#        hdr.add("EQUINOX", 2000.0000000000)
#pybind11#        hdr.add("RADESYS", "FK5")
#pybind11#        hdr.add("CRPIX1", 947.04531175212)
#pybind11#        hdr.add("CRPIX2", -305.70042176782)
#pybind11#        hdr.add("CD1_1", -5.6081060666063e-05)
#pybind11#        hdr.add("CD1_2", 1.1941349711530e-10)
#pybind11#        hdr.add("CD2_1", 1.1938226362497e-10)
#pybind11#        hdr.add("CD2_2", 5.6066392248206e-05)
#pybind11#        hdr.add("CRVAL1", 5.5350859380564)
#pybind11#        hdr.add("CRVAL2", -0.57805534748292)
#pybind11#        hdr.add("CUNIT1", "deg")
#pybind11#        hdr.add("CUNIT2", "deg")
#pybind11#        hdr.add("A_ORDER", 3)
#pybind11#        hdr.add("A_0_0", -3.4299726900155e-05)
#pybind11#        hdr.add("A_0_2", 2.9999243742039e-08)
#pybind11#        hdr.add("A_0_3", 5.3160367322875e-12)
#pybind11#        hdr.add("A_1_0", -1.1102230246252e-16)
#pybind11#        hdr.add("A_1_1", 1.7804837804549e-07)
#pybind11#        hdr.add("A_1_2", -3.9117665277930e-10)
#pybind11#        hdr.add("A_2_0", 1.2614116305773e-07)
#pybind11#        hdr.add("A_2_1", 2.4753748298399e-12)
#pybind11#        hdr.add("A_3_0", -4.0559790823371e-10)
#pybind11#        hdr.add("B_ORDER", 3)
#pybind11#        hdr.add("B_0_0", -0.00040333633853922)
#pybind11#        hdr.add("B_0_2", 2.7329405108287e-07)
#pybind11#        hdr.add("B_0_3", -4.1945333823804e-10)
#pybind11#        hdr.add("B_1_1", 1.0211300606274e-07)
#pybind11#        hdr.add("B_1_2", -1.1907781112538e-12)
#pybind11#        hdr.add("B_2_0", 7.1256679698479e-08)
#pybind11#        hdr.add("B_2_1", -4.0026664120969e-10)
#pybind11#        hdr.add("B_3_0", 7.2509034631981e-14)
#pybind11#        hdr.add("AP_ORDER", 5)
#pybind11#        hdr.add("AP_0_0", 0.065169424373537)
#pybind11#        hdr.add("AP_0_1", 3.5323035231808e-05)
#pybind11#        hdr.add("AP_0_2", -2.4878457741060e-08)
#pybind11#        hdr.add("AP_0_3", -1.4288745247360e-11)
#pybind11#        hdr.add("AP_0_4", -2.0000000098183)
#pybind11#        hdr.add("AP_0_5", 4.3337569354109e-19)
#pybind11#        hdr.add("AP_1_0", 1.9993638555698)
#pybind11#        hdr.add("AP_1_1", -2.0722860000493e-07)
#pybind11#        hdr.add("AP_1_2", 4.7562056847339e-10)
#pybind11#        hdr.add("AP_1_3", -8.5172068319818e-06)
#pybind11#        hdr.add("AP_1_4", -1.3242986537057e-18)
#pybind11#        hdr.add("AP_2_0", -1.4594781790233e-07)
#pybind11#        hdr.add("AP_2_1", -2.9254828606617e-12)
#pybind11#        hdr.add("AP_2_2", -2.7203380713516e-11)
#pybind11#        hdr.add("AP_2_3", 1.5030517486646e-19)
#pybind11#        hdr.add("AP_3_0", 4.7856034999197e-10)
#pybind11#        hdr.add("AP_3_1", 1.5571061278960e-15)
#pybind11#        hdr.add("AP_3_2", -3.2422164667295e-18)
#pybind11#        hdr.add("AP_4_0", 5.8904402441647e-16)
#pybind11#        hdr.add("AP_4_1", -4.5488928339401e-20)
#pybind11#        hdr.add("AP_5_0", -1.3198044795585e-18)
#pybind11#        hdr.add("BP_ORDER", 5)
#pybind11#        hdr.add("BP_0_0", 0.00025729974056661)
#pybind11#        hdr.add("BP_0_1", -0.00060857907313083)
#pybind11#        hdr.add("BP_0_2", -3.1283728005742e-07)
#pybind11#        hdr.add("BP_0_3", 5.0413932972962e-10)
#pybind11#        hdr.add("BP_0_4", -0.0046142128142681)
#pybind11#        hdr.add("BP_0_5", -2.2359607268985e-18)
#pybind11#        hdr.add("BP_1_0", 0.0046783112625990)
#pybind11#        hdr.add("BP_1_1", -1.2304042740813e-07)
#pybind11#        hdr.add("BP_1_2", -2.3756827881344e-12)
#pybind11#        hdr.add("BP_1_3", -3.9300202582816e-08)
#pybind11#        hdr.add("BP_1_4", -9.7385290942256e-21)
#pybind11#        hdr.add("BP_2_0", -6.5238116398890e-08)
#pybind11#        hdr.add("BP_2_1", 4.7855579009100e-10)
#pybind11#        hdr.add("BP_2_2", -1.2297758131839e-13)
#pybind11#        hdr.add("BP_2_3", -3.0849793267035e-18)
#pybind11#        hdr.add("BP_3_0", -9.3923321275113e-12)
#pybind11#        hdr.add("BP_3_1", -1.3193479628568e-17)
#pybind11#        hdr.add("BP_3_2", 2.1762350028059e-19)
#pybind11#        hdr.add("BP_4_0", -5.9687252632035e-16)
#pybind11#        hdr.add("BP_4_1", -1.4096893423344e-18)
#pybind11#        hdr.add("BP_5_0", 2.8085458107813e-19)
#pybind11#        hdr.add("CTYPE1", "RA---TAN-SIP")
#pybind11#        hdr.add("CTYPE2", "DEC--TAN-SIP")
#pybind11#        self.data = afwImage.makeWcs(hdr)
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#
#pybind11#
#pybind11#class TestMemory(lsst.utils.tests.MemoryTestCase):
#pybind11#    pass
#pybind11#
#pybind11#def setup_module(module):
#pybind11#    lsst.utils.tests.init()
#pybind11#
#pybind11#if __name__ == "__main__":
#pybind11#    lsst.utils.tests.init()
#pybind11#    unittest.main()
