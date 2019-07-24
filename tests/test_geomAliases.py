#
# LSST Data Management System
# Copyright 2018 LSST Corporation.
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
import unittest

import lsst.utils.tests
import lsst.geom
import lsst.afw.geom as afwGeom


class GeomAliasesTestCase(lsst.utils.tests.TestCase):

    def testAngleAliases(self):
        with self.assertWarns(FutureWarning):
            self.assertEqual(afwGeom.Angle(1), lsst.geom.Angle(1))

        self.assertIs(afwGeom.radians, lsst.geom.radians)
        self.assertIs(afwGeom.degrees, lsst.geom.degrees)
        self.assertIs(afwGeom.hours, lsst.geom.hours)
        self.assertIs(afwGeom.arcminutes, lsst.geom.arcminutes)
        self.assertIs(afwGeom.arcseconds, lsst.geom.arcseconds)

        self.assertIs(afwGeom.PI, lsst.geom.PI)
        self.assertIs(afwGeom.TWOPI, lsst.geom.TWOPI)
        self.assertIs(afwGeom.HALFPI, lsst.geom.HALFPI)
        self.assertIs(afwGeom.ONE_OVER_PI, lsst.geom.ONE_OVER_PI)
        self.assertIs(afwGeom.SQRTPI, lsst.geom.SQRTPI)
        self.assertIs(afwGeom.INVSQRTPI, lsst.geom.INVSQRTPI)
        self.assertIs(afwGeom.ROOT2, lsst.geom.ROOT2)

        with self.assertWarns(FutureWarning):
            self.assertEqual(afwGeom.degToRad(1), lsst.geom.degToRad(1))
        with self.assertWarns(FutureWarning):
            self.assertEqual(afwGeom.radToDeg(1), lsst.geom.radToDeg(1))
        with self.assertWarns(FutureWarning):
            self.assertEqual(afwGeom.radToArcsec(1), lsst.geom.radToArcsec(1))
        with self.assertWarns(FutureWarning):
            self.assertEqual(afwGeom.radToMas(1), lsst.geom.radToMas(1))
        with self.assertWarns(FutureWarning):
            self.assertEqual(afwGeom.arcsecToRad(1), lsst.geom.arcsecToRad(1))
        with self.assertWarns(FutureWarning):
            self.assertEqual(afwGeom.masToRad(1), lsst.geom.masToRad(1))

    def testCoordAliases(self):
        with self.assertWarns(FutureWarning):
            self.assertIsInstance(afwGeom.ExtentI(), lsst.geom.ExtentI)
        with self.assertWarns(FutureWarning):
            self.assertIsInstance(afwGeom.Extent2I(), lsst.geom.Extent2I)
        with self.assertWarns(FutureWarning):
            self.assertIsInstance(afwGeom.Extent3I(), lsst.geom.Extent3I)

        with self.assertWarns(FutureWarning):
            self.assertIsInstance(afwGeom.ExtentD(), lsst.geom.ExtentD)
        with self.assertWarns(FutureWarning):
            self.assertIsInstance(afwGeom.Extent2D(), lsst.geom.Extent2D)
        with self.assertWarns(FutureWarning):
            self.assertIsInstance(afwGeom.Extent3D(), lsst.geom.Extent3D)

        with self.assertWarns(FutureWarning):
            self.assertIsInstance(afwGeom.PointI(), lsst.geom.PointI)
        with self.assertWarns(FutureWarning):
            self.assertIsInstance(afwGeom.Point2I(), lsst.geom.Point2I)
        with self.assertWarns(FutureWarning):
            self.assertIsInstance(afwGeom.Point3I(), lsst.geom.Point3I)

        with self.assertWarns(FutureWarning):
            self.assertIsInstance(afwGeom.PointD(), lsst.geom.PointD)
        with self.assertWarns(FutureWarning):
            self.assertIsInstance(afwGeom.Point2D(), lsst.geom.Point2D)
        with self.assertWarns(FutureWarning):
            self.assertIsInstance(afwGeom.Point3D(), lsst.geom.Point3D)

    def testOtherAliases(self):
        with self.assertWarns(FutureWarning):
            self.assertIsInstance(afwGeom.BoxI(), lsst.geom.BoxI)
        with self.assertWarns(FutureWarning):
            self.assertIsInstance(afwGeom.BoxI(), lsst.geom.Box2I)
        with self.assertWarns(FutureWarning):
            self.assertIsInstance(afwGeom.BoxD(), lsst.geom.BoxD)
        with self.assertWarns(FutureWarning):
            self.assertIsInstance(afwGeom.BoxD(), lsst.geom.Box2D)

        with self.assertWarns(FutureWarning):
            self.assertIsInstance(afwGeom.SpherePoint(), lsst.geom.SpherePoint)

        with self.assertWarns(FutureWarning):
            self.assertIsInstance(afwGeom.AffineTransform(), lsst.geom.AffineTransform)
        with self.assertWarns(FutureWarning):
            self.assertIsInstance(afwGeom.LinearTransform(), lsst.geom.LinearTransform)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
