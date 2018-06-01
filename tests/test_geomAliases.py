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
        self.assertIs(afwGeom.Angle, lsst.geom.Angle)
        self.assertIs(afwGeom.AngleUnit, lsst.geom.AngleUnit)

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

        self.assertIs(afwGeom.degToRad, lsst.geom.degToRad)
        self.assertIs(afwGeom.radToDeg, lsst.geom.radToDeg)
        self.assertIs(afwGeom.radToArcsec, lsst.geom.radToArcsec)
        self.assertIs(afwGeom.radToMas, lsst.geom.radToMas)
        self.assertIs(afwGeom.arcsecToRad, lsst.geom.arcsecToRad)
        self.assertIs(afwGeom.masToRad, lsst.geom.masToRad)

    def testCoordAliases(self):
        self.assertIs(afwGeom.ExtentI, lsst.geom.ExtentI)
        self.assertIs(afwGeom.Extent2I, lsst.geom.Extent2I)
        self.assertIs(afwGeom.Extent3I, lsst.geom.Extent3I)

        self.assertIs(afwGeom.ExtentD, lsst.geom.ExtentD)
        self.assertIs(afwGeom.Extent2D, lsst.geom.Extent2D)
        self.assertIs(afwGeom.Extent3D, lsst.geom.Extent3D)

        self.assertIs(afwGeom.PointI, lsst.geom.PointI)
        self.assertIs(afwGeom.Point2I, lsst.geom.Point2I)
        self.assertIs(afwGeom.Point3I, lsst.geom.Point3I)

        self.assertIs(afwGeom.PointD, lsst.geom.PointD)
        self.assertIs(afwGeom.Point2D, lsst.geom.Point2D)
        self.assertIs(afwGeom.Point3D, lsst.geom.Point3D)

    def testOtherAliases(self):
        self.assertIs(afwGeom.BoxI, lsst.geom.BoxI)
        self.assertIs(afwGeom.BoxI, lsst.geom.Box2I)
        self.assertIs(afwGeom.BoxD, lsst.geom.BoxD)
        self.assertIs(afwGeom.BoxD, lsst.geom.Box2D)

        self.assertIs(afwGeom.SpherePoint, lsst.geom.SpherePoint)

        self.assertIs(afwGeom.AffineTransform, lsst.geom.AffineTransform)
        self.assertIs(afwGeom.LinearTransform, lsst.geom.LinearTransform)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
