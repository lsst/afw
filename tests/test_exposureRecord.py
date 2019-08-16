# This file is part of afw.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
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
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import unittest

import lsst.afw.geom
import lsst.afw.table
import lsst.geom
import lsst.utils.tests


class ExposureRecordTestCase(unittest.TestCase):
    def setUp(self):
        # Construct a trivial WCS:
        #
        # - reference pixel at position (10, 10) corresponds to a sky origin of (0, 0).
        # - scale 1 arcsec per pixel.
        self.crval = lsst.geom.SpherePoint(0, 0, lsst.geom.degrees)
        self.crpix = lsst.geom.Point2D(10, 10)
        cdMatrix = lsst.afw.geom.makeCdMatrix(1.0*lsst.geom.arcseconds)
        self.wcs = lsst.afw.geom.makeSkyWcs(crpix=self.crpix, crval=self.crval, cdMatrix=cdMatrix)

        # It is not possible to directly construct an ExposureRecord.
        # Instead, we construct an ExposureCatalog, then create a record in it.
        # Note that the record's bounding box is centred on the WCS reference pixel.
        catalog = lsst.afw.table.ExposureCatalog(lsst.afw.table.ExposureTable.makeMinimalSchema())
        self.record = catalog.addNew()
        self.record.setBBox(lsst.geom.Box2I(lsst.geom.Point2I(0, 0), lsst.geom.Point2I(20, 20)))
        self.record.setWcs(self.wcs)

        # Add a valid polygon to the record.
        # Note that the polygon covers one half (along the x axis) of the record:
        # 0 <= x < 10 is valid, 10 <= x < 20 is invalid.
        polygon = lsst.afw.geom.Polygon([lsst.geom.Point2D(*pt)
                                         for pt in [(0, 0), (0, 20), (10, 20), (10, 0)]])
        self.record.setValidPolygon(polygon)

    def testContainsSpherePoint(self):
        """Test ExposureRecord.contains() with a SpherePoint.
        """
        # By construction, the CRVAL is contained within the record.
        self.assertTrue(self.record.contains(self.crval))

        # Rotate the test point about a vertical axis.
        axis = lsst.geom.SpherePoint(0, 90, lsst.geom.degrees)

        # A 180 degree rotation is far outside the record.
        self.assertFalse(self.record.contains(self.crval.rotated(axis, 180*lsst.geom.degrees)))

        # A 1 arcsecond rotation in either direction is within the record.
        self.assertTrue(self.record.contains(self.crval.rotated(axis, 1.0 * lsst.geom.arcseconds)))
        self.assertTrue(self.record.contains(self.crval.rotated(axis, -1.0 * lsst.geom.arcseconds)))

        # A 1 arcsecond positive rotation is within the valid polygon.
        self.assertTrue(self.record.contains(self.crval.rotated(axis, 1.0 * lsst.geom.arcseconds),
                                             includeValidPolygon=True))

        # A 1 arcsecond negative rotation is outside the valid polygon.
        self.assertFalse(self.record.contains(self.crval.rotated(axis, -1.0 * lsst.geom.arcseconds),
                                              includeValidPolygon=True))

    def testContainsPoint(self):
        """Test ExposureRecord.contains() with a Point and a WCS.
        """
        # By construction, the reference pixel is contained within the record.
        self.assertTrue(self.record.contains(self.crpix, self.wcs))

        # Some points are clearly outside the record.
        self.assertFalse(self.record.contains(lsst.geom.Point2D(99, 99), self.wcs))

        # A small offset from the center should be within the record.
        points = [lsst.geom.Point2D(*pt) for pt in [(9, 10), (11, 10), (10, 9), (10, 11)]]
        for point in points:
            self.assertTrue(self.record.contains(point, self.wcs))

        # But only with x < 10 if we use the valid polygon.
        for point in points:
            if point.getX() < 10:
                self.assertTrue(self.record.contains(point, self.wcs, includeValidPolygon=True))
            else:
                self.assertFalse(self.record.contains(point, self.wcs, includeValidPolygon=True))


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
