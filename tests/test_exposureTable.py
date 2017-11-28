#
# LSST Data Management System
# Copyright 2008-2016 LSST Corporation.
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
Tests for lsst.afw.table.ExposureTable

Run with:
   python testExposureTable.py
or
   python
   >>> import testExposureTable; testExposureTable.run()
"""

from __future__ import absolute_import, division, print_function
import os.path
import unittest

import numpy as np

import lsst.utils.tests
import lsst.pex.exceptions
from lsst.daf.base import DateTime, PropertySet
import lsst.afw.table
from lsst.afw.geom import arcseconds, degrees, radians, Point2D, Extent2D, Box2D, makeSkyWcs, Polygon
import lsst.afw.coord
import lsst.afw.image
import lsst.afw.detection
from testTableArchivesLib import DummyPsf

try:
    type(display)
except NameError:
    display = False


class ExposureTableTestCase(lsst.utils.tests.TestCase):

    @staticmethod
    def createWcs():
        metadata = PropertySet()
        metadata.set("SIMPLE", "T")
        metadata.set("BITPIX", -32)
        metadata.set("NAXIS", 2)
        metadata.set("NAXIS1", 1024)
        metadata.set("NAXIS2", 1153)
        metadata.set("RADECSYS", 'FK5')
        metadata.set("EQUINOX", 2000.)
        metadata.setDouble("CRVAL1", 215.604025685476)
        metadata.setDouble("CRVAL2", 53.1595451514076)
        metadata.setDouble("CRPIX1", 1109.99981456774)
        metadata.setDouble("CRPIX2", 560.018167811613)
        metadata.set("CTYPE1", 'RA---SIN')
        metadata.set("CTYPE2", 'DEC--SIN')
        metadata.setDouble("CD1_1", 5.10808596133527E-05)
        metadata.setDouble("CD1_2", 1.85579539217196E-07)
        metadata.setDouble("CD2_2", -5.10281493481982E-05)
        metadata.setDouble("CD2_1", -8.27440751733828E-07)
        return makeSkyWcs(metadata)

    @staticmethod
    def createVisitInfo():
        return lsst.afw.image.VisitInfo(
            10313423,
            10.01,
            11.02,
            DateTime(65321.1, DateTime.MJD, DateTime.TAI),
            12345.1,
            45.1*lsst.afw.geom.degrees,
            lsst.afw.coord.IcrsCoord(23.1*degrees, 73.2*degrees),
            lsst.afw.coord.Coord(134.5*degrees, 33.3*degrees),
            1.73,
            73.2*degrees,
            lsst.afw.image.RotType.SKY,
            lsst.afw.coord.Observatory(11.1*degrees, 22.2*degrees, 0.333),
            lsst.afw.coord.Weather(1.1, 2.2, 34.5),
        )

    @staticmethod
    def makePolygon():
        return Polygon([Point2D(1, 2), Point2D(2, 1)])

    def comparePsfs(self, psf1, psf2):
        self.assertIsNotNone(psf1)
        self.assertIsNotNone(psf2)
        self.assertEqual(psf1.getValue(), psf2.getValue())

    def setUp(self):
        np.random.seed(1)
        schema = lsst.afw.table.ExposureTable.makeMinimalSchema()
        self.ka = schema.addField("a", type=np.float64, doc="doc for a")
        self.kb = schema.addField("b", type=np.int64, doc="doc for b")
        self.cat = lsst.afw.table.ExposureCatalog(schema)
        self.wcs = self.createWcs()
        self.psf = DummyPsf(2.0)
        self.bbox0 = lsst.afw.geom.Box2I(
            lsst.afw.geom.Box2D(
                self.wcs.getPixelOrigin() - lsst.afw.geom.Extent2D(5.0, 4.0),
                self.wcs.getPixelOrigin() + lsst.afw.geom.Extent2D(20.0, 30.0)
            )
        )
        self.bbox1 = lsst.afw.geom.Box2I(
            lsst.afw.geom.Box2D(
                self.wcs.getPixelOrigin() - lsst.afw.geom.Extent2D(15.0, 40.0),
                self.wcs.getPixelOrigin() + lsst.afw.geom.Extent2D(3.0, 6.0)
            )
        )
        self.calib = lsst.afw.image.Calib()
        self.calib.setFluxMag0(56.0, 2.2)
        self.visitInfo = self.createVisitInfo()
        record0 = self.cat.addNew()
        record0.setId(1)
        record0.set(self.ka, np.pi)
        record0.set(self.kb, 4)
        record0.setBBox(self.bbox0)
        record0.setPsf(self.psf)
        record0.setWcs(self.wcs)
        record0.setCalib(self.calib)
        record0.setVisitInfo(self.visitInfo)
        record0.setValidPolygon(None)
        record1 = self.cat.addNew()
        record1.setId(2)
        record1.set(self.ka, 2.5)
        record1.set(self.kb, 2)
        record1.setWcs(self.wcs)
        record1.setBBox(self.bbox1)
        record1.setValidPolygon(self.makePolygon())

    def tearDown(self):
        del self.cat
        del self.psf
        del self.wcs
        del self.calib
        del self.visitInfo

    def testAccessors(self):
        record0 = self.cat[0]
        record1 = self.cat[1]
        self.assertEqual(record0.getId(), 1)
        self.assertEqual(record1.getId(), 2)
        self.assertEqual(record0.getWcs(), self.wcs)
        self.assertEqual(record1.getWcs(), self.wcs)
        self.assertEqual(record0.getBBox(), self.bbox0)
        self.assertEqual(record1.getBBox(), self.bbox1)
        self.comparePsfs(record0.getPsf(), self.psf)
        self.assertIsNone(record1.getPsf())
        self.assertEqual(record0.getCalib(), self.calib)
        self.assertIsNone(record1.getCalib())
        self.assertEqual(record0.getVisitInfo(), self.visitInfo)
        self.assertIsNone(record1.getVisitInfo())
        self.assertEqual(record0.getValidPolygon(), None)
        self.assertEqual(record1.getValidPolygon(), self.makePolygon())

    def testPersistence(self):
        with lsst.utils.tests.getTempFilePath(".fits") as tmpFile:
            self.cat.writeFits(tmpFile)
            cat1 = lsst.afw.table.ExposureCatalog.readFits(tmpFile)
            self.assertEqual(self.cat[0].get(self.ka), cat1[0].get(self.ka))
            self.assertEqual(self.cat[0].get(self.kb), cat1[0].get(self.kb))
            self.comparePsfs(self.cat[0].getPsf(), cat1[0].getPsf())
            self.assertEqual(self.cat[0].getWcs(), cat1[0].getWcs())
            self.assertEqual(self.cat[1].get(self.ka), cat1[1].get(self.ka))
            self.assertEqual(self.cat[1].get(self.kb), cat1[1].get(self.kb))
            self.assertEqual(self.cat[1].getWcs(), cat1[1].getWcs())
            self.assertIsNone(self.cat[1].getPsf())
            self.assertIsNone(self.cat[1].getCalib())
            self.assertEqual(self.cat[0].getCalib(), cat1[0].getCalib())
            self.assertEqual(self.cat[0].getVisitInfo(),
                             cat1[0].getVisitInfo())
            self.assertIsNone(cat1[1].getVisitInfo())

    def testGeometry(self):
        bigBox = lsst.afw.geom.Box2D(lsst.afw.geom.Box2I(self.bbox0))
        bigBox.include(lsst.afw.geom.Box2D(self.bbox1))
        points = (np.random.rand(100, 2) * np.array([bigBox.getWidth(), bigBox.getHeight()]) +
                  np.array([bigBox.getMinX(), bigBox.getMinY()]))

        # make a very slightly perturbed wcs so the celestial transform isn't a
        # no-op
        crval2 = self.wcs.getSkyOrigin()
        crval2.reset(crval2.getLongitude() + 5*arcseconds,
                     crval2.getLatitude() - 5*arcseconds)
        wcs2 = makeSkyWcs(
            crval = crval2,
            crpix = self.wcs.getPixelOrigin() + lsst.afw.geom.Extent2D(30.0, -50.0),
            cdMatrix = self.wcs.getCdMatrix() * 1.1,
        )
        for x1, y1 in points:
            p1 = lsst.afw.geom.Point2D(x1, y1)
            c = self.wcs.pixelToSky(p1)
            p2 = wcs2.skyToPixel(c)
            subset1 = self.cat.subsetContaining(c)
            subset2 = self.cat.subsetContaining(p2, wcs2)
            for record in self.cat:
                inside = lsst.afw.geom.Box2D(record.getBBox()).contains(p1)
                self.assertEqual(inside, record.contains(c))
                self.assertEqual(inside, record.contains(p2, wcs2))
                self.assertEqual(inside, record.contains(p1, self.wcs))
                self.assertEqual(inside, record in subset1)
                self.assertEqual(inside, record in subset2)

        crazyPoint = lsst.afw.coord.IcrsCoord(crval2.getLongitude() + np.pi*radians,
                                              crval2.getLatitude())
        subset3 = self.cat.subsetContaining(crazyPoint)
        self.assertEqual(len(subset3), 0)

    def testCoaddInputs(self):
        coaddInputs = lsst.afw.image.CoaddInputs(
            lsst.afw.table.ExposureTable.makeMinimalSchema(),
            lsst.afw.table.ExposureTable.makeMinimalSchema()
        )
        coaddInputs.visits.addNew().setId(2)
        coaddInputs.ccds.addNew().setId(3)
        coaddInputs.ccds.addNew().setId(4)
        exposureIn = lsst.afw.image.ExposureF(10, 10)
        exposureIn.getInfo().setCoaddInputs(coaddInputs)
        with lsst.utils.tests.getTempFilePath(".fits") as filename:
            exposureIn.writeFits(filename)
            exposureOut = lsst.afw.image.ExposureF(filename)
            coaddInputsOut = exposureOut.getInfo().getCoaddInputs()
            self.assertEqual(len(coaddInputsOut.visits), 1)
            self.assertEqual(len(coaddInputsOut.ccds), 2)
            self.assertEqual(coaddInputsOut.visits[0].getId(), 2)
            self.assertEqual(coaddInputsOut.ccds[0].getId(), 3)
            self.assertEqual(coaddInputsOut.ccds[1].getId(), 4)

    def testReadV1Catalog(self):
        testDir = os.path.dirname(__file__)
        v1CatalogPath = os.path.join(
            testDir, "data", "exposure_catalog_v1.fits")
        catV1 = lsst.afw.table.ExposureCatalog.readFits(v1CatalogPath)
        self.assertEqual(self.cat[0].get(self.ka), catV1[0].get(self.ka))
        self.assertEqual(self.cat[0].get(self.kb), catV1[0].get(self.kb))
        self.comparePsfs(self.cat[0].getPsf(), catV1[0].getPsf())
        bbox = Box2D(Point2D(0, 0), Extent2D(2000, 2000))
        self.assertWcsAlmostEqualOverBBox(self.cat[0].getWcs(), catV1[0].getWcs(), bbox)
        self.assertEqual(self.cat[1].get(self.ka), catV1[1].get(self.ka))
        self.assertEqual(self.cat[1].get(self.kb), catV1[1].get(self.kb))
        self.assertEqual(self.cat[1].getWcs(), catV1[1].getWcs())
        self.assertIsNone(self.cat[1].getPsf())
        self.assertIsNone(self.cat[1].getCalib())
        self.assertEqual(self.cat[0].getCalib(), catV1[0].getCalib())
        self.assertIsNone(catV1[0].getVisitInfo())
        self.assertIsNone(catV1[1].getVisitInfo())


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
