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
#pybind11#Tests for lsst.afw.table.ExposureTable
#pybind11#
#pybind11#Run with:
#pybind11#   ./testExposureTable.py
#pybind11#or
#pybind11#   python
#pybind11#   >>> import testExposureTable; testExposureTable.run()
#pybind11#"""
#pybind11#
#pybind11#import unittest
#pybind11#import numpy
#pybind11#
#pybind11#import lsst.utils.tests
#pybind11#import lsst.pex.exceptions
#pybind11#import lsst.daf.base
#pybind11#import lsst.afw.table
#pybind11#import lsst.afw.geom
#pybind11#import lsst.afw.coord
#pybind11#import lsst.afw.image
#pybind11#import lsst.afw.detection
#pybind11#from testTableArchivesLib import DummyPsf
#pybind11#
#pybind11#try:
#pybind11#    type(display)
#pybind11#except NameError:
#pybind11#    display = False
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#
#pybind11#
#pybind11#class ExposureTableTestCase(lsst.utils.tests.TestCase):
#pybind11#
#pybind11#    @staticmethod
#pybind11#    def createWcs():
#pybind11#        metadata = lsst.daf.base.PropertySet()
#pybind11#        metadata.set("SIMPLE", "T")
#pybind11#        metadata.set("BITPIX", -32)
#pybind11#        metadata.set("NAXIS", 2)
#pybind11#        metadata.set("NAXIS1", 1024)
#pybind11#        metadata.set("NAXIS2", 1153)
#pybind11#        metadata.set("RADECSYS", 'FK5')
#pybind11#        metadata.set("EQUINOX", 2000.)
#pybind11#        metadata.setDouble("CRVAL1", 215.604025685476)
#pybind11#        metadata.setDouble("CRVAL2", 53.1595451514076)
#pybind11#        metadata.setDouble("CRPIX1", 1109.99981456774)
#pybind11#        metadata.setDouble("CRPIX2", 560.018167811613)
#pybind11#        metadata.set("CTYPE1", 'RA---SIN')
#pybind11#        metadata.set("CTYPE2", 'DEC--SIN')
#pybind11#        metadata.setDouble("CD1_1", 5.10808596133527E-05)
#pybind11#        metadata.setDouble("CD1_2", 1.85579539217196E-07)
#pybind11#        metadata.setDouble("CD2_2", -5.10281493481982E-05)
#pybind11#        metadata.setDouble("CD2_1", -8.27440751733828E-07)
#pybind11#        return lsst.afw.image.makeWcs(metadata)
#pybind11#
#pybind11#    def comparePsfs(self, psf1, psf2):
#pybind11#        psf1 = DummyPsf.swigConvert(psf1)
#pybind11#        psf2 = DummyPsf.swigConvert(psf2)
#pybind11#        self.assertIsNotNone(psf1)
#pybind11#        self.assertIsNotNone(psf2)
#pybind11#        self.assertEqual(psf1.getValue(), psf2.getValue())
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        numpy.random.seed(1)
#pybind11#        schema = lsst.afw.table.ExposureTable.makeMinimalSchema()
#pybind11#        self.ka = schema.addField("a", type=float, doc="doc for a")
#pybind11#        self.kb = schema.addField("b", type=int, doc="doc for b")
#pybind11#        self.cat = lsst.afw.table.ExposureCatalog(schema)
#pybind11#        self.wcs = self.createWcs()
#pybind11#        self.psf = DummyPsf(2.0)
#pybind11#        self.bbox0 = lsst.afw.geom.Box2I(
#pybind11#            lsst.afw.geom.Box2D(
#pybind11#                self.wcs.getPixelOrigin() - lsst.afw.geom.Extent2D(5.0, 4.0),
#pybind11#                self.wcs.getPixelOrigin() + lsst.afw.geom.Extent2D(20.0, 30.0)
#pybind11#            )
#pybind11#        )
#pybind11#        self.bbox1 = lsst.afw.geom.Box2I(
#pybind11#            lsst.afw.geom.Box2D(
#pybind11#                self.wcs.getPixelOrigin() - lsst.afw.geom.Extent2D(15.0, 40.0),
#pybind11#                self.wcs.getPixelOrigin() + lsst.afw.geom.Extent2D(3.0, 6.0)
#pybind11#            )
#pybind11#        )
#pybind11#        self.calib = lsst.afw.image.Calib()
#pybind11#        self.calib.setFluxMag0(56.0, 2.2)
#pybind11#        self.calib.setExptime(50.0)
#pybind11#        self.calib.setMidTime(lsst.daf.base.DateTime.now())
#pybind11#        record0 = self.cat.addNew()
#pybind11#        record0.setId(1)
#pybind11#        record0.set(self.ka, numpy.pi)
#pybind11#        record0.set(self.kb, 4)
#pybind11#        record0.setBBox(self.bbox0)
#pybind11#        record0.setPsf(self.psf)
#pybind11#        record0.setWcs(self.wcs)
#pybind11#        record0.setCalib(self.calib)
#pybind11#        record1 = self.cat.addNew()
#pybind11#        record1.setId(2)
#pybind11#        record1.set(self.ka, 2.5)
#pybind11#        record1.set(self.kb, 2)
#pybind11#        record1.setWcs(self.wcs)
#pybind11#        record1.setBBox(self.bbox1)
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        del self.cat
#pybind11#        del self.psf
#pybind11#        del self.wcs
#pybind11#        del self.calib
#pybind11#
#pybind11#    def testAccessors(self):
#pybind11#        record0 = self.cat[0]
#pybind11#        record1 = self.cat[1]
#pybind11#        self.assertEqual(record0.getId(), 1)
#pybind11#        self.assertEqual(record1.getId(), 2)
#pybind11#        self.assertEqual(record0.getWcs(), self.wcs)
#pybind11#        self.assertEqual(record1.getWcs(), self.wcs)
#pybind11#        self.assertEqual(record0.getBBox(), self.bbox0)
#pybind11#        self.assertEqual(record1.getBBox(), self.bbox1)
#pybind11#        self.comparePsfs(record0.getPsf(), self.psf)
#pybind11#        self.assertIsNone(record1.getPsf())
#pybind11#        self.assertEqual(record0.getCalib(), self.calib)
#pybind11#        self.assertIsNone(record1.getCalib())
#pybind11#
#pybind11#    def testPersistence(self):
#pybind11#        with lsst.utils.tests.getTempFilePath(".fits") as tmpFile:
#pybind11#            self.cat.writeFits(tmpFile)
#pybind11#            cat1 = lsst.afw.table.ExposureCatalog.readFits(tmpFile)
#pybind11#            self.assertEqual(self.cat[0].get(self.ka), cat1[0].get(self.ka))
#pybind11#            self.assertEqual(self.cat[0].get(self.kb), cat1[0].get(self.kb))
#pybind11#            self.comparePsfs(self.cat[0].getPsf(), cat1[0].getPsf())
#pybind11#            self.assertEqual(self.cat[0].getWcs(), cat1[0].getWcs())
#pybind11#            self.assertEqual(self.cat[1].get(self.ka), cat1[1].get(self.ka))
#pybind11#            self.assertEqual(self.cat[1].get(self.kb), cat1[1].get(self.kb))
#pybind11#            self.assertEqual(self.cat[1].getWcs(), cat1[1].getWcs())
#pybind11#            self.assertIsNone(self.cat[1].getPsf())
#pybind11#            self.assertIsNone(self.cat[1].getCalib())
#pybind11#            self.assertEqual(self.cat[0].getWcs().getId(), self.cat[
#pybind11#                             1].getWcs().getId())  # compare citizen IDs
#pybind11#            self.assertEqual(self.cat[0].getCalib(), cat1[0].getCalib())
#pybind11#
#pybind11#    def testGeometry(self):
#pybind11#        bigBox = lsst.afw.geom.Box2D(lsst.afw.geom.Box2I(self.bbox0))
#pybind11#        bigBox.include(lsst.afw.geom.Box2D(self.bbox1))
#pybind11#        points = (numpy.random.rand(100, 2) * numpy.array([bigBox.getWidth(), bigBox.getHeight()])
#pybind11#                  + numpy.array([bigBox.getMinX(), bigBox.getMinY()]))
#pybind11#
#pybind11#        # make a very slightly perturbed wcs so the celestial transform isn't a no-op
#pybind11#        crval2 = self.wcs.getSkyOrigin()
#pybind11#        crval2.reset(crval2.getLongitude() + 5 * lsst.afw.geom.arcseconds,
#pybind11#                     crval2.getLatitude() - 5 * lsst.afw.geom.arcseconds)
#pybind11#        wcs2 = lsst.afw.image.Wcs(
#pybind11#            crval2.getPosition(), self.wcs.getPixelOrigin() + lsst.afw.geom.Extent2D(30.0, -50.0),
#pybind11#            self.wcs.getCDMatrix() * 1.1
#pybind11#        )
#pybind11#        for x1, y1 in points:
#pybind11#            p1 = lsst.afw.geom.Point2D(x1, y1)
#pybind11#            c = self.wcs.pixelToSky(x1, y1)
#pybind11#            p2 = wcs2.skyToPixel(c)
#pybind11#            subset1 = self.cat.subsetContaining(c)
#pybind11#            subset2 = self.cat.subsetContaining(p2, wcs2)
#pybind11#            for record in self.cat:
#pybind11#                inside = lsst.afw.geom.Box2D(record.getBBox()).contains(p1)
#pybind11#                self.assertEqual(inside, record.contains(c))
#pybind11#                self.assertEqual(inside, record.contains(p2, wcs2))
#pybind11#                self.assertEqual(inside, record.contains(p1, self.wcs))
#pybind11#                self.assertEqual(inside, record in subset1)
#pybind11#                self.assertEqual(inside, record in subset2)
#pybind11#
#pybind11#        crazyPoint = lsst.afw.coord.IcrsCoord(crval2.getLongitude() + numpy.pi * lsst.afw.geom.radians,
#pybind11#                                              crval2.getLatitude())
#pybind11#        subset3 = self.cat.subsetContaining(crazyPoint)
#pybind11#        self.assertEqual(len(subset3), 0)
#pybind11#
#pybind11#    def testCoaddInputs(self):
#pybind11#        coaddInputs = lsst.afw.image.CoaddInputs(
#pybind11#            lsst.afw.table.ExposureTable.makeMinimalSchema(),
#pybind11#            lsst.afw.table.ExposureTable.makeMinimalSchema()
#pybind11#        )
#pybind11#        coaddInputs.visits.addNew().setId(2)
#pybind11#        coaddInputs.ccds.addNew().setId(3)
#pybind11#        coaddInputs.ccds.addNew().setId(4)
#pybind11#        exposureIn = lsst.afw.image.ExposureF(10, 10)
#pybind11#        exposureIn.getInfo().setCoaddInputs(coaddInputs)
#pybind11#        with lsst.utils.tests.getTempFilePath(".fits") as filename:
#pybind11#            exposureIn.writeFits(filename)
#pybind11#            exposureOut = lsst.afw.image.ExposureF(filename)
#pybind11#            coaddInputsOut = exposureOut.getInfo().getCoaddInputs()
#pybind11#            self.assertEqual(len(coaddInputsOut.visits), 1)
#pybind11#            self.assertEqual(len(coaddInputsOut.ccds), 2)
#pybind11#            self.assertEqual(coaddInputsOut.visits[0].getId(), 2)
#pybind11#            self.assertEqual(coaddInputsOut.ccds[0].getId(), 3)
#pybind11#            self.assertEqual(coaddInputsOut.ccds[1].getId(), 4)
#pybind11#
#pybind11#
#pybind11#class MemoryTester(lsst.utils.tests.MemoryTestCase):
#pybind11#    pass
#pybind11#
#pybind11#
#pybind11#def setup_module(module):
#pybind11#    lsst.utils.tests.init()
#pybind11#
#pybind11#
#pybind11#if __name__ == "__main__":
#pybind11#    lsst.utils.tests.init()
#pybind11#    unittest.main()
