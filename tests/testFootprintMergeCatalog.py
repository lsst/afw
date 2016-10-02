#pybind11##!/usr/bin/env python
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
#pybind11#from __future__ import division
#pybind11#import unittest
#pybind11#import lsst.utils.tests
#pybind11#import lsst.pex.exceptions
#pybind11#import lsst.afw.image as afwImage
#pybind11#import lsst.afw.geom as afwGeom
#pybind11#import lsst.afw.detection as afwDetect
#pybind11#import lsst.afw.table as afwTable
#pybind11#import numpy as np
#pybind11#
#pybind11#
#pybind11#def insertPsf(pos, im, psf, kernelSize, flux):
#pybind11#    for x, y in pos:
#pybind11#        x0 = x-kernelSize//2
#pybind11#        y0 = y-kernelSize//2
#pybind11#        tmpbox = afwGeom.Box2I(afwGeom.Point2I(x0, y0), afwGeom.Extent2I(kernelSize, kernelSize))
#pybind11#        tmp = psf.computeImage(afwGeom.Point2D(x0, y0))
#pybind11#        tmp *= flux
#pybind11#        im.getImage()[tmpbox] += tmp
#pybind11#
#pybind11#
#pybind11#def mergeCatalogs(catList, names, peakDist, idFactory, indivNames=[], samePeakDist=-1.):
#pybind11#    schema = afwTable.SourceTable.makeMinimalSchema()
#pybind11#    merged = afwDetect.FootprintMergeList(schema, names)
#pybind11#
#pybind11#    if not indivNames:
#pybind11#        indivNames = names
#pybind11#
#pybind11#    # Count the number of objects and peaks in this list
#pybind11#    mergedList = merged.getMergedSourceCatalog(catList, indivNames, peakDist,
#pybind11#                                               schema, idFactory, samePeakDist)
#pybind11#    nob = len(mergedList)
#pybind11#    npeaks = sum([len(ob.getFootprint().getPeaks()) for ob in mergedList])
#pybind11#
#pybind11#    return mergedList, nob, npeaks
#pybind11#
#pybind11#
#pybind11#def isPeakInCatalog(peak, catalog):
#pybind11#    for record in catalog:
#pybind11#        for p in record.getFootprint().getPeaks():
#pybind11#            if p.getI() == peak.getI():
#pybind11#                return True
#pybind11#    return False
#pybind11#
#pybind11#
#pybind11#class FootprintMergeCatalogTestCase(lsst.utils.tests.TestCase):
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        """Build up three different sets of objects that are to be merged"""
#pybind11#        pos1 = [(40, 40), (220, 35), (40, 48), (220, 50),
#pybind11#                (67, 67), (150, 50), (40, 90), (70, 160),
#pybind11#                (35, 255), (70, 180), (250, 200), (120, 120),
#pybind11#                (170, 180), (100, 210), (20, 210),
#pybind11#                ]
#pybind11#        pos2 = [(43, 45), (215, 31), (171, 258), (211, 117),
#pybind11#                (48, 99), (70, 160), (125, 45), (251, 33),
#pybind11#                (37, 170), (134, 191), (79, 223), (258, 182)
#pybind11#                ]
#pybind11#        pos3 = [(70, 170), (219, 41), (253, 173), (253, 192)]
#pybind11#
#pybind11#        box = afwGeom.Box2I(afwGeom.Point2I(0, 0), afwGeom.Point2I(300, 300))
#pybind11#        psfsig = 1.
#pybind11#        kernelSize = 41
#pybind11#        flux = 1000
#pybind11#
#pybind11#        # Create a different sized psf for each image and insert them at the desired positions
#pybind11#        im1 = afwImage.MaskedImageD(box)
#pybind11#        psf1 = afwDetect.GaussianPsf(kernelSize, kernelSize, psfsig)
#pybind11#
#pybind11#        im2 = afwImage.MaskedImageD(box)
#pybind11#        psf2 = afwDetect.GaussianPsf(kernelSize, kernelSize, 2*psfsig)
#pybind11#
#pybind11#        im3 = afwImage.MaskedImageD(box)
#pybind11#        psf3 = afwDetect.GaussianPsf(kernelSize, kernelSize, 1.3*psfsig)
#pybind11#
#pybind11#        insertPsf(pos1, im1, psf1, kernelSize, flux)
#pybind11#        insertPsf(pos2, im2, psf2, kernelSize, flux)
#pybind11#        insertPsf(pos3, im3, psf3, kernelSize, flux)
#pybind11#
#pybind11#        schema = afwTable.SourceTable.makeMinimalSchema()
#pybind11#        self.idFactory = afwTable.IdFactory.makeSimple()
#pybind11#        self.table = afwTable.SourceTable.make(schema, self.idFactory)
#pybind11#
#pybind11#        # Create SourceCatalogs from these objects
#pybind11#        fp1 = afwDetect.FootprintSet(im1, afwDetect.Threshold(0.001), "DETECTED")
#pybind11#        self.catalog1 = afwTable.SourceCatalog(self.table)
#pybind11#        fp1.makeSources(self.catalog1)
#pybind11#
#pybind11#        fp2 = afwDetect.FootprintSet(im2, afwDetect.Threshold(0.001), "DETECTED")
#pybind11#        self.catalog2 = afwTable.SourceCatalog(self.table)
#pybind11#        fp2.makeSources(self.catalog2)
#pybind11#
#pybind11#        fp3 = afwDetect.FootprintSet(im3, afwDetect.Threshold(0.001), "DETECTED")
#pybind11#        self.catalog3 = afwTable.SourceCatalog(self.table)
#pybind11#        fp3.makeSources(self.catalog3)
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        del self.catalog1
#pybind11#        del self.catalog2
#pybind11#        del self.catalog3
#pybind11#        del self.table
#pybind11#
#pybind11#    def testMerge1(self):
#pybind11#        # Add the first catalog only
#pybind11#        merge, nob, npeak = mergeCatalogs([self.catalog1], ["1"], [-1],
#pybind11#                                          self.idFactory)
#pybind11#        self.assertEqual(nob, 14)
#pybind11#        self.assertEqual(npeak, 15)
#pybind11#
#pybind11#        for record in merge:
#pybind11#            self.assertTrue(record.get("merge_footprint_1"))
#pybind11#            for peak in record.getFootprint().getPeaks():
#pybind11#                self.assertTrue(peak.get("merge_peak_1"))
#pybind11#
#pybind11#        # area for each object
#pybind11#        pixArea = np.empty(14)
#pybind11#        pixArea.fill(69)
#pybind11#        pixArea[1] = 135
#pybind11#        measArea = [i.getFootprint().getArea() for i in merge]
#pybind11#        np.testing.assert_array_equal(pixArea, measArea)
#pybind11#
#pybind11#        # Add the first catalog and second catalog with the wrong name, which should result
#pybind11#        # an exception being raised
#pybind11#        with self.assertRaises(lsst.pex.exceptions.LogicError):
#pybind11#            mergeCatalogs([self.catalog1, self.catalog2], ["1", "2"], [0, 0], self.idFactory, ["1", "3"])
#pybind11#
#pybind11#        # Add the first catalog and second catalog with the wrong number of peakDist elements,
#pybind11#        # which should raise an exception
#pybind11#        with self.assertRaises(ValueError):
#pybind11#            mergeCatalogs([self.catalog1, self.catalog2], ["1", "2"], [0], self.idFactory, ["1", "3"])
#pybind11#
#pybind11#        # Add the first catalog and second catalog with the wrong number of filters,
#pybind11#        # which should raise an exception
#pybind11#        with self.assertRaises(ValueError):
#pybind11#            mergeCatalogs([self.catalog1, self.catalog2], ["1"], [0], self.idFactory, ["1", "3"])
#pybind11#
#pybind11#        # Add the first catalog and second catalog with minPeak < 1 so it will not add new peaks
#pybind11#        merge, nob, npeak = mergeCatalogs([self.catalog1, self.catalog2],
#pybind11#                                          ["1", "2"], [0, -1],
#pybind11#                                          self.idFactory)
#pybind11#        self.assertEqual(nob, 22)
#pybind11#        self.assertEqual(npeak, 23)
#pybind11#        # area for each object
#pybind11#        pixArea = np.ones(22)
#pybind11#        pixArea[0] = 275
#pybind11#        pixArea[1] = 270
#pybind11#        pixArea[2:5].fill(69)
#pybind11#        pixArea[5] = 323
#pybind11#        pixArea[6] = 69
#pybind11#        pixArea[7] = 261
#pybind11#        pixArea[8:14].fill(69)
#pybind11#        pixArea[14:22].fill(261)
#pybind11#        measArea = [i.getFootprint().getArea() for i in merge]
#pybind11#        np.testing.assert_array_equal(pixArea, measArea)
#pybind11#
#pybind11#        for record in merge:
#pybind11#            for peak in record.getFootprint().getPeaks():
#pybind11#                # Should only get peaks from catalog2 if catalog1 didn't contribute to the footprint
#pybind11#                if record.get("merge_footprint_1"):
#pybind11#                    self.assertTrue(peak.get("merge_peak_1"))
#pybind11#                    self.assertFalse(peak.get("merge_peak_2"))
#pybind11#                else:
#pybind11#                    self.assertFalse(peak.get("merge_peak_1"))
#pybind11#                    self.assertTrue(peak.get("merge_peak_2"))
#pybind11#
#pybind11#        # Same as previous with another catalog
#pybind11#        merge, nob, npeak = mergeCatalogs([self.catalog1, self.catalog2, self.catalog3],
#pybind11#                                          ["1", "2", "3"], [0, -1, -1],
#pybind11#                                          self.idFactory)
#pybind11#        self.assertEqual(nob, 19)
#pybind11#        self.assertEqual(npeak, 20)
#pybind11#        pixArea = np.ones(19)
#pybind11#        pixArea[0] = 416
#pybind11#        pixArea[1] = 270
#pybind11#        pixArea[2:4].fill(69)
#pybind11#        pixArea[4] = 323
#pybind11#        pixArea[5] = 69
#pybind11#        pixArea[6] = 406
#pybind11#        pixArea[7] = 69
#pybind11#        pixArea[8] = 493
#pybind11#        pixArea[9:13].fill(69)
#pybind11#        pixArea[12:19].fill(261)
#pybind11#        measArea = [i.getFootprint().getArea() for i in merge]
#pybind11#        np.testing.assert_array_equal(pixArea, measArea)
#pybind11#
#pybind11#        for record in merge:
#pybind11#            for peak in record.getFootprint().getPeaks():
#pybind11#                # Should only get peaks from catalog2 if catalog1 didn't contribute to the footprint
#pybind11#                if record.get("merge_footprint_1"):
#pybind11#                    self.assertTrue(peak.get("merge_peak_1"))
#pybind11#                    self.assertFalse(peak.get("merge_peak_2"))
#pybind11#                else:
#pybind11#                    self.assertFalse(peak.get("merge_peak_1"))
#pybind11#                    self.assertTrue(peak.get("merge_peak_2"))
#pybind11#
#pybind11#        # Add all the catalogs with minPeak = 0 so all peaks will be added
#pybind11#        merge, nob, npeak = mergeCatalogs([self.catalog1, self.catalog2, self.catalog3],
#pybind11#                                          ["1", "2", "3"], [0, 0, 0],
#pybind11#                                          self.idFactory)
#pybind11#        self.assertEqual(nob, 19)
#pybind11#        self.assertEqual(npeak, 30)
#pybind11#        measArea = [i.getFootprint().getArea() for i in merge]
#pybind11#        np.testing.assert_array_equal(pixArea, measArea)
#pybind11#
#pybind11#        # Add all the catalogs with minPeak = 10 so some peaks will be added to the footprint
#pybind11#        merge, nob, npeak = mergeCatalogs([self.catalog1, self.catalog2, self.catalog3],
#pybind11#                                          ["1", "2", "3"], 10, self.idFactory)
#pybind11#        self.assertEqual(nob, 19)
#pybind11#        self.assertEqual(npeak, 25)
#pybind11#        measArea = [i.getFootprint().getArea() for i in merge]
#pybind11#        np.testing.assert_array_equal(pixArea, measArea)
#pybind11#
#pybind11#        for record in merge:
#pybind11#            for peak in record.getFootprint().getPeaks():
#pybind11#                if peak.get("merge_peak_1"):
#pybind11#                    self.assertTrue(record.get("merge_footprint_1"))
#pybind11#                    self.assertTrue(isPeakInCatalog(peak, self.catalog1))
#pybind11#                elif peak.get("merge_peak_2"):
#pybind11#                    self.assertTrue(record.get("merge_footprint_2"))
#pybind11#                    self.assertTrue(isPeakInCatalog(peak, self.catalog2))
#pybind11#                elif peak.get("merge_peak_3"):
#pybind11#                    self.assertTrue(record.get("merge_footprint_3"))
#pybind11#                    self.assertTrue(isPeakInCatalog(peak, self.catalog3))
#pybind11#                else:
#pybind11#                    self.fail("At least one merge.peak flag must be set")
#pybind11#
#pybind11#        # Add all the catalogs with minPeak = 100 so no new peaks will be added
#pybind11#        merge, nob, npeak = mergeCatalogs([self.catalog1, self.catalog2, self.catalog3],
#pybind11#                                          ["1", "2", "3"], 100, self.idFactory)
#pybind11#        self.assertEqual(nob, 19)
#pybind11#        self.assertEqual(npeak, 20)
#pybind11#        measArea = [i.getFootprint().getArea() for i in merge]
#pybind11#        np.testing.assert_array_equal(pixArea, measArea)
#pybind11#
#pybind11#        for record in merge:
#pybind11#            for peak in record.getFootprint().getPeaks():
#pybind11#                if peak.get("merge_peak_1"):
#pybind11#                    self.assertTrue(record.get("merge_footprint_1"))
#pybind11#                    self.assertTrue(isPeakInCatalog(peak, self.catalog1))
#pybind11#                elif peak.get("merge_peak_2"):
#pybind11#                    self.assertTrue(record.get("merge_footprint_2"))
#pybind11#                    self.assertTrue(isPeakInCatalog(peak, self.catalog2))
#pybind11#                elif peak.get("merge_peak_3"):
#pybind11#                    self.assertTrue(record.get("merge_footprint_3"))
#pybind11#                    self.assertTrue(isPeakInCatalog(peak, self.catalog3))
#pybind11#                else:
#pybind11#                    self.fail("At least one merge_peak flag must be set")
#pybind11#
#pybind11#        # Add footprints with large samePeakDist so that any footprint that merges will also
#pybind11#        # have the peak flagged
#pybind11#        merge, nob, npeak = mergeCatalogs([self.catalog1, self.catalog2, self.catalog3],
#pybind11#                                          ["1", "2", "3"], 100, self.idFactory, samePeakDist=40)
#pybind11#
#pybind11#        # peaks detected in more than one catalog
#pybind11#        multiPeakIndex = [0, 2, 5, 7, 9]
#pybind11#        peakIndex = 0
#pybind11#        for record in merge:
#pybind11#            for peak in record.getFootprint().getPeaks():
#pybind11#                numPeak = np.sum([peak.get("merge_peak_1"), peak.get("merge_peak_2"),
#pybind11#                                  peak.get("merge_peak_3")])
#pybind11#                if peakIndex in multiPeakIndex:
#pybind11#                    self.assertGreater(numPeak, 1)
#pybind11#                else:
#pybind11#                    self.assertEqual(numPeak, 1)
#pybind11#                peakIndex += 1
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
