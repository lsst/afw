#!/usr/bin/env python

import unittest
import lsst.utils.tests as tests
import lsst.pex.exceptions
import lsst.afw.image as afwImage
import lsst.afw.geom as afwGeom
import lsst.afw.detection as afwDetect
import lsst.afw.table as afwTable
import numpy as np
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def insertPsf(pos, im, psf, kernelSize, flux):
    for x, y in pos:
        x0 = x-kernelSize/2
        y0 = y-kernelSize/2
        tmpbox =  afwGeom.Box2I(afwGeom.Point2I(x0,y0),afwGeom.Extent2I(kernelSize,kernelSize))
        tmp = psf.computeImage(afwGeom.Point2D(x0,y0))
        tmp *= flux
        im.getImage()[tmpbox] += tmp

def mergeCatalogs(catList, names, peakDist, idFactory, indivNames=[], samePeakDist=-1.):
    schema = afwTable.SourceTable.makeMinimalSchema()
    merged = afwDetect.FootprintMergeList(schema, names)

    if not indivNames:
        indivNames = names

    # Count the number of objects and peaks in this list
    mergedList = merged.getMergedSourceCatalog(catList, indivNames, peakDist,
                                               schema, idFactory, samePeakDist)
    nob = len(mergedList)
    npeaks = sum([ len(ob.getFootprint().getPeaks()) for ob in mergedList])

    return mergedList, nob, npeaks

def isPeakInCatalog(peak, catalog):
    for record in catalog:
        for p in record.getFootprint().getPeaks():
            if p.getI() == peak.getI():
                return True
    return False

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class FootprintMergeCatalogTestCase(tests.TestCase):

    def setUp(self):

        """Build up three different sets of objects that are to be merged"""
        pos1 = [(40, 40), (220, 35), (40, 48), (220, 50),
                (67, 67),(150, 50), (40, 90), (70, 160),
                (35, 255), (70, 180), (250, 200), (120, 120),
                (170, 180),(100, 210), (20, 210),
                ]
        pos2 = [(43, 45), (215, 31), (171, 258), (211, 117),
                (48, 99), (70, 160), (125, 45), (251, 33),
                (37, 170),(134, 191), (79, 223),(258, 182)
                ]
        pos3 = [(70,170),(219,41),(253,173),(253,192)]

        box = afwGeom.Box2I(afwGeom.Point2I(0,0), afwGeom.Point2I(300,300))
        psfsig = 1.
        kernelSize = 41
        flux = 1000

        # Create a different sized psf for each image and insert them at the desired positions
        im1 = afwImage.MaskedImageD(box)
        psf1 = afwDetect.GaussianPsf(kernelSize, kernelSize, psfsig)

        im2 = afwImage.MaskedImageD(box)
        psf2 = afwDetect.GaussianPsf(kernelSize, kernelSize, 2*psfsig)

        im3 = afwImage.MaskedImageD(box)
        psf3 = afwDetect.GaussianPsf(kernelSize, kernelSize, 1.3*psfsig)

        insertPsf(pos1, im1, psf1, kernelSize, flux)
        insertPsf(pos2, im2, psf2, kernelSize, flux)
        insertPsf(pos3, im3, psf3, kernelSize, flux)

        schema = afwTable.SourceTable.makeMinimalSchema()
        self.idFactory = afwTable.IdFactory.makeSimple()
        self.table = afwTable.SourceTable.make(schema, self.idFactory)

        # Create SourceCatalogs from these objects
        fp1 = afwDetect.FootprintSet(im1, afwDetect.Threshold(0.001), "DETECTED")
        self.catalog1 = afwTable.SourceCatalog(self.table)
        fp1.makeSources(self.catalog1)

        fp2 = afwDetect.FootprintSet(im2, afwDetect.Threshold(0.001), "DETECTED")
        self.catalog2 = afwTable.SourceCatalog(self.table)
        fp2.makeSources(self.catalog2)

        fp3 = afwDetect.FootprintSet(im3, afwDetect.Threshold(0.001), "DETECTED")
        self.catalog3 = afwTable.SourceCatalog(self.table)
        fp3.makeSources(self.catalog3)

    def tearDown(self):
        del self.catalog1
        del self.catalog2
        del self.catalog3
        del self.table


    def testMerge1(self):
        # Add the first catalog only
        merge, nob, npeak = mergeCatalogs([self.catalog1], ["1"], [-1],
                                          self.idFactory)
        self.assertEqual(nob, 14)
        self.assertEqual(npeak, 15)

        for record in merge:
            self.assertTrue(record.get("merge.footprint.1"))
            for peak in record.getFootprint().getPeaks():
                self.assertTrue(peak.get("merge.peak.1"))

        # area for each object
        pixArea = np.empty(14)
        pixArea.fill(69)
        pixArea[1] = 135
        measArea = [i.getFootprint().getArea() for i in merge]
        self.assert_(np.all(pixArea == measArea))

        # Add the first catalog and second catalog with the wrong name, which should result
        # an exception being raised
        self.assertRaisesLsstCpp(lsst.pex.exceptions.LogicError,
                                 mergeCatalogs, [self.catalog1,self.catalog2], ["1","2"], [0, 0],
                                 self.idFactory, ["1","3"])

        # Add the first catalog and second catalog with the wrong number of peakDist elements,
        # which should raise an exception
        self.assertRaises(ValueError, mergeCatalogs, [self.catalog1,self.catalog2], ["1","2"], [0],
                          self.idFactory, ["1","3"])

        # Add the first catalog and second catalog with the wrong number of filters,
        # which should raise an exception
        self.assertRaises(ValueError, mergeCatalogs, [self.catalog1,self.catalog2], ["1"], [0],
                          self.idFactory, ["1","3"])

        # Add the first catalog and second catalog with minPeak < 1 so it will not add new peaks
        merge, nob, npeak = mergeCatalogs([self.catalog1, self.catalog2],
                                          ["1", "2"], [0, -1],
                                          self.idFactory)
        self.assertEqual(nob, 22)
        self.assertEqual(npeak, 23)
        # area for each object
        pixArea = np.ones(22)
        pixArea[0] = 275
        pixArea[1] = 270
        pixArea[2:5].fill(69)
        pixArea[5] = 323
        pixArea[6] = 69
        pixArea[7] = 261
        pixArea[8:14].fill(69)
        pixArea[14:22].fill(261)
        measArea = [i.getFootprint().getArea() for i in merge]
        self.assert_(np.all(pixArea == measArea))

        for record in merge:
            for peak in record.getFootprint().getPeaks():
                # Should only get peaks from catalog2 if catalog1 didn't contribute to the footprint
                if record.get("merge.footprint.1"):
                    self.assertTrue(peak.get("merge.peak.1"))
                    self.assertFalse(peak.get("merge.peak.2"))
                else:
                    self.assertFalse(peak.get("merge.peak.1"))
                    self.assertTrue(peak.get("merge.peak.2"))

        # Same as previous with another catalog
        merge, nob, npeak = mergeCatalogs([self.catalog1, self.catalog2, self.catalog3],
                                          ["1", "2", "3"], [0, -1, -1],
                                          self.idFactory)
        self.assertEqual(nob, 19)
        self.assertEqual(npeak, 20)
        pixArea = np.ones(19)
        pixArea[0] = 416
        pixArea[1] = 270
        pixArea[2:4].fill(69)
        pixArea[4] = 323
        pixArea[5] = 69
        pixArea[6] = 406
        pixArea[7] = 69
        pixArea[8] = 493
        pixArea[9:13].fill(69)
        pixArea[12:19].fill(261)
        measArea = [i.getFootprint().getArea() for i in merge]
        self.assert_(np.all(pixArea == measArea))

        for record in merge:
            for peak in record.getFootprint().getPeaks():
                # Should only get peaks from catalog2 if catalog1 didn't contribute to the footprint
                if record.get("merge.footprint.1"):
                    self.assertTrue(peak.get("merge.peak.1"))
                    self.assertFalse(peak.get("merge.peak.2"))
                else:
                    self.assertFalse(peak.get("merge.peak.1"))
                    self.assertTrue(peak.get("merge.peak.2"))

        # Add all the catalogs with minPeak = 0 so all peaks will be added
        merge, nob, npeak = mergeCatalogs([self.catalog1, self.catalog2, self.catalog3],
                                          ["1", "2", "3"], [0, 0, 0],
                                          self.idFactory)
        self.assertEqual(nob, 19)
        self.assertEqual(npeak, 30)
        measArea = [i.getFootprint().getArea() for i in merge]
        self.assert_(np.all(pixArea == measArea))

        # Add all the catalogs with minPeak = 10 so some peaks will be added to the footprint
        merge, nob, npeak = mergeCatalogs([self.catalog1, self.catalog2, self.catalog3],
                                          ["1", "2", "3"], 10, self.idFactory)
        self.assertEqual(nob, 19)
        self.assertEqual(npeak, 25)
        measArea = [i.getFootprint().getArea() for i in merge]
        self.assert_(np.all(pixArea == measArea))

        for record in merge:
            for peak in record.getFootprint().getPeaks():
                if peak.get("merge.peak.1"):
                    self.assertTrue(record.get("merge.footprint.1"))
                    self.assertTrue(isPeakInCatalog(peak, self.catalog1))
                elif peak.get("merge.peak.2"):
                    self.assertTrue(record.get("merge.footprint.2"))
                    self.assertTrue(isPeakInCatalog(peak, self.catalog2))
                elif peak.get("merge.peak.3"):
                    self.assertTrue(record.get("merge.footprint.3"))
                    self.assertTrue(isPeakInCatalog(peak, self.catalog3))
                else:
                    self.fail("At least one merge.peak flag must be set")

        # Add all the catalogs with minPeak = 100 so no new peaks will be added
        merge, nob, npeak = mergeCatalogs([self.catalog1, self.catalog2, self.catalog3],
                                          ["1", "2", "3"], 100, self.idFactory)
        self.assertEqual(nob, 19)
        self.assertEqual(npeak, 20)
        measArea = [i.getFootprint().getArea() for i in merge]
        self.assert_(np.all(pixArea == measArea))

        for record in merge:
            for peak in record.getFootprint().getPeaks():
                if peak.get("merge.peak.1"):
                    self.assertTrue(record.get("merge.footprint.1"))
                    self.assertTrue(isPeakInCatalog(peak, self.catalog1))
                elif peak.get("merge.peak.2"):
                    self.assertTrue(record.get("merge.footprint.2"))
                    self.assertTrue(isPeakInCatalog(peak, self.catalog2))
                elif peak.get("merge.peak.3"):
                    self.assertTrue(record.get("merge.footprint.3"))
                    self.assertTrue(isPeakInCatalog(peak, self.catalog3))
                else:
                    self.fail("At least one merge.peak flag must be set")

        # Add footprints with large samePeakDist so that any footprint that merges will also
        # have the peak flagged
        merge, nob, npeak = mergeCatalogs([self.catalog1, self.catalog2, self.catalog3],
                                          ["1", "2", "3"], 100, self.idFactory, samePeakDist=40)

        # peaks detected in more than one catalog
        multiPeakIndex = [0, 2, 5, 7, 9]
        peakIndex = 0
        for record in merge:
            for peak in record.getFootprint().getPeaks():
                numPeak = np.sum([peak.get("merge.peak.1"),peak.get("merge.peak.2"),
                                  peak.get("merge.peak.3")])
                if peakIndex in multiPeakIndex:
                    self.assertTrue(numPeak > 1)
                else:
                    self.assertTrue(numPeak == 1)
                peakIndex += 1

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    tests.init()

    suites = []
    suites += unittest.makeSuite(FootprintMergeCatalogTestCase)
    suites += unittest.makeSuite(tests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit=False):
    """Run the tests"""
    tests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)
