
#
# Copyright 2008-2017  AURA/LSST.
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
# see <https://www.lsstcorp.org/LegalNotices/>.
#

from __future__ import absolute_import, division, print_function
import unittest
import tempfile

import lsst.utils.tests
import lsst.afw.geom as afwGeom
import lsst.afw.detection as afwDet
import lsst.afw.table as afwTable


class FootprintTestCase(unittest.TestCase):
    def setUp(self):
        self.spanRad = 4
        self.regionRad = 10
        self.spans = afwGeom.SpanSet.fromShape(self.spanRad,
                                                      afwGeom.Stencil.BOX)
        minPoint = afwGeom.Point2I(-self.regionRad, -self.regionRad)
        maxPoint = afwGeom.Point2I(self.regionRad, self.regionRad)
        self.region = afwGeom.Box2I(minPoint, maxPoint)
        self.schema = afwDet.PeakTable.makeMinimalSchema()
        # Run the the constructors test to ensure the Footprints are setUp
        self.testConstructors()

    def tearDown(self):
        del self.spans
        del self.region
        del self.schema
        del self.footprint
        del self.footprintWithRegion
        del self.footprintWithSchema
        del self.footprintWithSchemaRegion
        del self.emptyFootprint

    def testConstructors(self):
        '''
        Test that each of the constructors constructs a valid Footprint,
        if any of these fails, an exception will be raised and the test
        will fail.
        '''
        self.footprint = afwDet.Footprint(self.spans)
        self.footprintWithRegion = afwDet.Footprint(self.spans, self.region)
        self.footprintWithSchema = afwDet.Footprint(self.spans, self.schema)
        self.footprintWithSchemaRegion = afwDet.Footprint(self.spans,
                                                          self.schema,
                                                          self.region)
        self.emptyFootprint = afwDet.Footprint()
        self.assertEqual(len(self.emptyFootprint.spans), 0)
        self.assertEqual(len(self.emptyFootprint.peaks), 0)

    def testIsHeavy(self):
        self.assertFalse(self.footprint.isHeavy())

    def testGetSetSpans(self):
        '''
        Test getting and setting the SpanSet member of the Footprint with both
        the getters and setters and the python property accessor
        '''
        self.assertEqual(self.footprint.getSpans(), self.spans)
        self.assertEqual(self.footprint.spans, self.spans)
        tempSpanSet = afwGeom.SpanSet.fromShape(2, afwGeom.Stencil.BOX)
        self.footprint.setSpans(tempSpanSet)
        self.assertEqual(self.footprint.spans, tempSpanSet)
        # reset back to original with property
        self.footprint.spans = self.spans
        self.assertEqual(self.footprint.spans, self.spans)

    def testPeakFunctionality(self):
        newSchema = afwDet.PeakTable.makeMinimalSchema()
        newField = afwTable.FieldI("extra", "doc", "na")
        newSchema.addField(newField)
        self.footprint.setPeakSchema(newSchema)
        names = self.footprint.getPeaks().getSchema().getNames()
        self.assertIn("extra", names)
        # reset the schema back
        self.footprint.setPeakSchema(self.schema)
        peakData = [[2, 2, 10], [0, 3, 21], [1, 9, 17]]
        for peak in peakData:
            self.footprint.addPeak(*peak)
        # Sort the peaks by value (use the property peaks to test that method
        # of access)
        sortKey = self.footprint.peaks.getSchema()['peakValue'].asKey()
        self.footprint.sortPeaks(sortKey)
        for i, peak in enumerate(self.footprint.peaks):
            self.assertEqual(peak['i_x'], i)

        # Test that peaks outside the Footprint are removed
        self.footprint.removeOrphanPeaks()
        self.assertEqual(len(self.footprint.peaks), 2)
        for peak in self.footprint.peaks:
            self.assertNotEqual(peak['i_x'], 1)

    def testGeometry(self):
        # Move the base footprint by 2 in x and 2 in y
        offsetX = 2
        offsetY = -3
        self.footprint.shift(offsetX, offsetY)
        # verify that this shifts the center from 0,0 as the default
        # constructed footprint has
        center = self.footprint.getCentroid()
        self.assertEqual(center.getX(), offsetX)
        self.assertEqual(center.getY(), offsetY)

        shape = 6.66666
        covShape = 0
        places = 4
        self.assertAlmostEqual(self.footprint.getShape().getIxx(),
                               shape, places)
        self.assertAlmostEqual(self.footprint.getShape().getIyy(),
                               shape, places)
        self.assertEqual(self.footprint.getShape().getIxy(), covShape)

        # Shift the footprint back
        self.footprint.shift(afwGeom.ExtentI(-offsetX, -offsetY))

        bBox = self.footprint.getBBox()
        self.assertEqual(bBox.getMinX(), -self.spanRad)
        self.assertEqual(bBox.getMinY(), -self.spanRad)
        self.assertEqual(bBox.getMaxX(), self.spanRad)
        self.assertEqual(bBox.getMaxY(), self.spanRad)

        # Test the point membership in a Footprint
        memberPoint = afwGeom.Point2I(1, 1)
        self.assertTrue(self.footprint.contains(memberPoint))
        self.assertIn(memberPoint, self.footprint)

        nonMemberPoint = afwGeom.Point2I(100, 100)
        self.assertFalse(self.footprint.contains(nonMemberPoint))
        self.assertNotIn(nonMemberPoint, self.footprint)

    def testRegion(self):
        self.assertEqual(self.footprintWithRegion.getRegion(), self.region)
        largeRad = 20
        testRegion = afwGeom.Box2I(afwGeom.Point2I(-largeRad, -largeRad),
                                   afwGeom.Point2I(largeRad, largeRad))
        self.footprintWithRegion.setRegion(testRegion)
        self.assertEqual(testRegion, self.footprintWithRegion.getRegion())

    def testMutationFunctionality(self):
        clipRad = 2
        clipBox = afwGeom.Box2I(afwGeom.Point2I(-clipRad, -clipRad),
                                afwGeom.Point2I(clipRad, clipRad))
        self.footprint.clipTo(clipBox)
        # Fetch the bounding box using the property notation
        bBox = self.footprint.getBBox()
        # Check the bounding box is now at the bounds which were clipped to
        self.assertEqual(bBox.getMinX(), -clipRad)
        self.assertEqual(bBox.getMinY(), -clipRad)
        self.assertEqual(bBox.getMaxX(), clipRad)
        self.assertEqual(bBox.getMaxY(), clipRad)

        # Set the footprint back to what it was
        self.footprint = afwDet.Footprint(self.spans)

        # Test erode
        kernelRad = 1
        kernel = afwGeom.SpanSet.fromShape(kernelRad,
                                                  afwGeom.Stencil.BOX)
        self.footprint.erode(kernel)

        # Verify the eroded dimensions
        bBox = self.footprint.getBBox()
        self.assertEqual(bBox.getMinX(), -3)
        self.assertEqual(bBox.getMinY(), -3)
        self.assertEqual(bBox.getMaxX(), 3)
        self.assertEqual(bBox.getMaxY(), 3)

        # Dilate the footprint back to the origional
        self.footprint.dilate(kernel)
        self.assertEqual(self.footprint.spans, self.spans)

        # erode using the alternate call syntax
        self.footprint.erode(1, afwGeom.Stencil.BOX)

        # verify the eroded dimensions
        bBox = self.footprint.getBBox()
        self.assertEqual(bBox.getMinX(), -3)
        self.assertEqual(bBox.getMinY(), -3)
        self.assertEqual(bBox.getMaxX(), 3)
        self.assertEqual(bBox.getMaxY(), 3)

        # Dilate the footprint back to the origional using alternate signature
        self.footprint.dilate(1, afwGeom.Stencil.BOX)
        self.assertEqual(self.footprint.spans, self.spans)

    def testSplit(self):
        spanList = [afwGeom.Span(0, 2, 4),
                    afwGeom.Span(1, 2, 4),
                    afwGeom.Span(2, 2, 4),
                    afwGeom.Span(10, 4, 7),
                    afwGeom.Span(11, 4, 7),
                    afwGeom.Span(12, 4, 7)]

        spans = afwGeom.SpanSet(spanList)
        region = afwGeom.Box2I(afwGeom.PointI(-6, -6), afwGeom.PointI(20, 20))
        multiFoot = afwDet.Footprint(spans, region)

        records = [multiFoot.addPeak(3, 1, 100),
                   multiFoot.addPeak(5, 11, 100)]

        # Verify that the footprint is multi-component
        self.assertFalse(multiFoot.isContiguous())

        footprintList = multiFoot.split()

        self.assertEqual(len(footprintList), 2)
        for i, fp in enumerate(footprintList):
            # check that the correct Spans are populated for each
            tempSpan = afwGeom.SpanSet(spanList[i*3:i*3+3])
            self.assertEqual(fp.spans, tempSpan)

            # check that the peaks are split properly
            self.assertEqual(len(fp.peaks), 1)
            self.assertEqual(fp.peaks[0], records[i])

    def testPersistence(self):
        # populate the peaks for the peak tests
        self.testPeakFunctionality()

        with tempfile.NamedTemporaryFile() as f:
            # Persist the Footprint to file and read it back
            self.footprint.writeFits(f.name)
            footprintFromFile = afwDet.Footprint.readFits(f.name)

        # Check that the Footprint before and after saving are the same
        self.assertEqual(self.footprint, footprintFromFile)

        # Clean up after ourselves
        del footprintFromFile

    def testLegacyFootprints(self):
        fileName = 'tests/data/preSpanSetsFootprint.fits'
        legacyFootprint = afwDet.Footprint.readFits(fileName)

        # Calculate some quantifying numbers from the legacy Footprint to ensure
        # it loaded properly
        self.assertEqual(len(legacyFootprint.spans), 303)
        self.assertEqual(len(legacyFootprint.peaks), 48)
        self.assertEqual(legacyFootprint.spans.getBBox(),
                         afwGeom.Box2I(afwGeom.Point2I(32676, 27387),
                                       afwGeom.Extent2I(175, 153)))
        legacyCenter = legacyFootprint.spans.computeCentroid()
        self.assertAlmostEqual(legacyCenter.getY(), 27456.70733, 5)
        self.assertAlmostEqual(legacyCenter.getX(), 32775.47611, 5)

        del legacyFootprint


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
