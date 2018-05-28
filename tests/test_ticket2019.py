
import unittest

import lsst.utils.tests
import lsst.afw.detection as afwDet
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.afw.geom as afwGeom


class SourceHeavyFootprintTestCase(unittest.TestCase):

    def test1(self):
        im = afwImage.ImageF(100, 100)
        im += 42.
        spanSet = afwGeom.SpanSet.fromShape(10).shiftedBy(50, 50)
        fp = afwDet.Footprint(spanSet)
        mi = afwImage.MaskedImageF(im)
        # set a mask bit before grabbing the heavyfootprint
        mi.mask[50, 50, afwImage.LOCAL] = 1
        heavy = afwDet.makeHeavyFootprint(fp, mi)
        # reset it
        mi.mask[50, 50, afwImage.LOCAL] = 0

        schema = afwTable.SourceTable.makeMinimalSchema()
        table = afwTable.SourceTable.make(schema)
        table.preallocate(10)
        catalog = afwTable.SourceCatalog(table)
        catalog.addNew()
        # This used to segfault
        catalog[0].setFootprint(heavy)

        fp = catalog[0].getFootprint()
        # change one pixel...
        self.assertEqual(mi.image[50, 50, afwImage.LOCAL], 42)
        self.assertEqual(mi.mask[50, 50, afwImage.LOCAL], 0)
        mi.image[50, 50, afwImage.LOCAL] = 100
        mi.mask[50, 50, afwImage.LOCAL] = 2
        mi.mask[51, 50, afwImage.LOCAL] = 2
        self.assertEqual(mi.image[50, 50, afwImage.LOCAL], 100)
        self.assertEqual(mi.mask[50, 50, afwImage.LOCAL], 2)
        self.assertEqual(mi.mask[51, 50, afwImage.LOCAL], 2)
        # reinsert the heavy footprint; it should reset the pixel value.
        # insert(MaskedImage)
        fp.insert(mi)
        self.assertEqual(mi.image[50, 50, afwImage.LOCAL], 42)
        self.assertEqual(mi.mask[50, 50, afwImage.LOCAL], 1)
        self.assertEqual(mi.mask[51, 50, afwImage.LOCAL], 0)

        # Also test insert(Image)
        im = mi.image
        self.assertEqual(im[50, 50, afwImage.LOCAL], 42)
        im[50, 50, afwImage.LOCAL] = 100
        self.assertEqual(im[50, 50, afwImage.LOCAL], 100)
        self.assertEqual(mi.image[50, 50, afwImage.LOCAL], 100)
        # reinsert the heavy footprint; it should reset the pixel value.
        fp.insert(im)
        self.assertEqual(im[50, 50, afwImage.LOCAL], 42)
        self.assertEqual(mi.image[50, 50, afwImage.LOCAL], 42)
        self.assertEqual(mi.mask[50, 50, afwImage.LOCAL], 1)
        self.assertEqual(mi.mask[51, 50, afwImage.LOCAL], 0)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
