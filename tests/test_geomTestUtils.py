import unittest

import lsst.utils.tests
import lsst.afw.geom as afwGeom
from lsst.afw.geom.testUtils import BoxGrid


class BoxGridTestCase(lsst.utils.tests.TestCase):
    """!Unit tests for BoxGrid"""

    def test3By4(self):
        """!Test a 3x4 box divided into a 3x2 grid, such that each sub-box is 1x2
        """
        for boxClass in (afwGeom.Box2I, afwGeom.Box2D):
            pointClass = type(boxClass().getMin())
            extentClass = type(boxClass().getDimensions())

            minPt = pointClass(-1, 3)
            extent = extentClass(3, 4)
            numColRow = (3, 2)
            outerBox = boxClass(minPt, extent)
            boxGrid = BoxGrid(box=outerBox, numColRow=numColRow)
            for box in boxGrid:
                self.assertEqual(box.getDimensions(), extentClass(1, 2))
            for row in range(numColRow[1]):
                for col in range(numColRow[0]):
                    box = boxGrid[col, row]
                    predMin = outerBox.getMin() + extentClass(col*1, row*2)
                    self.assertEqual(box.getMin(), predMin)

    def testIntUneven(self):
        """!Test dividing an integer box into an uneven grid

        Divide a 5x4 box into 2x3 regions
        """
        minPt = afwGeom.Point2I(0, 0)
        extent = afwGeom.Extent2I(5, 7)
        numColRow = (2, 3)
        outerBox = afwGeom.Box2I(minPt, extent)
        boxGrid = BoxGrid(box=outerBox, numColRow=numColRow)
        desColStarts = (0, 2)
        desWidths = (2, 3)
        desRowStarts = (0, 2, 4)
        desHeights = (2, 2, 3)
        for row in range(numColRow[1]):
            desRowStart = desRowStarts[row]
            desHeight = desHeights[row]
            for col in range(numColRow[0]):
                desColStart = desColStarts[col]
                desWidth = desWidths[col]
                box = boxGrid[col, row]
                self.assertEqual(tuple(box.getMin()),
                                 (desColStart, desRowStart))
                self.assertEqual(tuple(box.getDimensions()),
                                 (desWidth, desHeight))


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
