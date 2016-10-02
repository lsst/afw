#pybind11#from builtins import range
#pybind11#import unittest
#pybind11#
#pybind11#import lsst.utils.tests
#pybind11#import lsst.afw.geom as afwGeom
#pybind11#from lsst.afw.geom.testUtils import BoxGrid
#pybind11#
#pybind11#
#pybind11#class BoxGridTestCase(lsst.utils.tests.TestCase):
#pybind11#    """!Unit tests for BoxGrid"""
#pybind11#
#pybind11#    def test3By4(self):
#pybind11#        """!Test a 3x4 box divided into a 3x2 grid, such that each sub-box is 1x2
#pybind11#        """
#pybind11#        for boxClass in (afwGeom.Box2I, afwGeom.Box2D):
#pybind11#            pointClass = type(boxClass().getMin())
#pybind11#            extentClass = type(boxClass().getDimensions())
#pybind11#
#pybind11#            minPt = pointClass(-1, 3)
#pybind11#            extent = extentClass(3, 4)
#pybind11#            numColRow = (3, 2)
#pybind11#            outerBox = boxClass(minPt, extent)
#pybind11#            boxGrid = BoxGrid(box=outerBox, numColRow=numColRow)
#pybind11#            for box in boxGrid:
#pybind11#                self.assertEqual(box.getDimensions(), extentClass(1, 2))
#pybind11#            for row in range(numColRow[1]):
#pybind11#                for col in range(numColRow[0]):
#pybind11#                    box = boxGrid[col, row]
#pybind11#                    predMin = outerBox.getMin() + extentClass(col*1, row*2)
#pybind11#                    self.assertEqual(box.getMin(), predMin)
#pybind11#
#pybind11#    def testIntUneven(self):
#pybind11#        """!Test dividing an integer box into an uneven grid
#pybind11#
#pybind11#        Divide a 5x4 box into 2x3 regions
#pybind11#        """
#pybind11#        minPt = afwGeom.Point2I(0, 0)
#pybind11#        extent = afwGeom.Extent2I(5, 7)
#pybind11#        numColRow = (2, 3)
#pybind11#        outerBox = afwGeom.Box2I(minPt, extent)
#pybind11#        boxGrid = BoxGrid(box=outerBox, numColRow=numColRow)
#pybind11#        desColStarts = (0, 2)
#pybind11#        desWidths = (2, 3)
#pybind11#        desRowStarts = (0, 2, 4)
#pybind11#        desHeights = (2, 2, 3)
#pybind11#        for row in range(numColRow[1]):
#pybind11#            desRowStart = desRowStarts[row]
#pybind11#            desHeight = desHeights[row]
#pybind11#            for col in range(numColRow[0]):
#pybind11#                desColStart = desColStarts[col]
#pybind11#                desWidth = desWidths[col]
#pybind11#                box = boxGrid[col, row]
#pybind11#                self.assertEqual(tuple(box.getMin()), (desColStart, desRowStart))
#pybind11#                self.assertEqual(tuple(box.getDimensions()), (desWidth, desHeight))
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
