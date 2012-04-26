import os, sys, unittest
import lsst.utils.tests as utilsTests
import lsst.afw.detection as afwDet
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.afw.geom as afwGeom

class SourceHeavyFootprintTestCase(unittest.TestCase):
    def test1(self):
        im = afwImage.ImageF(100, 100)
        fp = afwDet.Footprint(afwGeom.Point2I(50,50), 10.)
        #seed = 42
        #rand = afwMath.Random(afwMath.Random.MT19937, seed)
        #afwMath.randomGaussianImage(im, rand)
        mi = afwImage.MaskedImageF(im)
        heavy = afwDet.makeHeavyFootprint(fp, mi)
        schema = afwTable.SourceTable.makeMinimalSchema()
        table = afwTable.SourceTable.make(schema)
        table.preallocate(10)
        catalog = afwTable.SourceCatalog(table)
        catalog.addNew()
        # This used to segfault
        catalog[0].setFootprint(heavy)


            
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(SourceHeavyFootprintTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit = False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)

        
        

        
