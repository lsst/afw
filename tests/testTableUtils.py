#!/usr/bin/env python

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

import unittest
import numpy

import lsst.utils.tests as utilsTests
import lsst.afw.table as afwTable
import lsst.afw.geom as afwGeom
import lsst.afw.coord as afwCoord
import lsst.afw.image as afwImage

class TableUtilsTestCase(unittest.TestCase):
        
    def testSourceRecalibrate(self):
        """Test that sourceRecalibrate correctly changes a simple shape in a source catalog"""
        
        # create a source catalog with enough defined for a shape and position
        schema = afwTable.SourceTable.makeMinimalSchema()
        pkey   = schema.addField("point", type="PointD")
        mkey   = schema.addField("moment", type="MomentsD")
        
        table  = afwTable.SourceTable.make(schema)
        table.defineCentroid("point")
        table.defineShape("moment")
        
        sources = afwTable.SourceCatalog(table)

        # add 1 source at some arbitrary location
        s = sources.addNew()
        s.setId(1)
        s.setPointD(pkey, afwGeom.PointD(1000.0, 1000.0))
        s.setMomentsD(mkey, afwGeom.ellipses.Quadrupole(1.0, 2.0, 0.0))
        
        # use 1 arcsec pixels with a flipped WCS ... no cross terms
        pixScale = 1.0*afwGeom.arcseconds
        crval    = afwCoord.Coord(afwGeom.Point2D(numpy.pi/2.0, numpy.pi/4.0))
        crpix    = afwGeom.Point2D(1000.0, 1000.0)
        wcs      = afwImage.makeWcs(crval, crpix, 0.0, -pixScale.asDegrees(), pixScale.asDegrees(), 0.0)

        # recalibrate the src to align with
        shapesToConvert = 'moment',
        sources2  = afwTable.sourceRecalibrate(sources, wcs, shapesToConvert=shapesToConvert)

        # the result should have flipped ixx and iyy and left ixy alone
        s, s2 = sources[0], sources2[0]
        print "shape before: ", s.getShape()
        print "shape after:  ", s2.getShape()
        self.assertAlmostEqual(s.getIxx(), s2.getIyy())
        self.assertAlmostEqual(s.getIyy(), s2.getIxx())
        self.assertAlmostEqual(s.getIxy(), s2.getIxy(), 5)

            

def suite():
    """Returns a suite containing all the test cases in this module."""
    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(TableUtilsTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(exit=False):
    """Run the utilsTests"""
    utilsTests.run(suite(), exit)

if __name__ == "__main__":
    run(True)
