#!/usr/bin/env python

# 
# LSST Data Management System
# Copyright 2008, 2009, 2010 LSST Corporation.
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
Tests for C++ Source and SourceVector Python wrappers (including persistence)

Run with:
   python Source_1.py
or
   python
   >>> import unittest; T=load("Source_1"); unittest.TextTestRunner(verbosity=1).run(T.suite())
"""

import unittest

import lsst.utils.tests as utilsTests
import lsst.afw.detection as afwDet
import lsst.afw.geom as afwGeom

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class SourceToDiaSourceTestCase(unittest.TestCase):
    """A test case for converting Sources to DiaSources"""
    def setUp(self):

        R = afwGeom.radians
        self.methods = {
            "Id" : 3,
            "Ra": 4* R, "Dec" : 2* R,
            "XFlux" : 1.0, "YFlux" : 1.0,
            "RaFlux" : 1.0* R, "DecFlux" : 1.0* R,
            "XPeak" : 1.0, "YPeak" : 1.0,
            "RaPeak" : 1.0* R, "DecPeak" : 1.0* R,
            "XAstrom" : 1.0, "YAstrom" : 1.0,
            "RaAstrom" : 1.0* R, "DecAstrom" : 1.0* R,
            "PsfFlux" : 1.0, "ApFlux" : 2.0,
            "Ixx" : 0.3, "Iyy" : 0.4, "Ixy" : 0.5,
            "PsfIxx" : 0.3, "PsfIyy" : 0.3, "PsfIxy" : 0.3,
            "E1" : 0.3, "E1Err" : 0.3, "E2" : 0.4, "E2Err" : 0.5,
            "Shear1" : 0.3, "Shear1Err" : 0.3, "Shear2" : 0.4, "Shear2Err" : 0.5,
            "Sigma" : 0.5, "SigmaErr" : 0.6,
            "Resolution" : 0.5,
            }
        
        self.source = afwDet.Source()
        for k, v in self.methods.items():
            method = getattr(self.source, "set"+k)
            method(v)

    def tearDown(self):
        del self.source
   
   
    def testMake(self):
        diaSource = afwDet.makeDiaSourceFromSource(self.source)

        for k in self.methods.keys():
            diaSrcVal = getattr(diaSource, "get"+k)()
            srcVal = getattr(self.source, "get"+k)()
            print k, diaSrcVal, srcVal
            assert(diaSrcVal == srcVal)
     
def suite():
    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(SourceToDiaSourceTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

if __name__ == "__main__":
    utilsTests.run(suite())
