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

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class SourceToDiaSourceTestCase(unittest.TestCase):
    """A test case for converting Sources to DiaSources"""
    def setUp(self):
        self.source = afwDet.Source()

        self.source.setRa(4)
        self.source.setId(3)

    def tearDown(self):
        del self.source
   
   
    def testMake(self):
        diaSource = afwDet.makeDiaSourceFromSource(self.source)
        assert(diaSource.getId() == self.source.getId())
        assert(diaSource.getRa() == self.source.getRa())
     
def suite():
    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(SourceToDiaSourceTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

if __name__ == "__main__":
    utilsTests.run(suite())
