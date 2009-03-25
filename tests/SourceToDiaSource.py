#!/usr/bin/env python
"""
Tests for C++ Source and SourceVector Python wrappers (including persistence)

Run with:
   python Source_1.py
or
   python
   >>> import unittest; T=load("Source_1"); unittest.TextTestRunner(verbosity=1).run(T.suite())
"""
import pdb
import unittest
import random
import time

import lsst.daf.base as dafBase
import lsst.pex.policy as dafPolicy
import lsst.daf.persistence as dafPers
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
