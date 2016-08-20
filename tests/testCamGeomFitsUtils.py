#!/usr/bin/env python
from __future__ import absolute_import, division
from builtins import object
#
# LSST Data Management System
# Copyright 2015 LSST Corporation.
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
Tests for lsst.afw.cameraGeom.Detector
"""
import unittest

import lsst.utils.tests
import lsst.pex.exceptions
import lsst.daf.base as dafBase
import lsst.afw.image as afwImage
from lsst.afw.cameraGeom.fitsUtils import getByKey, setByKey, HeaderAmpMap, HeaderDetectorMap, DetectorBuilder


class AmpTestObject(object):

    def name(self, a):
        self.name = a

    def testsec(self, b):
        self.testsec = b

    def defaultval(self, c):
        self.defaultval = c


class DetTestObject(object):

    def __init__(self):
        return


class FitsUtilsTestCase(unittest.TestCase):

    def toTestStr(self, string):
        return "Test String"

    def setUp(self):
        self.metadata = dafBase.PropertySet()
        self.metadata.set('HELLO', 'hello')
        self.metadata.set('NAME', 'Presto')
        self.metadata.set('COPYOVER', 'copy over')
        self.metadata.set('DONTCLOBBER', 'do not clobber')
        self.metadata.set('TESTSEC', '1 2 3 4 5')
        exp = afwImage.ExposureF(10, 10)
        exp.setMetadata(self.metadata)
#        exp.writeFits('testfitsutils.fits')
        self.exposure = exp
        self.mdMapList = [('NAME', 'name'),
                          ('TESTSEC', 'testsec', None, self.toTestStr),
                          ('TESTDEF', 'defaultval', 'Default', None)]

    def tearDown(self):
        del self.metadata
        del self.mdMapList
        del self.exposure
#        os.remove('testfitsutils.fits')

    def testBasics(self):
        """Test getters and other basics
        """
        self.assertEqual(getByKey(self.metadata, 'HELLO'), 'hello')
        self.assertTrue(getByKey(self.metadata, 'NOTAKEY') is None)
        setByKey(self.metadata, 'NEWKEY', 'new key', False)
        self.assertEqual(getByKey(self.metadata, 'NEWKEY'), 'new key')
        setByKey(self.metadata, 'DONTCLOBBER', 'clobbered', False)
        self.assertNotEqual(getByKey(self.metadata, 'DONTCLOBBER'), 'clobbered')
        setByKey(self.metadata, 'COPYOVER', 'replaced', True)
        self.assertEqual(getByKey(self.metadata, 'COPYOVER'), 'replaced')

    def testMapper(self):
        """Test mapper
        """
        tao = AmpTestObject()
        ham = HeaderAmpMap()
        for tup in self.mdMapList:
            ham.addEntry(*tup)
        ham.setAttributes(tao, self.metadata)
        self.assertEqual(tao.name, self.metadata.get('NAME'))
        self.assertEqual(tao.testsec, 'Test String')
        self.assertEqual(tao.defaultval, 'Default')

        tdo = DetTestObject()
        hdm = HeaderDetectorMap()
        for tup in self.mdMapList:
            hdm.addEntry(*tup)
        hdm.setAttributes(tdo, self.metadata)
        self.assertEqual(tdo.name, self.metadata.get('NAME'))
        self.assertEqual(tdo.testsec, 'Test String')
        self.assertEqual(tdo.defaultval, 'Default')

    def testDetectorBuilder(self):
        """Test the buildDetector method
           Just tests whether the constructors return without error.  The internals are tested above.
        """
        with lsst.utils.tests.getTempFilePath(".fits") as tmpFile:
            self.exposure.writeFits(tmpFile)
            # test that it raises without setting non-defaulted keywords
            self.assertRaises(Exception, DetectorBuilder(tmpFile, [tmpFile, tmpFile]))
            # ignore non-defaulted keywords
            detBuilder = DetectorBuilder(tmpFile, [tmpFile, ], doRaise=False)
            detBuilder.makeCalib()
            detBuilder.makeExposure(afwImage.ImageF(10, 10), afwImage.MaskU(10, 10), afwImage.ImageF(10, 10))


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass

def setup_module(module):
    lsst.utils.tests.init()

if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()