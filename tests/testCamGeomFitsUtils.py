#pybind11##!/usr/bin/env python
#pybind11#from __future__ import absolute_import, division
#pybind11#from builtins import object
#pybind11##
#pybind11## LSST Data Management System
#pybind11## Copyright 2015 LSST Corporation.
#pybind11##
#pybind11## This product includes software developed by the
#pybind11## LSST Project (http://www.lsst.org/).
#pybind11##
#pybind11## This program is free software: you can redistribute it and/or modify
#pybind11## it under the terms of the GNU General Public License as published by
#pybind11## the Free Software Foundation, either version 3 of the License, or
#pybind11## (at your option) any later version.
#pybind11##
#pybind11## This program is distributed in the hope that it will be useful,
#pybind11## but WITHOUT ANY WARRANTY; without even the implied warranty of
#pybind11## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#pybind11## GNU General Public License for more details.
#pybind11##
#pybind11## You should have received a copy of the LSST License Statement and
#pybind11## the GNU General Public License along with this program.  If not,
#pybind11## see <http://www.lsstcorp.org/LegalNotices/>.
#pybind11##
#pybind11#"""
#pybind11#Tests for lsst.afw.cameraGeom.Detector
#pybind11#"""
#pybind11#import unittest
#pybind11#
#pybind11#import lsst.utils.tests
#pybind11#import lsst.pex.exceptions
#pybind11#import lsst.daf.base as dafBase
#pybind11#import lsst.afw.image as afwImage
#pybind11#from lsst.afw.cameraGeom.fitsUtils import getByKey, setByKey, HeaderAmpMap, HeaderDetectorMap, DetectorBuilder
#pybind11#
#pybind11#
#pybind11#class AmpTestObject(object):
#pybind11#
#pybind11#    def name(self, a):
#pybind11#        self.name = a
#pybind11#
#pybind11#    def testsec(self, b):
#pybind11#        self.testsec = b
#pybind11#
#pybind11#    def defaultval(self, c):
#pybind11#        self.defaultval = c
#pybind11#
#pybind11#
#pybind11#class DetTestObject(object):
#pybind11#
#pybind11#    def __init__(self):
#pybind11#        return
#pybind11#
#pybind11#
#pybind11#class FitsUtilsTestCase(unittest.TestCase):
#pybind11#
#pybind11#    def toTestStr(self, string):
#pybind11#        return "Test String"
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        self.metadata = dafBase.PropertySet()
#pybind11#        self.metadata.set('HELLO', 'hello')
#pybind11#        self.metadata.set('NAME', 'Presto')
#pybind11#        self.metadata.set('COPYOVER', 'copy over')
#pybind11#        self.metadata.set('DONTCLOBBER', 'do not clobber')
#pybind11#        self.metadata.set('TESTSEC', '1 2 3 4 5')
#pybind11#        exp = afwImage.ExposureF(10, 10)
#pybind11#        exp.setMetadata(self.metadata)
#pybind11##        exp.writeFits('testfitsutils.fits')
#pybind11#        self.exposure = exp
#pybind11#        self.mdMapList = [('NAME', 'name'),
#pybind11#                          ('TESTSEC', 'testsec', None, self.toTestStr),
#pybind11#                          ('TESTDEF', 'defaultval', 'Default', None)]
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        del self.metadata
#pybind11#        del self.mdMapList
#pybind11#        del self.exposure
#pybind11##        os.remove('testfitsutils.fits')
#pybind11#
#pybind11#    def testBasics(self):
#pybind11#        """Test getters and other basics
#pybind11#        """
#pybind11#        self.assertEqual(getByKey(self.metadata, 'HELLO'), 'hello')
#pybind11#        self.assertIsNone(getByKey(self.metadata, 'NOTAKEY'))
#pybind11#        setByKey(self.metadata, 'NEWKEY', 'new key', False)
#pybind11#        self.assertEqual(getByKey(self.metadata, 'NEWKEY'), 'new key')
#pybind11#        setByKey(self.metadata, 'DONTCLOBBER', 'clobbered', False)
#pybind11#        self.assertNotEqual(getByKey(self.metadata, 'DONTCLOBBER'), 'clobbered')
#pybind11#        setByKey(self.metadata, 'COPYOVER', 'replaced', True)
#pybind11#        self.assertEqual(getByKey(self.metadata, 'COPYOVER'), 'replaced')
#pybind11#
#pybind11#    def testMapper(self):
#pybind11#        """Test mapper
#pybind11#        """
#pybind11#        tao = AmpTestObject()
#pybind11#        ham = HeaderAmpMap()
#pybind11#        for tup in self.mdMapList:
#pybind11#            ham.addEntry(*tup)
#pybind11#        ham.setAttributes(tao, self.metadata)
#pybind11#        self.assertEqual(tao.name, self.metadata.get('NAME'))
#pybind11#        self.assertEqual(tao.testsec, 'Test String')
#pybind11#        self.assertEqual(tao.defaultval, 'Default')
#pybind11#
#pybind11#        tdo = DetTestObject()
#pybind11#        hdm = HeaderDetectorMap()
#pybind11#        for tup in self.mdMapList:
#pybind11#            hdm.addEntry(*tup)
#pybind11#        hdm.setAttributes(tdo, self.metadata)
#pybind11#        self.assertEqual(tdo.name, self.metadata.get('NAME'))
#pybind11#        self.assertEqual(tdo.testsec, 'Test String')
#pybind11#        self.assertEqual(tdo.defaultval, 'Default')
#pybind11#
#pybind11#    def testDetectorBuilder(self):
#pybind11#        """Test the buildDetector method
#pybind11#           Just tests whether the constructors return without error.  The internals are tested above.
#pybind11#        """
#pybind11#        with lsst.utils.tests.getTempFilePath(".fits") as tmpFile:
#pybind11#            self.exposure.writeFits(tmpFile)
#pybind11#            # test that it raises without setting non-defaulted keywords
#pybind11#            self.assertRaises(Exception, DetectorBuilder(tmpFile, [tmpFile, tmpFile]))
#pybind11#            # ignore non-defaulted keywords
#pybind11#            detBuilder = DetectorBuilder(tmpFile, [tmpFile, ], doRaise=False)
#pybind11#            detBuilder.makeCalib()
#pybind11#            detBuilder.makeExposure(afwImage.ImageF(10, 10), afwImage.MaskU(10, 10), afwImage.ImageF(10, 10))
#pybind11#
#pybind11#
#pybind11#class TestMemory(lsst.utils.tests.MemoryTestCase):
#pybind11#    pass
#pybind11#
#pybind11#def setup_module(module):
#pybind11#    lsst.utils.tests.init()
#pybind11#
#pybind11#if __name__ == "__main__":
#pybind11#    lsst.utils.tests.init()
#pybind11#    unittest.main()
