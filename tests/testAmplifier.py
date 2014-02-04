#!/usr/bin/env python2
from __future__ import absolute_import, division
# 
# LSST Data Management System
# Copyright 2014 LSST Corporation.
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
Tests for lsst.afw.cameraGeom.Amplifier
"""
import unittest

import lsst.utils.tests
from lsst.pex.exceptions import LsstCppException
import lsst.afw.geom as afwGeom
import lsst.afw.cameraGeom as cameraGeom

class AmplifierWrapper(object):
    def __init__(self, name, doRawAmp=True, doBadBBox=False):
        """Construct an Amplifier
        """
        self.name = name
        self.bbox = afwGeom.Box2I(afwGeom.Point2I(-1, 1), afwGeom.Extent2I(123, 307))
        if doBadBBox:
            self.bbox.grow(afwGeom.Extent2I(1, 0))
        self.gain = 1.71234e3
        self.readNoise = 0.521237e2
        if doRawAmp:
            self.rawAmplifier = cameraGeom.RawAmplifier(
                afwGeom.Box2I(afwGeom.Point2I(-25, 2), afwGeom.Extent2I(550, 629)),
                afwGeom.Box2I(afwGeom.Point2I(-2, 29), afwGeom.Extent2I(123, 307)),
                afwGeom.Box2I(afwGeom.Point2I(150, 29), afwGeom.Extent2I(25, 307)),
                afwGeom.Box2I(afwGeom.Point2I(-2, 201), afwGeom.Extent2I(123, 6)),
                afwGeom.Box2I(afwGeom.Point2I(-20, 2), afwGeom.Extent2I(5, 307)),
                False,
                True,
                afwGeom.Extent2I(-97, 253),
            )
        else:
            self.rawAmplifier = None
        self.amplifier = cameraGeom.Amplifier(
            self.name,
            self.bbox,
            self.gain,
            self.readNoise,
            self.rawAmplifier,
        )


class AmplifierTestCase(unittest.TestCase):
    def testConstructorNoRawAmp(self):
        """Test constructor with no raw amplifier
        """
        aw = AmplifierWrapper(name="amp 1", doRawAmp=False)
        amp = aw.amplifier
        self.assertEquals(aw.name, amp.getName())
        self.assertEquals(aw.bbox, amp.getBBox())
        self.assertEquals(aw.gain, amp.getGain())
        self.assertEquals(aw.readNoise, amp.getReadNoise())
        self.assertFalse(amp.hasRawAmplifier())
        self.assertTrue(amp.getRawAmplifier() is None)

    def testConstructorWithRawAmp(self):
        """Test constructor with a raw amplifier
        """
        aw = AmplifierWrapper(name="", doRawAmp=True)
        amp = aw.amplifier
        self.assertEquals(aw.name, amp.getName())
        self.assertEquals(aw.bbox, amp.getBBox())
        self.assertEquals(aw.gain, amp.getGain())
        self.assertEquals(aw.readNoise, amp.getReadNoise())
        self.assertTrue(amp.hasRawAmplifier())

    def testConstructorWithBadRawAmp(self):
        """Test constructor with a raw amplifier with bad data size
        """
        self.assertRaises(LsstCppException, AmplifierWrapper, name="amp 1", doBadBBox=True)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    lsst.utils.tests.init()

    suites = []
    suites += unittest.makeSuite(AmplifierTestCase)
    suites += unittest.makeSuite(lsst.utils.tests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit = False):
    """Run the tests"""
    lsst.utils.tests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)
