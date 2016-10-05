#pybind11##!/usr/bin/env python
#pybind11#from __future__ import absolute_import, division
#pybind11#
#pybind11##
#pybind11## LSST Data Management System
#pybind11## Copyright 2008, 2009, 2010 LSST Corporation.
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
#pybind11#
#pybind11#import unittest
#pybind11#import pyfits
#pybind11#
#pybind11#import lsst.afw.image
#pybind11#import lsst.afw.geom
#pybind11#import lsst.utils.tests
#pybind11#import lsst.daf.base
#pybind11#
#pybind11#
#pybind11#class WcsFitsTableTestCase(unittest.TestCase):
#pybind11#    """Test that we can read and write Wcs objects saved to FITS binary tables.
#pybind11#    """
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        # metadata taken from CFHT data
#pybind11#        # v695856-e0/v695856-e0-c000-a00.sci.fits
#pybind11#
#pybind11#        self.metadata = lsst.daf.base.PropertySet()
#pybind11#
#pybind11#        self.metadata.set("SIMPLE", "T")
#pybind11#        self.metadata.set("BITPIX", -32)
#pybind11#        self.metadata.set("NAXIS", 2)
#pybind11#        self.metadata.set("NAXIS1", 1024)
#pybind11#        self.metadata.set("NAXIS2", 1153)
#pybind11#        self.metadata.set("RADECSYS", "FK5")
#pybind11#        self.metadata.set("EQUINOX", 2000.)
#pybind11#
#pybind11#        self.metadata.setDouble("CRVAL1", 215.604025685476)
#pybind11#        self.metadata.setDouble("CRVAL2", 53.1595451514076)
#pybind11#        self.metadata.setDouble("CRPIX1", 1109.99981456774)
#pybind11#        self.metadata.setDouble("CRPIX2", 560.018167811613)
#pybind11#        self.metadata.set("CTYPE1", "RA---SIN")
#pybind11#        self.metadata.set("CTYPE2", "DEC--SIN")
#pybind11#
#pybind11#        self.metadata.setDouble("CD1_1", 5.10808596133527E-05)
#pybind11#        self.metadata.setDouble("CD1_2", 1.85579539217196E-07)
#pybind11#        self.metadata.setDouble("CD2_2", -5.10281493481982E-05)
#pybind11#        self.metadata.setDouble("CD2_1", -8.27440751733828E-07)
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        del self.metadata
#pybind11#
#pybind11#    def doFitsRoundTrip(self, fileName, wcsIn):
#pybind11#        wcsIn.writeFits(fileName)
#pybind11#        wcsOut = lsst.afw.image.Wcs.readFits(fileName)
#pybind11#        return wcsOut
#pybind11#
#pybind11#    def testSimpleWcs(self):
#pybind11#        with lsst.utils.tests.getTempFilePath(".fits") as fileName:
#pybind11#            wcsIn = lsst.afw.image.makeWcs(self.metadata)
#pybind11#            wcsOut = self.doFitsRoundTrip(fileName, wcsIn)
#pybind11#            self.assertEqual(wcsIn, wcsOut)
#pybind11#
#pybind11#    def addSipMetadata(self):
#pybind11#        self.metadata.add("A_ORDER", 3)
#pybind11#        self.metadata.add("A_0_0", -3.4299726900155e-05)
#pybind11#        self.metadata.add("A_0_2", 2.9999243742039e-08)
#pybind11#        self.metadata.add("A_0_3", 5.3160367322875e-12)
#pybind11#        self.metadata.add("A_1_0", -1.1102230246252e-16)
#pybind11#        self.metadata.add("A_1_1", 1.7804837804549e-07)
#pybind11#        self.metadata.add("A_1_2", -3.9117665277930e-10)
#pybind11#        self.metadata.add("A_2_0", 1.2614116305773e-07)
#pybind11#        self.metadata.add("A_2_1", 2.4753748298399e-12)
#pybind11#        self.metadata.add("A_3_0", -4.0559790823371e-10)
#pybind11#        self.metadata.add("B_ORDER", 3)
#pybind11#        self.metadata.add("B_0_0", -0.00040333633853922)
#pybind11#        self.metadata.add("B_0_2", 2.7329405108287e-07)
#pybind11#        self.metadata.add("B_0_3", -4.1945333823804e-10)
#pybind11#        self.metadata.add("B_1_1", 1.0211300606274e-07)
#pybind11#        self.metadata.add("B_1_2", -1.1907781112538e-12)
#pybind11#        self.metadata.add("B_2_0", 7.1256679698479e-08)
#pybind11#        self.metadata.add("B_2_1", -4.0026664120969e-10)
#pybind11#        self.metadata.add("B_3_0", 7.2509034631981e-14)
#pybind11#        self.metadata.add("AP_ORDER", 5)
#pybind11#        self.metadata.add("AP_0_0", 0.065169424373537)
#pybind11#        self.metadata.add("AP_0_1", 3.5323035231808e-05)
#pybind11#        self.metadata.add("AP_0_2", -2.4878457741060e-08)
#pybind11#        self.metadata.add("AP_0_3", -1.4288745247360e-11)
#pybind11#        self.metadata.add("AP_0_4", -2.0000000098183)
#pybind11#        self.metadata.add("AP_0_5", 4.3337569354109e-19)
#pybind11#        self.metadata.add("AP_1_0", 1.9993638555698)
#pybind11#        self.metadata.add("AP_1_1", -2.0722860000493e-07)
#pybind11#        self.metadata.add("AP_1_2", 4.7562056847339e-10)
#pybind11#        self.metadata.add("AP_1_3", -8.5172068319818e-06)
#pybind11#        self.metadata.add("AP_1_4", -1.3242986537057e-18)
#pybind11#        self.metadata.add("AP_2_0", -1.4594781790233e-07)
#pybind11#        self.metadata.add("AP_2_1", -2.9254828606617e-12)
#pybind11#        self.metadata.add("AP_2_2", -2.7203380713516e-11)
#pybind11#        self.metadata.add("AP_2_3", 1.5030517486646e-19)
#pybind11#        self.metadata.add("AP_3_0", 4.7856034999197e-10)
#pybind11#        self.metadata.add("AP_3_1", 1.5571061278960e-15)
#pybind11#        self.metadata.add("AP_3_2", -3.2422164667295e-18)
#pybind11#        self.metadata.add("AP_4_0", 5.8904402441647e-16)
#pybind11#        self.metadata.add("AP_4_1", -4.5488928339401e-20)
#pybind11#        self.metadata.add("AP_5_0", -1.3198044795585e-18)
#pybind11#        self.metadata.add("BP_ORDER", 5)
#pybind11#        self.metadata.add("BP_0_0", 0.00025729974056661)
#pybind11#        self.metadata.add("BP_0_1", -0.00060857907313083)
#pybind11#        self.metadata.add("BP_0_2", -3.1283728005742e-07)
#pybind11#        self.metadata.add("BP_0_3", 5.0413932972962e-10)
#pybind11#        self.metadata.add("BP_0_4", -0.0046142128142681)
#pybind11#        self.metadata.add("BP_0_5", -2.2359607268985e-18)
#pybind11#        self.metadata.add("BP_1_0", 0.0046783112625990)
#pybind11#        self.metadata.add("BP_1_1", -1.2304042740813e-07)
#pybind11#        self.metadata.add("BP_1_2", -2.3756827881344e-12)
#pybind11#        self.metadata.add("BP_1_3", -3.9300202582816e-08)
#pybind11#        self.metadata.add("BP_1_4", -9.7385290942256e-21)
#pybind11#        self.metadata.add("BP_2_0", -6.5238116398890e-08)
#pybind11#        self.metadata.add("BP_2_1", 4.7855579009100e-10)
#pybind11#        self.metadata.add("BP_2_2", -1.2297758131839e-13)
#pybind11#        self.metadata.add("BP_2_3", -3.0849793267035e-18)
#pybind11#        self.metadata.add("BP_3_0", -9.3923321275113e-12)
#pybind11#        self.metadata.add("BP_3_1", -1.3193479628568e-17)
#pybind11#        self.metadata.add("BP_3_2", 2.1762350028059e-19)
#pybind11#        self.metadata.add("BP_4_0", -5.9687252632035e-16)
#pybind11#        self.metadata.add("BP_4_1", -1.4096893423344e-18)
#pybind11#        self.metadata.add("BP_5_0", 2.8085458107813e-19)
#pybind11#        self.metadata.set("CTYPE1", "RA---TAN-SIP")
#pybind11#        self.metadata.set("CTYPE2", "DEC--TAN-SIP")
#pybind11#
#pybind11#    def testTanWcs(self):
#pybind11#        self.addSipMetadata()
#pybind11#        wcsIn = lsst.afw.image.makeWcs(self.metadata)
#pybind11#        with lsst.utils.tests.getTempFilePath(".fits") as fileName:
#pybind11#            wcsOut = self.doFitsRoundTrip(fileName, wcsIn)
#pybind11#            wcsIn1 = lsst.afw.image.cast_TanWcs(wcsIn)
#pybind11#            wcsOut1 = lsst.afw.image.cast_TanWcs(wcsOut)
#pybind11#            self.assertIsNotNone(wcsIn1)
#pybind11#            self.assertIsNotNone(wcsOut1)
#pybind11#            self.assertTrue(wcsIn1.hasDistortion())
#pybind11#            self.assertTrue(wcsOut1.hasDistortion())
#pybind11#            self.assertEqual(wcsIn1, wcsOut1)
#pybind11#
#pybind11#    def testExposure(self):
#pybind11#        """Test that we load the Wcs from the binary table instead of headers when possible."""
#pybind11#        self.addSipMetadata()
#pybind11#        wcsIn = lsst.afw.image.makeWcs(self.metadata)
#pybind11#        dim = lsst.afw.geom.Extent2I(20, 30)
#pybind11#        expIn = lsst.afw.image.ExposureF(dim)
#pybind11#        expIn.setWcs(wcsIn)
#pybind11#        with lsst.utils.tests.getTempFilePath(".fits") as fileName:
#pybind11#            expIn.writeFits(fileName)
#pybind11#            # Manually mess up the headers, so we'd know if we were loading the Wcs from that;
#pybind11#            # when there is a WCS in the header and a WCS in the FITS table, we should use the
#pybind11#            # latter, because the former might just be an approximation.
#pybind11#            fits = pyfits.open(fileName)
#pybind11#            fits[1].header.remove("CTYPE1")
#pybind11#            fits[1].header.remove("CTYPE2")
#pybind11#            fits.writeto(fileName, clobber=True)
#pybind11#            # now load it using afw
#pybind11#            expOut = lsst.afw.image.ExposureF(fileName)
#pybind11#            wcsOut = expOut.getWcs()
#pybind11#            self.assertEqual(wcsIn, wcsOut)
#pybind11#
#pybind11#    def testWcsWhenNonPersistable(self):
#pybind11#        """Test that we can round-trip a WCS even when it is not persistable"""
#pybind11#        import os
#pybind11#
#pybind11#        fileName = os.path.join(os.path.split(__file__)[0], "data", "ZPN.fits")
#pybind11#        exp = lsst.afw.image.ExposureF(fileName)
#pybind11#        del fileName
#pybind11#
#pybind11#        self.assertFalse(exp.getWcs().isPersistable(),
#pybind11#                         "Test assumes that ZPN projections are not persistable")
#pybind11#
#pybind11#        with lsst.utils.tests.getTempFilePath(".fits") as fileName:
#pybind11#            exp.writeFits(fileName)
#pybind11#            exp2 = lsst.afw.image.ExposureF(fileName)
#pybind11#            self.assertEqual(exp.getWcs(), exp2.getWcs())
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
