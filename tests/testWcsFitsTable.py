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

from __future__ import absolute_import, division, print_function
import unittest

import pyfits

import lsst.afw.image
import lsst.afw.geom
import lsst.utils.tests
import lsst.daf.base


class WcsFitsTableTestCase(unittest.TestCase):
    """Test that we can read and write Wcs objects saved to FITS binary tables.
    """

    def setUp(self):
        # metadata taken from CFHT data
        # v695856-e0/v695856-e0-c000-a00.sci.fits

        self.metadata = lsst.daf.base.PropertySet()

        self.metadata.set("SIMPLE", "T")
        self.metadata.set("BITPIX", -32)
        self.metadata.set("NAXIS", 2)
        self.metadata.set("NAXIS1", 1024)
        self.metadata.set("NAXIS2", 1153)
        self.metadata.set("RADECSYS", "FK5")
        self.metadata.set("EQUINOX", 2000.)

        self.metadata.setDouble("CRVAL1", 215.604025685476)
        self.metadata.setDouble("CRVAL2", 53.1595451514076)
        self.metadata.setDouble("CRPIX1", 1109.99981456774)
        self.metadata.setDouble("CRPIX2", 560.018167811613)
        self.metadata.set("CTYPE1", "RA---SIN")
        self.metadata.set("CTYPE2", "DEC--SIN")

        self.metadata.setDouble("CD1_1", 5.10808596133527E-05)
        self.metadata.setDouble("CD1_2", 1.85579539217196E-07)
        self.metadata.setDouble("CD2_2", -5.10281493481982E-05)
        self.metadata.setDouble("CD2_1", -8.27440751733828E-07)

    def tearDown(self):
        del self.metadata

    def doFitsRoundTrip(self, fileName, wcsIn):
        wcsIn.writeFits(fileName)
        wcsOut = lsst.afw.image.Wcs.readFits(fileName)
        return wcsOut

    def testSimpleWcs(self):
        with lsst.utils.tests.getTempFilePath(".fits") as fileName:
            wcsIn = lsst.afw.image.makeWcs(self.metadata)
            wcsOut = self.doFitsRoundTrip(fileName, wcsIn)
            self.assertEqual(wcsIn, wcsOut)

    def addSipMetadata(self):
        self.metadata.add("A_ORDER", 3)
        self.metadata.add("A_0_0", -3.4299726900155e-05)
        self.metadata.add("A_0_2", 2.9999243742039e-08)
        self.metadata.add("A_0_3", 5.3160367322875e-12)
        self.metadata.add("A_1_0", -1.1102230246252e-16)
        self.metadata.add("A_1_1", 1.7804837804549e-07)
        self.metadata.add("A_1_2", -3.9117665277930e-10)
        self.metadata.add("A_2_0", 1.2614116305773e-07)
        self.metadata.add("A_2_1", 2.4753748298399e-12)
        self.metadata.add("A_3_0", -4.0559790823371e-10)
        self.metadata.add("B_ORDER", 3)
        self.metadata.add("B_0_0", -0.00040333633853922)
        self.metadata.add("B_0_2", 2.7329405108287e-07)
        self.metadata.add("B_0_3", -4.1945333823804e-10)
        self.metadata.add("B_1_1", 1.0211300606274e-07)
        self.metadata.add("B_1_2", -1.1907781112538e-12)
        self.metadata.add("B_2_0", 7.1256679698479e-08)
        self.metadata.add("B_2_1", -4.0026664120969e-10)
        self.metadata.add("B_3_0", 7.2509034631981e-14)
        self.metadata.add("AP_ORDER", 5)
        self.metadata.add("AP_0_0", 0.065169424373537)
        self.metadata.add("AP_0_1", 3.5323035231808e-05)
        self.metadata.add("AP_0_2", -2.4878457741060e-08)
        self.metadata.add("AP_0_3", -1.4288745247360e-11)
        self.metadata.add("AP_0_4", -2.0000000098183)
        self.metadata.add("AP_0_5", 4.3337569354109e-19)
        self.metadata.add("AP_1_0", 1.9993638555698)
        self.metadata.add("AP_1_1", -2.0722860000493e-07)
        self.metadata.add("AP_1_2", 4.7562056847339e-10)
        self.metadata.add("AP_1_3", -8.5172068319818e-06)
        self.metadata.add("AP_1_4", -1.3242986537057e-18)
        self.metadata.add("AP_2_0", -1.4594781790233e-07)
        self.metadata.add("AP_2_1", -2.9254828606617e-12)
        self.metadata.add("AP_2_2", -2.7203380713516e-11)
        self.metadata.add("AP_2_3", 1.5030517486646e-19)
        self.metadata.add("AP_3_0", 4.7856034999197e-10)
        self.metadata.add("AP_3_1", 1.5571061278960e-15)
        self.metadata.add("AP_3_2", -3.2422164667295e-18)
        self.metadata.add("AP_4_0", 5.8904402441647e-16)
        self.metadata.add("AP_4_1", -4.5488928339401e-20)
        self.metadata.add("AP_5_0", -1.3198044795585e-18)
        self.metadata.add("BP_ORDER", 5)
        self.metadata.add("BP_0_0", 0.00025729974056661)
        self.metadata.add("BP_0_1", -0.00060857907313083)
        self.metadata.add("BP_0_2", -3.1283728005742e-07)
        self.metadata.add("BP_0_3", 5.0413932972962e-10)
        self.metadata.add("BP_0_4", -0.0046142128142681)
        self.metadata.add("BP_0_5", -2.2359607268985e-18)
        self.metadata.add("BP_1_0", 0.0046783112625990)
        self.metadata.add("BP_1_1", -1.2304042740813e-07)
        self.metadata.add("BP_1_2", -2.3756827881344e-12)
        self.metadata.add("BP_1_3", -3.9300202582816e-08)
        self.metadata.add("BP_1_4", -9.7385290942256e-21)
        self.metadata.add("BP_2_0", -6.5238116398890e-08)
        self.metadata.add("BP_2_1", 4.7855579009100e-10)
        self.metadata.add("BP_2_2", -1.2297758131839e-13)
        self.metadata.add("BP_2_3", -3.0849793267035e-18)
        self.metadata.add("BP_3_0", -9.3923321275113e-12)
        self.metadata.add("BP_3_1", -1.3193479628568e-17)
        self.metadata.add("BP_3_2", 2.1762350028059e-19)
        self.metadata.add("BP_4_0", -5.9687252632035e-16)
        self.metadata.add("BP_4_1", -1.4096893423344e-18)
        self.metadata.add("BP_5_0", 2.8085458107813e-19)
        self.metadata.set("CTYPE1", "RA---TAN-SIP")
        self.metadata.set("CTYPE2", "DEC--TAN-SIP")

    def testTanWcs(self):
        self.addSipMetadata()
        wcsIn = lsst.afw.image.makeWcs(self.metadata)
        with lsst.utils.tests.getTempFilePath(".fits") as fileName:
            wcsOut = self.doFitsRoundTrip(fileName, wcsIn)
            wcsIn1 = wcsIn
            wcsOut1 = wcsOut
            self.assertIsNotNone(wcsIn1)
            self.assertIsNotNone(wcsOut1)
            self.assertTrue(wcsIn1.hasDistortion())
            self.assertTrue(wcsOut1.hasDistortion())
            self.assertEqual(wcsIn1, wcsOut1)

    def testExposure(self):
        """Test that we load the Wcs from the binary table instead of headers when possible."""
        self.addSipMetadata()
        wcsIn = lsst.afw.image.makeWcs(self.metadata)
        dim = lsst.afw.geom.Extent2I(20, 30)
        expIn = lsst.afw.image.ExposureF(dim)
        expIn.setWcs(wcsIn)
        with lsst.utils.tests.getTempFilePath(".fits") as fileName:
            expIn.writeFits(fileName)
            # Manually mess up the headers, so we'd know if we were loading the Wcs from that;
            # when there is a WCS in the header and a WCS in the FITS table, we should use the
            # latter, because the former might just be an approximation.
            fits = pyfits.open(fileName)
            fits[1].header.remove("CTYPE1")
            fits[1].header.remove("CTYPE2")
            fits.writeto(fileName, clobber=True)
            # now load it using afw
            expOut = lsst.afw.image.ExposureF(fileName)
            wcsOut = expOut.getWcs()
            self.assertEqual(wcsIn, wcsOut)

    def testWcsWhenNonPersistable(self):
        """Test that we can round-trip a WCS even when it is not persistable"""
        import os

        fileName = os.path.join(os.path.split(__file__)[0], "data", "ZPN.fits")
        exp = lsst.afw.image.ExposureF(fileName)
        del fileName

        self.assertFalse(exp.getWcs().isPersistable(),
                         "Test assumes that ZPN projections are not persistable")

        with lsst.utils.tests.getTempFilePath(".fits") as fileName:
            exp.writeFits(fileName)
            exp2 = lsst.afw.image.ExposureF(fileName)
            self.assertEqual(exp.getWcs(), exp2.getWcs())

    def testSkyOriginPrecision(self):
        """Test that we don't lose precision in CRVAL when round-tripping
        (DM-10105), using a WCS in which we previously did.
        """
        metadata = lsst.daf.base.PropertyList()
        metadata.add('CD1_1', -4.66666666666667e-05)
        metadata.add('CD1_2', 0.0)
        metadata.add('CD2_1', 0.0)
        metadata.add('CD2_2', 4.66666666666667e-05)
        metadata.add('CRPIX1', 18000.0)
        metadata.add('CRPIX2', 18000.0)
        metadata.add('CRVAL1', 247.5)
        metadata.add('CRVAL2', -87.0247933884297)
        metadata.add('CTYPE1', 'RA---TAN')
        metadata.add('CTYPE2', 'DEC--TAN')
        metadata.add('CUNIT1', 'deg')
        metadata.add('CUNIT2', 'deg')
        metadata.add('NAXIS', 2)
        metadata.add('RADESYS', 'ICRS')
        wcs1 = lsst.afw.image.makeWcs(metadata)
        with lsst.utils.tests.getTempFilePath(".fits") as fileName:
            wcs1.writeFits(fileName)
            wcs2 = lsst.afw.image.Wcs.readFits(fileName)
            self.assertEqual(wcs1, wcs2)



class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
