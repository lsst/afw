#
# LSST Data Management System
# Copyright 2017 LSST Corporation.
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

import os
import unittest
import itertools

from lsst.daf.base import PropertyList

import lsst.afw.fits
import lsst.utils.tests


class FitsTestCase(lsst.utils.tests.TestCase):
    def testIgnoreKeywords(self):
        """Check that certain keywords are ignored in read/write of headers"""
        # May appear only once in the FITS file (because cfitsio will insist on putting them there)
        single = ["SIMPLE", "BITPIX", "EXTEND", "NAXIS"]
        # May not appear at all in the FITS file (cfitsio doesn't write these by default)
        notAtAll = [
            # FITS core keywords
            "GCOUNT", "PCOUNT", "XTENSION", "BSCALE", "BZERO", "TZERO", "TSCAL",
            # FITS compression keywords
            "ZBITPIX", "ZIMAGE", "ZCMPTYPE", "ZSIMPLE", "ZEXTEND", "ZBLANK", "ZDATASUM", "ZHECKSUM",
            "ZNAXIS", "ZTILE", "ZNAME", "ZVAL",
            # Not essential these be excluded, but will prevent fitsverify warnings
            "DATASUM", "CHECKSUM",
            ]
        # Additional keywords to check; these should go straight through
        # Some of these are longer/shorter versions of strings above,
        # to test that the checks for just the start of strings is working.
        others = ["FOOBAR", "SIMPLETN", "DATASUMX", "NAX", "SIM"]

        header = PropertyList()
        for ii, key in enumerate(single + notAtAll + others):
            header.add(key, ii)
        fitsFile = lsst.afw.fits.MemFileManager()
        with lsst.afw.fits.Fits(fitsFile, "w") as fits:
            fits.createEmpty()
            fits.writeMetadata(header)
        with lsst.afw.fits.Fits(fitsFile, "r") as fits:
            metadata = fits.readMetadata()
            for key in single:
                self.assertEqual(metadata.valueCount(key), 1, key)
            for key in notAtAll:
                self.assertEqual(metadata.valueCount(key), 0, key)
            for key in others:
                self.assertEqual(metadata.valueCount(key), 1, key)

class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    import sys
    setup_module(sys.modules[__name__])
    unittest.main()
