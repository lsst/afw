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

import unittest
import numpy as np

import lsst.utils.tests
from lsst.daf.base import PropertyList
from lsst.afw.fits import makeLimitedFitsHeader


class MakeLimitedFitsHeaderTestCase(lsst.utils.tests.TestCase):

    def assertHeadersEqual(self, header, expectedHeader, rtol=np.finfo(np.float64).resolution):
        """Compare 80 characters at a time

        Floating point values are extracted from the FITS card and compared
        as floating point numbers rather than as strings.

        Parameters
        ----------
        header : `str`
            FITS-style header string calculated by the test.
        expectedHeader : `str`
            Reference header string.
        rtol = `float`, optional
            Tolerance to use for floating point comparisons.  This parameters
            is passed directly to `~lsst.utils.tests.assertFloatsAlmostEqual`.
            The default is for double precision comparison.
        """
        self.assertEqual(len(header), len(expectedHeader),
                         msg="Compare header lengths")
        start = 0
        while start < len(header):
            end = start + 80
            # Strip trailing whitespace to make the diff clearer
            this = header[start:end].rstrip()
            expected = expectedHeader[start:end].rstrip()
            with self.subTest(this=this, expected=expected):
                # For floating point numbers compare as numbers
                # rather than strings
                if "'" not in expected and ("." in expected or "E" in expected):
                    nchars = 10
                    self.assertEqual(this[:nchars], expected[:nchars],
                                     msg=f"Compare first {nchars} characters of '{this}'"
                                     f" with expected '{expected}'")
                    self.assertFloatsAlmostEqual(float(this[9:]), float(expected[9:]),
                                                 rtol=rtol)
                else:
                    self.assertEqual(this, expected)
            start += 80

    def checkExcludeNames(self, metadata, expectedLines):
        """Check that makeLimitedFitsHeader properly excludes specified names
        """
        names = metadata.names()

        # skip each name in turn, then skip all names at once
        excludeNamesList = [set([name]) for name in names]
        excludeNamesList.append(set(names))

        for excludeNames in excludeNamesList:
            header = makeLimitedFitsHeader(metadata, excludeNames=excludeNames)
            expectedHeader = "".join("%-80s" % val for val in expectedLines
                                     if val[0:8].strip() not in excludeNames)
            self.assertHeadersEqual(header, expectedHeader)

    def testBasics(self):
        """Check basic formatting and skipping bad values
        """
        metadata = PropertyList()
        dataList = [
            ("ABOOL", True),
            ("AFLOAT", 1.2e25),
            ("AFLOAT2", 1.0e30),
            ("ANINT", -5),
            ("AFLOATZ", 0.0),  # ensure a float stays a float
            ("INTFLOAT", -5.0),
            ("LONGFLT", 0.0089626337538440005),
            ("LONGNAME1", 1),  # name is longer than 8 characters; skip it
            ("LONGSTR", "skip this item because the formatted value "
                "is too long: longer than 80 characters "),
            ("ASTRING1", "value for string"),
        ]
        for name, value in dataList:
            metadata.set(name, value)

        header = makeLimitedFitsHeader(metadata)

        expectedLines = [  # without padding to 80 chars
            "ABOOL   = T",
            "AFLOAT  =              1.2E+25",
            "AFLOAT2 =                1E+30",
            "ANINT   =                   -5",
            "AFLOATZ =                    0.0",
            "INTFLOAT=                   -5.0",
            "LONGFLT = 0.0089626337538440005",
            "ASTRING1= 'value for string'",
        ]
        expectedHeader = "".join("%-80s" % val for val in expectedLines)

        self.assertHeadersEqual(header, expectedHeader)

        self.checkExcludeNames(metadata, expectedLines)

    def testSinglePrecision(self):
        """Check that single precision floats do work"""
        metadata = PropertyList()

        # Numeric form of single precision floats need smaller precision
        metadata.setFloat("SINGLE", 3.14159)
        metadata.setFloat("SINGLEI", 5.0)
        metadata.setFloat("SINGLEE", -5.9e20)
        metadata.setFloat("EXP", -5e10)

        header = makeLimitedFitsHeader(metadata)

        expectedLines = [  # without padding to 80 chars
            "SINGLE  =              3.14159",
            "SINGLEI =                  5.0",
            "SINGLEE =             -5.9E+20",
            "EXP     =               -5E+10",
        ]
        expectedHeader = "".join("%-80s" % val for val in expectedLines)

        self.assertHeadersEqual(header, expectedHeader, rtol=np.finfo(np.float32).resolution)

    def testArrayValues(self):
        """Check that only the final value is used from an array
        """
        metadata = PropertyList()
        # work around DM-13232 by setting ABOOL one value at a time
        for value in [True, True, True, False]:
            metadata.add("ABOOL", value)
        dataList = [
            ("AFLOAT", [1.2e25, -5.6]),
            ("ANINT", [-5, 752, 1052]),
            ("ASTRING1", ["value for string", "more"]),
        ]
        for name, value in dataList:
            metadata.set(name, value)

        header = makeLimitedFitsHeader(metadata)

        expectedLines = [  # without padding to 80 chars
            "ABOOL   = F",
            "AFLOAT  =                 -5.6",
            "ANINT   =                 1052",
            "ASTRING1= 'more'",
        ]
        expectedHeader = "".join("%-80s" % val for val in expectedLines)

        self.assertHeadersEqual(header, expectedHeader)

        self.checkExcludeNames(metadata, expectedLines)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    import sys
    setup_module(sys.modules[__name__])
    unittest.main()
