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

import lsst.utils.tests
from lsst.daf.base import PropertyList
from lsst.afw.fits import makeLimitedFitsHeader


class MakeLimitedFitsHeaderTestCase(lsst.utils.tests.TestCase):

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
            self.assertEqual(header, expectedHeader)

    def testBasics(self):
        """Check basic formatting and skipping bad values
        """
        metadata = PropertyList()
        dataList = [
            ("ABOOL", True),
            ("AFLOAT", 1.2e25),
            ("ANINT", -5),
            ("LONGNAME1", 1),  # name is longer than 8 characters; skip it
            ("LONGSTR", "skip this item because the formatted value "
                "is too long: longer than 80 characters "),
            ("ASTRING1", "value for string"),
        ]
        for name, value in dataList:
            metadata.set(name, value)

        header = makeLimitedFitsHeader(metadata)

        expectedLines = [  # without padding to 80 chars
            "ABOOL   = 1",
            "AFLOAT  =              1.2E+25",
            "ANINT   =                   -5",
            "ASTRING1= 'value for string'",
        ]
        expectedHeader = "".join("%-80s" % val for val in expectedLines)

        self.assertEqual(header, expectedHeader)

        self.checkExcludeNames(metadata, expectedLines)

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
            "ABOOL   = 0",
            "AFLOAT  =                 -5.6",
            "ANINT   =                 1052",
            "ASTRING1= 'more'",
        ]
        expectedHeader = "".join("%-80s" % val for val in expectedLines)

        self.assertEqual(header, expectedHeader)

        self.checkExcludeNames(metadata, expectedLines)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    import sys
    setup_module(sys.modules[__name__])
    unittest.main()
