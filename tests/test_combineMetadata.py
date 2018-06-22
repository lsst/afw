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
from lsst.afw.fits import combineMetadata


class CombineMetadataTestCase(lsst.utils.tests.TestCase):

    def assertMetadataEqual(self, md1, md2):
        names1 = md1.getOrderedNames()
        names2 = md2.getOrderedNames()
        self.assertEqual(names1, names2)
        for name in names1:
            item1 = md1.getArray(name)
            item2 = md2.getArray(name)
            self.assertEqual(item1, item2)
            self.assertEqual(type(item1), type(item2))

    def testNoConflicts(self):
        """Test combination with valid values and no overlap,
        except COMMENT and HISTORY, which are combined
        """
        md1 = PropertyList()
        md1.set("int1", [1, 2])
        md1.set("float1", 1.23)
        md1.set("string1", "md1 string1 value")
        md1.set("COMMENT", "md1 comment")
        md1.set("HISTORY", "md1 history")
        md1Copy = md1.deepCopy()

        md2 = PropertyList()
        md2.set("int2", 2)
        md2.set("float2", [2.34, -3.45])
        md2.set("string2", "md2 string2 value")
        md2.set("COMMENT", "md2 comment")
        md2.set("HISTORY", "md2 history")
        md2Copy = md2.deepCopy()

        result = combineMetadata(md1, md2)
        self.assertEqual(result.getOrderedNames(),
                         ["int1", "float1", "string1", "COMMENT", "HISTORY",
                         "int2", "float2", "string2"])
        self.assertEqual(result.getArray("COMMENT"), ["md1 comment", "md2 comment"])
        self.assertEqual(result.getArray("HISTORY"), ["md1 history", "md2 history"])
        for name in md1.getOrderedNames():
            if name in ("COMMENT", "HISTORY"):
                continue
            self.assertEqual(result.getScalar(name), md1.getArray(name)[-1])
        for name in md2.getOrderedNames():
            if name in ("COMMENT", "HISTORY"):
                continue
            self.assertEqual(result.getScalar(name), md2.getArray(name)[-1])

        # input should be unchanged
        self.assertMetadataEqual(md1, md1Copy)
        self.assertMetadataEqual(md2, md2Copy)

    def testIgnoreInvalid(self):
        """Test that invalid items in the either argument are ignored
        """
        md1 = PropertyList()
        # Set COMMENT and HISTORY to invalid values -- anything other than string
        # (regardless if it is a scalar or an array);
        # for md1 use arrays and md2 use scalars, just to try both
        md1.set("COMMENT", [5, 6])
        md1.set("HISTORY", [3.5, 6.1])
        md1Copy = md1.deepCopy()

        md2 = PropertyList()
        # Set COMMENT and HISTORY to invalid values; see comment above md1.set("COMMENT", ...)
        md2.set("COMMENT", 7)
        md2.set("HISTORY", 1.06)
        md2Copy = md2.deepCopy()

        result = combineMetadata(md1, md2)
        resultNames = result.getOrderedNames()
        self.assertEqual(resultNames, [])

        # input should be unchanged
        self.assertMetadataEqual(md1, md1Copy)
        self.assertMetadataEqual(md2, md2Copy)

    def testReplaceDuplicates(self):
        """Test that names in `second` override those in `first`, regardless of type
        """
        # names that start with "item" appear in both sets of metadata
        md1 = PropertyList()
        md1.set("int1", 5)
        md1.set("itema", [1, 2])
        md1.set("float1", 3.1)
        md1.set("itemb", 1.23)
        md1.set("string1", "md1 string1 value")
        md1.set("itemc", "md1 string value")
        md1Copy = md1.deepCopy()

        md2 = PropertyList()
        md2.set("itemc", 2)
        md2.set("int2", 2)
        md2.set("itemb", ["some data", "more data"])
        md2.set("float2", 2.34)
        md2.set("itema", 5.27)
        md2.set("string2", "md2 string value")
        md2Names = md2.getOrderedNames()
        md2Copy = md2.deepCopy()

        result = combineMetadata(md1, md2)
        expectedNames = ["int1", "float1", "string1"] + list(md2Names)
        self.assertEqual(result.getOrderedNames(), expectedNames)
        md2NameSet = set(md2Names)
        for name in result.getOrderedNames():
            if name in md2NameSet:
                self.assertEqual(result.getScalar(name), md2.getArray(name)[-1])
            else:
                self.assertEqual(result.getScalar(name), md1.getArray(name)[-1])

        # input should be unchanged
        self.assertMetadataEqual(md1, md1Copy)
        self.assertMetadataEqual(md2, md2Copy)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    import sys
    setup_module(sys.modules[__name__])
    unittest.main()
