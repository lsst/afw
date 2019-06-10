# This file is part of afw.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
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
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from collections.abc import MutableMapping
import unittest

import lsst.utils.tests

from lsst.afw.typehandling import SimpleGenericMap
from lsst.afw.typehandling.testUtils import MutableGenericMapTestBaseClass


class SimpleGenericMapTestSuite(MutableGenericMapTestBaseClass):

    @staticmethod
    def makeMap(mapType, values):
        """Initialize a map type using __setattr__ instead of a bulk constructor.
        """
        result = mapType()
        for key, value in values.items():
            result[key] = value
        return result

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.targets = {(str, SimpleGenericMap[str])}
        cls.examples = {
            "SimpleGenericMap(testData(str))": (str, SimpleGenericMap[str], cls.getTestData(str)),
            "SimpleGenericMap(testKeys(str) : 0)":
                (str, SimpleGenericMap[str], {key: 0 for key in cls.getTestData(str).keys()}),
            "SimpleGenericMap(dtype=str)": (str, SimpleGenericMap[str], {}),
        }

    def tearDown(self):
        pass

    def testClass(self):
        for (_, target) in self.targets:
            self.assertTrue(issubclass(target, MutableMapping))
            self.assertIsInstance(target(), MutableMapping)

    def testInitKeywords(self):
        for (keyType, target) in self.targets:
            self.checkInitKwargs(target, self.getTestData(keyType), msg=str(target))

    def testInitPairs(self):
        for (keyType, target) in self.targets:
            self.checkInitPairs(keyType, target, self.getTestData(keyType), msg=str(target))

    def testInitMapping(self):
        for (keyType, target) in self.targets:
            # Init from dict
            self.checkInitMapping(keyType, target, self.getTestData(keyType), msg=str(target))
            # Init from GenericMap
            self.checkInitMapping(keyType, target, self.makeMap(target, self.getTestData(keyType)),
                                  msg=str(target))

    def testFromKeys(self):
        for (keyType, target) in self.targets:
            keys = self.getTestData(keyType).keys()
            for value in self.getTestData(keyType).values():
                self.checkFromKeys(keyType, target, keys, value,
                                   msg=" class=%s, value=%r" % (target, value))
            self.checkFromKeysDefault(keyType, target, keys, msg=" class=%s, no value" % (target))

    def testCopy(self):
        for label, (keyType, mappingType, contents) in self.examples.items():
            mapping1 = self.makeMap(mappingType, contents)
            mapping2 = mapping1.copy()
            self.assertEqual(mapping1, mapping2, msg="%s" % label)
            mapping1[keyType(42)] = "A random value!"
            self.assertNotEqual(mapping1, mapping2, msg="%s" % label)

    def testEquality(self):
        for label1, (_, mappingType1, contents1) in self.examples.items():
            mapping1 = self.makeMap(mappingType1, contents1)
            for label2, (_, mappingType2, contents2) in self.examples.items():
                mapping2 = self.makeMap(mappingType2, contents2)
                if contents1 == contents2:
                    self.assertIsNot(mapping1, mapping2, msg="%s vs %s" % (label1, label2))
                    self.assertEqual(mapping1, mapping2, msg="%s vs %s" % (label1, label2))
                    self.assertEqual(mapping1, contents2, msg="%s vs dict(%s)" % (label1, label2))
                    self.assertEqual(contents1, mapping2, msg="dict(%s) vs %s" % (label1, label2))
                else:
                    self.assertNotEqual(mapping1, mapping2, msg="%s vs %s" % (label1, label2))
                    self.assertNotEqual(mapping1, contents2, msg="%s vs dict(%s)" % (label1, label2))
                    self.assertNotEqual(contents1, mapping2, msg="dict(%s) vs %s" % (label1, label2))

    def testBool(self):
        for label, (_, mappingType, contents) in self.examples.items():
            mapping = self.makeMap(mappingType, contents)
            if contents:
                self.assertTrue(mapping, msg=label)
            else:
                self.assertFalse(mapping, msg=label)

    def testContains(self):
        for label, (keyType, mappingType, contents) in self.examples.items():
            mapping = self.makeMap(mappingType, contents)
            self.checkContains(keyType, mapping, contents, msg=label)

    def testContents(self):
        for label, (keyType, mappingType, contents) in self.examples.items():
            mapping = self.makeMap(mappingType, contents)
            self.checkContents(keyType, mapping, contents, msg=label)

    def testGet(self):
        for label, (keyType, mappingType, contents) in self.examples.items():
            mapping = self.makeMap(mappingType, contents)
            self.checkGet(keyType, mapping, contents, msg=label)

    def testIteration(self):
        for label, (_, mappingType, contents) in self.examples.items():
            mapping = self.makeMap(mappingType, contents)
            self.checkIteration(mapping, contents, msg=label)

    def testViews(self):
        for label, (_, mappingType, contents) in self.examples.items():
            self.checkMutableViews(mappingType, contents, msg=label)

    def testInsertItem(self):
        for (keyType, target) in self.targets:
            self.checkInsertItem(keyType, target, self.getTestData(keyType), msg=str(target))

    def testSetdefault(self):
        for (keyType, target) in self.targets:
            self.checkSetdefault(keyType, target, self.getTestData(keyType), msg=str(target))

    def testUpdateMapping(self):
        for (keyType, target) in self.targets:
            # Update from dict
            self.checkUpdateMapping(keyType, target, self.getTestData(keyType), msg=str(target))
            # Update from GenericMap
            self.checkUpdateMapping(keyType, target, self.makeMap(target, self.getTestData(keyType)),
                                    msg=str(target))

    def testUpdatePairs(self):
        for (keyType, target) in self.targets:
            self.checkUpdatePairs(keyType, target, self.getTestData(keyType), msg=str(target))

    def testUpdateKwargs(self):
        for (keyType, target) in self.targets:
            self.checkUpdateKwargs(target, self.getTestData(keyType), msg=str(target))

    def testReplaceItem(self):
        for (keyType, target) in self.targets:
            self.checkReplaceItem(keyType, target(), msg=str(target))

    def testRemoveItem(self):
        for (keyType, target) in self.targets:
            self.checkRemoveItem(keyType, target, self.getTestData(keyType), msg=str(target))

    def testPop(self):
        for (keyType, target) in self.targets:
            self.checkPop(keyType, target, self.getTestData(keyType), msg=str(target))

    def testPopitem(self):
        for (keyType, target) in self.targets:
            self.checkPopitem(target, self.getTestData(keyType), msg=str(target))

    def testClear(self):
        for (keyType, target) in self.targets:
            self.checkClear(target, self.getTestData(keyType), msg=str(target))


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
