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

import copy
import unittest

import lsst.utils.tests

from lsst.afw.typehandling import Storable, UnsupportedOperationException


class PointlessStorable(Storable):
    """Test of default behavior of Storable.
    """
    pass


class PointyStorable(Storable):
    """Test that we can inherit from Storable in Python.
    """
    def __init__(self, state):
        super().__init__()
        self._state = state

    def __str__(self):
        return "value = %s" % self._state

    def __repr__(self):
        return "PointyStorable(%r)" % self._state

    def __hash__(self):
        return hash(self._state)

    def __copy__(self):
        return PointyStorable(self._state)

    def __deepcopy__(self, memo=None):
        return PointyStorable(copy.deepcopy(self._state, memo))

    def __eq__(self, other):
        return self._state == other._state


class StorableTestSuite(lsst.utils.tests.TestCase):

    def setUp(self):
        self.testbed = PointlessStorable()

    def testCopy(self):
        with self.assertRaises(UnsupportedOperationException):
            copy.copy(self.testbed)
        with self.assertRaises(UnsupportedOperationException):
            copy.deepcopy(self.testbed)

    def testStr(self):
        with self.assertRaises(UnsupportedOperationException):
            str(self.testbed)

    def testRepr(self):
        with self.assertRaises(UnsupportedOperationException):
            repr(self.testbed)

    def testHash(self):
        with self.assertRaises(UnsupportedOperationException):
            hash(self.testbed)

    def testEq(self):
        self.assertNotEqual(self.testbed, PointlessStorable())


class PointyStorableTestSuite(lsst.utils.tests.TestCase):

    def setUp(self):
        self.aList = [42]
        self.testbed = PointyStorable(self.aList)

    def testCopy(self):
        shallow = copy.copy(self.testbed)
        self.assertIsNot(shallow, self.testbed)
        self.assertEqual(shallow, self.testbed)

        deep = copy.deepcopy(self.testbed)
        self.assertIsNot(deep, self.testbed)
        self.assertEqual(deep, self.testbed)

        self.aList.append(43)
        self.assertEqual(shallow, PointyStorable([42, 43]))
        self.assertEqual(deep, PointyStorable([42]))

    def testStr(self):
        self.assertEqual(str(self.testbed), "value = [42]")

    def testRepr(self):
        self.assertEqual(repr(self.testbed), "PointyStorable([42])")

    def testHash(self):
        with self.assertRaises(TypeError):
            hash(self.testbed)

    def testEq(self):
        self.assertEqual(self.testbed, PointyStorable([42]))
        self.assertNotEqual(self.testbed, PointyStorable(0))


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
