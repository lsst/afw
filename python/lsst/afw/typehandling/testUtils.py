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

__all__ = []

from collections import Counter

import numpy as np

import lsst.utils.tests

from lsst.afw.typehandling import Storable


class GenericMapTestBaseClass(lsst.utils.tests.TestCase):
    """Base class for unit tests of GenericMap.

    Subclasses must call `GenericMapTestBaseClass.setUp(self)`
    if they provide their own version.

    This class is not *quite* a generic Mapping testbed, because it assumes
    that the map being tested only accepts keys of a particular type, stored as
    the `dtype` member.
    """

    class SimpleStorable(Storable):
        """A subclass of Storable for testing purposes.
        """
        def __repr__(self):
            return "Simplest possible representation"

        def __hash__(self):
            return 100

        def __eq__(self, other):
            """Warning: violates both  substitution and equality symmetry!
            """
            return self.__class__ == other.__class__

    class ComplexStorable(SimpleStorable):
        """A subclass of Storable for testing purposes.
        """
        def __init__(self, storage):
            super().__init__()
            self._storage = storage

        def __repr__(self):
            return "ComplexStorable(%r)" % self._storage

        def __hash__(self):
            return hash(self._storage)

        def __eq__(self, other):
            """Warning: violates both substitution and equality symmetry!
            """
            if self.__class__ == other.__class__:
                return self._storage == other._storage
            else:
                return False

    class NotAStorable:
        """A class that should not be a legal value in a GenericMap.
        """
        def __str__(self):
            return "Non-Storable"

    _testData = {
        0: True,
        1: 42,
        2: 42.0,
        3: "How many roads must a man walk down?",
        4: SimpleStorable(),
        5: ComplexStorable(-100.0),
        6: None,
    }

    @classmethod
    def getTestData(cls, keyClass):
        """Generic dataset for testing GenericMap classes that can handle it.

        Parameters
        ----------
        keyClass : `type`
            The type of key expected by the GenericMap.
        """
        return {keyClass(key): value for key, value in cls._testData.items()}

    def setUp(self):
        """Set up a test

        Subclasses must call this method if they override setUp.
        """
        super().setUp()
        # tell unittest to use the msg argument of asserts as a supplement
        # to the error message, rather than as the whole error message
        self.longMessage = True

    # Mapping must have:
    #   __str__
    #   __repr__
    #   __eq__
    #   __ne__
    #   __contains__
    #   __getitem__
    #   get
    #   __iter__
    #   __len__
    #   __bool__
    #   keys
    #   items
    #   values

    def checkInitMapping(self, mapClass, contents, msg=""):
        """Check initialization from a mapping.

        Parameters
        ----------
        mapClass : `lsst.afw.typehandling.GenericMap`-type
            The class whose ``__init__`` method will be tested.
        contents : `Mapping`
            The key-value pairs to insert into the map
        msg : `str`
            Error message suffix describing test parameters
        """
        keyType = mapClass.dtype
        genericMap = mapClass(contents)
        self.checkContents(genericMap, contents, msg=msg)

        extraContents = {key: value for key, value in contents.items()}  # contents need not define copy()
        extraKey = keyType(101)
        extraValue = 'Extra value'
        extraContents[extraKey] = extraValue
        genericMap = mapClass(contents, **{keyType(101): extraValue})
        self.checkContents(genericMap, extraContents, msg=msg)

        with self.assertRaises(TypeError, msg=msg):
            mapClass({keyType(0): GenericMapTestBaseClass.NotAStorable()})

    def checkInitPairs(self, mapClass, contents, msg=""):
        """Check initialization from an iterable of pairs.

        Parameters
        ----------
        mapClass: `lsst.afw.typehandling.GenericMap`-type
            The class whose ``__init__`` method will be tested.
        contents : `Mapping`
            The key-value pairs to insert into the map
        msg : `str`
            Error message suffix describing test parameters
        """
        keyType = mapClass.dtype
        genericMap = mapClass(contents.items())
        self.checkContents(genericMap, contents, msg=msg)

        extraContents = {key: value for key, value in contents.items()}  # contents need not define copy()
        extraKey = keyType(101)
        extraValue = 'Extra value'
        extraContents[extraKey] = extraValue
        genericMap = mapClass(contents.items(), **{keyType(101): extraValue})
        self.checkContents(genericMap, extraContents, msg=msg)

        with self.assertRaises(TypeError, msg=msg):
            mapClass([(keyType(0), GenericMapTestBaseClass.NotAStorable())])

    def checkInitKwargs(self, mapClass, contents, msg=""):
        """Check bulk insertion from keywords into a GenericMap.

        Parameters
        ----------
        mapClass: `lsst.afw.typehandling.GenericMapS`-type
            The class whose ``__init__`` method will be tested.
            Must allow string keys.
        contents : `Mapping`
            The key-value pairs to insert into the map
        msg : `str`
            Error message suffix describing test parameters
        """
        genericMap = mapClass(**contents)
        self.checkContents(genericMap, contents, msg=msg)

        with self.assertRaises(TypeError, msg=msg):
            mapClass(notAKey=GenericMapTestBaseClass.NotAStorable())

    def checkFromKeysDefault(self, mapClass, keys, msg=""):
        """Check initialization using the ``fromkeys`` factory.

        Unlike `checkFromKeys`, this method lets ``fromkeys`` use its default
        value (which may give different behavior, in nonconforming
        implementations, from explicitly passing `None`).

        Parameters
        ----------
        mapClass: `lsst.afw.typehandling.GenericMap`-type
            The class whose ``fromkeys`` method will be tested.
        keys : `iterable`
            The keys to insert into the map.
        msg : `str`
            Error message suffix describing test parameters
        """
        genericMap = mapClass.fromkeys(keys)
        self.assertIsInstance(genericMap, mapClass, msg=msg)
        self.checkContents(genericMap, dict.fromkeys(keys), msg=msg)

    def checkFromKeys(self, mapClass, keys, value, msg=""):
        """Check initialization using the ``fromkeys`` factory.

        Parameters
        ----------
        mapClass: `lsst.afw.typehandling.GenericMap`-type
            The class whose ``fromkeys`` method will be tested.
        keys : `iterable`
            The keys to insert into the map.
        value
            A legal value for a GenericMap.
        msg : `str`
            Error message suffix describing test parameters
        """
        genericMap = mapClass.fromkeys(keys, value)
        self.assertIsInstance(genericMap, mapClass, msg=msg)
        self.checkContents(genericMap, dict.fromkeys(keys, value), msg=msg)

    def checkContains(self, genericMap, contents, msg=""):
        """Check the contents of a GenericMap.

        Parameters
        ----------
        genericMap : `lsst.afw.typehandling.GenericMap`
            The map to test.
        contents : `Mapping`
            The key-value pairs that should be present in ``genericMap``
        msg : `str`
            Error message suffix describing test parameters
        """
        for key in contents:
            self.assertIn(key, genericMap, msg=msg)

        keyType = genericMap.dtype
        for key in range(30):
            if keyType(key) not in contents:
                self.assertNotIn(keyType(key), genericMap, msg=msg)

        wrongType = float if keyType is not float else int
        with self.assertRaises(TypeError):
            wrongType(0) in genericMap

    def checkContents(self, genericMap, contents, msg=""):
        """Check the contents of a GenericMap.

        Parameters
        ----------
        genericMap : `lsst.afw.typehandling.GenericMap`
            The map to test.
        contents : `Mapping`
            The key-value pairs that should be present in ``genericMap``
        msg : `str`
            Error message suffix describing test parameters
        """
        for key, value in contents.items():
            self.assertEqual(genericMap[key], value, msg=msg)

        keyType = genericMap.dtype
        for key in (keyType(key) for key in range(30) if keyType(key) not in contents):
            with self.assertRaises(KeyError, msg=msg):
                genericMap[key]

        wrongType = float if keyType is not float else int
        with self.assertRaises(TypeError):
            genericMap[wrongType(0)]

    def checkGet(self, genericMap, contents, msg=""):
        """Check that GenericMap.get works correctly.

        Parameters
        ----------
        genericMap : `lsst.afw.typehandling.GenericMap`
            The map to test.
        contents : `Mapping`
            The key-value pairs that should be present in ``genericMap``
        msg : `str`
            Error message suffix describing test parameters
        """
        default = "Not a default value"
        for key, value in contents.items():
            self.assertEqual(genericMap.get(key), value, msg=msg)
            self.assertEqual(genericMap.get(key, default), value, msg=msg)

        keyType = genericMap.dtype
        for key in (keyType(key) for key in range(30) if keyType(key) not in contents):
            self.assertEqual(genericMap.get(key), None, msg=msg)
            self.assertEqual(genericMap.get(key, default), default, msg=msg)

        wrongType = float if keyType is not float else int
        with self.assertRaises(TypeError):
            genericMap.get(wrongType(0))

    def checkIteration(self, genericMap, contents, msg=""):
        """Check the result of iterating over a GenericMap.

        Parameters
        ----------
        genericMap : `lsst.afw.typehandling.GenericMap`
            The map to test.
        contents : `Mapping`
            The key-value pairs that should be present in ``genericMap``
        msg : `str`
            Error message suffix describing test parameters
        """
        self.assertEqual({key: genericMap[key] for key in genericMap}, dict(contents), msg=msg)

    def checkViews(self, genericMap, contents, msg=""):
        """Check the views provided by a GenericMap.

        Parameters
        ----------
        genericMap : `lsst.afw.typehandling.GenericMap`
            The map to test.
        contents : `Mapping`
            The key-value pairs that should be present in ``genericMap``
        msg : `str`
            Error message suffix describing test parameters
        """
        self.assertEqual(set(genericMap.keys()), set(contents.keys()), msg=msg)
        self.assertEqual(Counter(genericMap.values()), Counter(contents.values()), msg=msg)
        self.assertEqual(Counter(genericMap.items()), Counter(contents.items()), msg=msg)


class MutableGenericMapTestBaseClass(GenericMapTestBaseClass):
    """Base class for unit tests of GenericMap that allow insertion/deletion.

    Subclasses must call `MutableGenericMapTestBaseClass.setUp(self)`
    if they provide their own version.
    """

    @classmethod
    def _fillMap(cls, mapFactory, contents):
        """Create a new GenericMap with particular contents.

        Parameters
        ----------
        mapFactory : callable
            A zero-argument callable that creates an empty
            `lsst.afw.typehandling.GenericMap` object
        contents : `Mapping`
            The key-value pairs that should be present in the new map.

        Returns
        -------
        map : `lsst.afw.typehandling.GenericMap`
            a GenericMap equivalent to ``contents``
        """
        return cls._fillPartialMap(mapFactory, contents, len(contents))

    @classmethod
    def _fillPartialMap(cls, mapFactory, contents, numElements):
        """Create a new GenericMap with particular contents.

        Parameters
        ----------
        mapFactory : callable
            A zero-argument callable that creates an empty
            `lsst.afw.typehandling.GenericMap` object
        contents : `Mapping`
            The key-value pairs that should be present in the new map.
        numElements : `int`
            The number of elements from ``contents`` to be inserted.

        Returns
        -------
        map : `lsst.afw.typehandling.GenericMap`
            a GenericMap containing ``numElements`` of ``contents`` or all of
            ``contents``, whichever is smaller
        """
        newMap = mapFactory()
        for i, (key, value) in enumerate(contents.items()):
            if i < numElements:
                newMap[key] = value
            else:
                break
        return newMap

    # MutableMapping must have:
    #   __setitem__
    #   setdefault
    #   __delitem__
    #   pop
    #   popitem
    #   clear
    #   update

    def checkInsertItem(self, mapFactory, contents, msg=""):
        """Check element insertion in a GenericMap.

        Parameters
        ----------
        mapFactory : callable
            A zero-argument callable that creates an empty
            `lsst.afw.typehandling.GenericMap` object of the type to be tested
        contents : `Mapping`
            The key-value pairs to insert into the map
        msg : `str`
            Error message suffix describing test parameters
        """
        genericMap = mapFactory()

        for length, (key, value) in enumerate(contents.items()):
            loopMsg = msg + " Inserting %r=%r" % (key, value)
            genericMap[key] = value
            self.assertEqual(len(genericMap), length+1, msg=loopMsg)
            self.assertEqual(genericMap[key], value, msg=loopMsg)

        self.assertEqual(dict(genericMap), dict(contents), msg=msg)

        keyType = genericMap.dtype
        with self.assertRaises(TypeError, msg=msg):
            genericMap[keyType(0)] = GenericMapTestBaseClass.NotAStorable()

        wrongType = float if keyType is not float else int
        with self.assertRaises(TypeError):
            genericMap[wrongType(0)] = 0

    def checkSetdefault(self, mapFactory, contents, msg=""):
        """Check that GenericMap.setdefault works correctly.

        Parameters
        ----------
        mapFactory : callable
            A zero-argument callable that creates an empty
            `lsst.afw.typehandling.GenericMap` object of the type to be tested
        contents : `Mapping`
            The key-value pairs to insert into the map
        msg : `str`
            Error message suffix describing test parameters
        """
        genericMap = mapFactory()

        keyType = genericMap.dtype
        result = genericMap.setdefault(keyType(0))
        self.assertEqual(len(genericMap), 1, msg=msg)
        self.assertIsNone(result, msg=msg)
        self.assertIsNone(genericMap[keyType(0)], msg=msg)
        del genericMap[keyType(0)]

        default = "This is a default"
        for length, (key, _) in enumerate(contents.items()):
            loopMsg = msg + " Defaulting %r" % (key)
            result = genericMap.setdefault(key, default)
            self.assertEqual(len(genericMap), length+1, msg=loopMsg)
            self.assertEqual(result, default, msg=loopMsg)
            self.assertEqual(genericMap[key], default, msg=loopMsg)

        self.assertEqual(genericMap.keys(), contents.keys(), msg=msg)

        with self.assertRaises(TypeError, msg=msg):
            genericMap.setdefault(keyType(100), GenericMapTestBaseClass.NotAStorable())

        wrongType = float if keyType is not float else int
        with self.assertRaises(TypeError):
            genericMap.setdefault(wrongType(0), default)

        genericMap = self._fillMap(mapFactory, contents)
        for length, (key, value) in enumerate(contents.items()):
            loopMsg = msg + " Defaulting existing %r=%r" % (key, value)
            result = genericMap.setdefault(key, default)
            self.assertEqual(len(genericMap), len(contents), msg=loopMsg)
            self.assertEqual(result, contents[key], msg=loopMsg)
            self.assertEqual(genericMap[key], contents[key], msg=loopMsg)

    def checkUpdateMapping(self, mapFactory, contents, msg=""):
        """Check bulk insertion from a mapping into a GenericMap.

        Parameters
        ----------
        mapFactory : callable
            A zero-argument callable that creates an empty
            `lsst.afw.typehandling.GenericMap` object of the type to be tested
        contents : `Mapping`
            The key-value pairs to insert into the map
        msg : `str`
            Error message suffix describing test parameters
        """
        genericMap = self._fillPartialMap(mapFactory, dict.fromkeys(contents, 0), len(contents)/2)
        self.assertLess(len(genericMap), len(contents), msg=msg)

        genericMap.update(contents)
        self.assertEqual(dict(genericMap), dict(contents), msg=msg)

        keyType = genericMap.dtype
        with self.assertRaises(TypeError, msg=msg):
            genericMap.update({keyType(0): GenericMapTestBaseClass.NotAStorable()})

        wrongType = float if keyType is not float else int
        with self.assertRaises(TypeError, msg=msg):
            genericMap.update({wrongType(0): 0})

    def checkUpdatePairs(self, mapFactory, contents, msg=""):
        """Check bulk insertion from an iterable of pairs into a GenericMap.

        Parameters
        ----------
        mapFactory : callable
            A zero-argument callable that creates an empty
            `lsst.afw.typehandling.GenericMap` object of the type to be tested
        contents : `Mapping`
            The key-value pairs to insert into the map
        msg : `str`
            Error message suffix describing test parameters
        """
        genericMap = self._fillPartialMap(mapFactory, dict.fromkeys(contents, 0), len(contents)/2)
        self.assertLess(len(genericMap), len(contents), msg=msg)

        genericMap.update(contents.items())
        self.assertEqual(dict(genericMap), dict(contents), msg=msg)

        keyType = genericMap.dtype
        with self.assertRaises(TypeError, msg=msg):
            genericMap.update([(keyType(0), GenericMapTestBaseClass.NotAStorable())])

        wrongType = float if keyType is not float else int
        with self.assertRaises(TypeError, msg=msg):
            genericMap.update([(wrongType(0), 0)])

    def checkUpdateKwargs(self, mapFactory, contents, msg=""):
        """Check bulk insertion from keywords into a GenericMap.

        Parameters
        ----------
        mapFactory : callable
            A zero-argument callable that creates an empty
            `lsst.afw.typehandling.GenericMap` object of the type to be tested
            Must allow string keys.
        contents : `Mapping`
            The key-value pairs to insert into the map
        msg : `str`
            Error message suffix describing test parameters
        """
        genericMap = self._fillPartialMap(mapFactory, dict.fromkeys(contents, 0), len(contents)/2)
        self.assertLess(len(genericMap), len(contents), msg=msg)

        genericMap.update(**contents)
        self.assertEqual(dict(genericMap), dict(contents), msg=msg)

        with self.assertRaises(TypeError, msg=msg):
            genericMap.update(notAKey=GenericMapTestBaseClass.NotAStorable())

    def checkReplaceItem(self, genericMap, msg=""):
        """Check element replacement in a GenericMap.

        Parameters
        ----------
        genericMap : `lsst.afw.typehandling.GenericMap`
            The map to test. Must be empty.
        msg : `str`
            Error message suffix describing test parameters
        """
        self.assertFalse(genericMap, msg=msg)
        keyType = genericMap.dtype
        key = keyType(42)

        for value in self.getTestData(keyType).values():
            loopMsg = msg + " Inserting %r=%r" % (key, value)
            genericMap[key] = value  # value may be of a different type
            self.assertEqual(len(genericMap), 1, msg=loopMsg)
            self.assertEqual(genericMap[key], value, msg=loopMsg)

        self.assertEqual(dict(genericMap), {key: value}, msg=msg)

        with self.assertRaises(TypeError, msg=msg):
            genericMap[key] = GenericMapTestBaseClass.NotAStorable()

    def checkRemoveItem(self, mapFactory, contents, msg=""):
        """Check element removal from a GenericMap.

        Parameters
        ----------
        mapFactory : callable
            A zero-argument callable that creates an empty
            `lsst.afw.typehandling.GenericMap` object of the type to be tested
        contents : `Mapping`
            The key-value pairs initially occupying the map
        msg : `str`
            Error message suffix describing test parameters
        """
        genericMap = self._fillMap(mapFactory, contents)

        keyType = genericMap.dtype
        with self.assertRaises(KeyError, msg=msg):
            del genericMap[keyType(2019)]

        keysToRemove = list(contents.keys())
        np.random.shuffle(keysToRemove)
        for numPrevious, rawKey in enumerate(keysToRemove):
            key = keyType(rawKey)
            loopMsg = msg + " Deleting %r" % (key)
            del genericMap[key]
            self.assertEqual(len(genericMap), len(contents)-numPrevious-1, msg=loopMsg)
            self.assertNotIn(key, genericMap, msg=loopMsg)

        wrongType = float if keyType is not float else int
        with self.assertRaises(TypeError):
            del genericMap[wrongType(0)]

    def checkPop(self, mapFactory, contents, msg=""):
        """Check that GenericMap.pop works correctly.

        Parameters
        ----------
        mapFactory : callable
            A zero-argument callable that creates an empty
            `lsst.afw.typehandling.GenericMap` object of the type to be tested
        contents : `Mapping`
            The key-value pairs initially occupying the map
        msg : `str`
            Error message suffix describing test parameters
        """
        genericMap = self._fillMap(mapFactory, contents)

        keyType = genericMap.dtype
        with self.assertRaises(KeyError, msg=msg):
            genericMap.pop(keyType(2019))

        default = "This is a default"
        result = genericMap.pop(keyType(2019), default)
        self.assertEqual(dict(genericMap), dict(contents), msg=msg)
        self.assertEqual(result, default)

        wrongType = float if keyType is not float else int
        with self.assertRaises(TypeError):
            genericMap.pop(wrongType(0))

        itemsToRemove = list(contents.items())
        np.random.shuffle(itemsToRemove)
        for numPrevious, (rawKey, value) in enumerate(itemsToRemove):
            key = keyType(rawKey)
            loopMsg = msg + " Popping %r=%r" % (key, value)
            result = genericMap.pop(key)
            self.assertEqual(len(genericMap), len(contents)-numPrevious-1, msg=loopMsg)
            self.assertNotIn(key, genericMap, msg=loopMsg)
            self.assertEqual(result, value, msg=loopMsg)

    def checkPopitem(self, mapFactory, contents, msg=""):
        """Check that GenericMap.popitem works correctly.

        Parameters
        ----------
        mapFactory : callable
            A zero-argument callable that creates an empty
            `lsst.afw.typehandling.GenericMap` object of the type to be tested
        contents : `Mapping`
            The key-value pairs initially occupying the map
        msg : `str`
            Error message suffix describing test parameters
        """
        genericMap = self._fillMap(mapFactory, contents)

        for numPrevious in range(len(genericMap)):
            key, value = genericMap.popitem()
            loopMsg = msg + " Popping %r=%r" % (key, value)
            self.assertIn((key, value), contents.items(), msg=loopMsg)
            self.assertEqual(len(genericMap), len(contents)-numPrevious-1, msg=loopMsg)
            self.assertNotIn(key, genericMap, msg=loopMsg)

        with self.assertRaises(KeyError, msg=msg):
            genericMap.popitem()

    def checkClear(self, mapFactory, contents, msg=""):
        """Check erasing a GenericMap.

        Parameters
        ----------
        mapFactory : callable
            A zero-argument callable that creates an empty
            `lsst.afw.typehandling.GenericMap` object of the type to be tested
        contents : `Mapping`
            The key-value pairs initially occupying the map
        msg : `str`
            Error message suffix describing test parameters
        """
        genericMap = self._fillMap(mapFactory, contents)
        self.assertTrue(genericMap, msg=msg)

        genericMap.clear()
        self.assertFalse(genericMap, msg=msg)
        self.assertEqual(len(genericMap), 0, msg=msg)
        for key in genericMap:
            self.fail("Unexpected key: %s" % key, msg=msg)

    def checkMutableViews(self, mapFactory, contents, msg=""):
        """Check that the views of a GenericMap update automatically.

        This test supersedes `GenericMapTestBaseClass.checkViews`.

        Parameters
        ----------
        genericMap : `lsst.afw.typehandling.GenericMap`
            The map to test.
        contents : `Mapping`
            The key-value pairs that should be present in ``genericMap``
        msg : `str`
            Error message suffix describing test parameters
        """
        genericMap = mapFactory()
        keyView = genericMap.keys()
        valueView = genericMap.values()
        itemView = genericMap.items()

        for view in (keyView, valueView, itemView):
            self.assertEqual(len(view), len(genericMap), msg=msg)
        # is initial state correct?
        self.checkViews(genericMap, {}, msg)

        for key, value in contents.items():
            genericMap[key] = value

        for view in (keyView, valueView, itemView):
            self.assertEqual(len(view), len(genericMap), msg=msg)
        # is final state correct?
        self.checkViews(genericMap, contents, msg)
