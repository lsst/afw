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

__all__ = ["GenericMap", "MutableGenericMap"]

from collections.abc import Mapping, MutableMapping

from lsst.utils import TemplateMeta
from ._typehandling import GenericMapS, MutableGenericMapS


class GenericMap(metaclass=TemplateMeta):
    """An abstract `~collections.abc.Mapping` for use when sharing a
    map between C++ and Python.

    For compatibility with C++, ``GenericMap`` has the following
    restrictions:

        - all keys must be of the same type
        - values must be built-in types or subclasses of
          `lsst.afw.typehandling.Storable`. Almost any user-defined class in
          C++ or Python can have `~lsst.afw.typehandling.Storable` as a mixin.

    As a safety precaution, `~lsst.afw.typehandling.Storable` objects that are
    added from C++ may be copied when you retrieve them from Python, making it
    impossible to modify them in-place. This issue does not affect objects that
    are added from Python, or objects that are always passed by
    :cpp:class:`shared_ptr` in C++.
    """

    def __repr__(self):
        className = type(self).__name__
        return className + "({" + ", ".join("%r: %r" % (key, value) for key, value in self.items()) + "})"

    # Support equality with any Mapping, including dict
    # Not clear why Mapping.__eq__ doesn't work
    def __eq__(self, other):
        if len(self) != len(other):
            return False

        for key, value in self.items():
            try:
                if (value != other[key]):
                    return False
            except KeyError:
                return False
        return True

    # Easier than making GenericMap actually inherit from Mapping
    keys = Mapping.keys
    values = Mapping.values
    items = Mapping.items


GenericMap.register(str, GenericMapS)
Mapping.register(GenericMapS)


class MutableGenericMap(GenericMap):
    """An abstract `~collections.abc.MutableMapping` for use when sharing a
    map between C++ and Python.

    For compatibility with C++, ``MutableGenericMap`` has the following
    restrictions:

        - all keys must be of the same type
        - values must be built-in types or subclasses of
          `lsst.afw.typehandling.Storable`. Almost any user-defined class in
          C++ or Python can have `~lsst.afw.typehandling.Storable` as a mixin.

    As a safety precaution, `~lsst.afw.typehandling.Storable` objects that are
    added from C++ may be copied when you retrieve them from Python, making it
    impossible to modify them in-place. This issue does not affect objects that
    are added from Python, or objects that are always passed by
    :cpp:class:`shared_ptr` in C++.

    Notes
    -----
    Key-type specializations of ``MutableGenericMap`` are available as, e.g.,
    ``MutableGenericMap[str]``.
    """

    # Easier than making MutableGenericMap actually inherit from MutableMapping
    setdefault = MutableMapping.setdefault
    update = MutableMapping.update

    # MutableMapping.pop relies on implementation details of MutableMapping
    def pop(self, key, default=None):
        try:
            value = self[key]
            del self[key]
            return value
        except KeyError:
            if default is not None:
                return default
            else:
                raise


MutableGenericMap.register(str, MutableGenericMapS)
MutableMapping.register(MutableGenericMapS)


class AutoKeyMeta(TemplateMeta):
    """A metaclass for abstract mappings whose key type is implied by their
    constructor arguments.

    This metaclass requires that the mapping have a `dict`-like constructor,
    i.e., it takes a mapping or an iterable of key-value pairs as its first
    positional parameter.

    This class differs from `~lsst.utils.TemplateMeta` only in that the dtype
    (or equivalent) constructor keyword is optional. If it is omitted, the
    class will attempt to infer it from the first argument.
    """

    def __call__(cls, *args, **kwargs):  # noqa N805, non-self first param
        if len(cls.TEMPLATE_PARAMS) != 1:
            raise ValueError("AutoKeyMeta requires exactly one template parameter")
        dtypeKey = cls.TEMPLATE_PARAMS[0]
        dtype = kwargs.get(dtypeKey, None)

        # Try to infer dtype if not provided
        if dtype is None and len(args) >= 1:
            dtype = cls._guessKeyType(args[0])
            if dtype is not None:
                kwargs[dtypeKey] = dtype

        return super().__call__(*args, **kwargs)

    def _guessKeyType(cls, inputData):  # noqa N805, non-self first param
        """Try to infer the key type of a map from its input.

        Parameters
        ----------
        inputData : `~collections.abc.Mapping` or iterable of pairs
            Any object that can be passed to a `dict`-like constructor. Keys
            are assumed homogeneous (if not, a
            `~lsst.afw.typehandling.GenericMap` constructor will raise
            `TypeError` no matter what key type, if any, is provided).

        Returns
        -------
        keyType : `type`
            The type of the keys in ``inputData``, or `None` if the type could
            not be inferred.
        """
        if inputData:
            firstKey = None
            if isinstance(inputData, Mapping):
                # mapping to copy
                firstKey = iter(inputData.keys()).__next__()
            elif not isinstance(inputData, str):
                # iterable of key-value pairs
                try:
                    firstKey = iter(inputData).__next__()[0]
                except TypeError:
                    # Not an iterable of pairs
                    pass
            if firstKey:
                return type(firstKey)
        # Any other input is either empty or an invalid input to dict-like constructors
        return None
