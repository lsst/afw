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

from collections.abc import Mapping, MutableMapping

from lsst.utils import continueClass
from ._typehandling import GenericMapS, MutableGenericMapS

Mapping.register(GenericMapS)

@continueClass  # noqa F811
class GenericMapS:
    def __repr__(self):
        className = type(self).__name__
        return className + "({" + ", ".join("%r: %r" % (key, value) for key, value in self.items()) + "})"

    def __iter__(self):
        for key in self._keys():
            yield key

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


MutableMapping.register(MutableGenericMapS)

@continueClass  # noqa F811
class MutableGenericMapS:
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
