#
# LSST Data Management System
# Copyright 2017 LSST/AURA.
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

__all__ = ["Key", "Field", "SchemaItem"]

from past.builtins import basestring
from builtins import str as futurestr
from future.utils import with_metaclass

import numpy as np
import fnmatch
import re
import collections
import astropy.units

from ...geom import Angle
from lsst.utils import continueClass, TemplateMeta

from .schema import _Key, _Field, _SchemaItem, Schema

# Objects we prefer to use over the C++ string name for
# Key/Field/SchemaItem types.
_dtypes = {
    "String": str,
    "B": np.uint8,
    "U": np.uint16,
    "I": np.int32,
    "L": np.int64,
    "F": np.float32,
    "D": np.float64,
    "Angle": Angle
}


class Key(with_metaclass(TemplateMeta, object)):
    pass


class Field(with_metaclass(TemplateMeta, object)):
    pass


class SchemaItem(with_metaclass(TemplateMeta, object)):
    pass


def _registerInstantiations(abc, types):
    """Iterate over a private dict (filled by template instantiations in C++)
    to register template instantiations a TemplateMeta ABCs.

    If an entry for the type string exists in _dtypes, we use that instead of
    the string as the key, and use the string as an alias.
    """
    for k, v in types.items():
        dtype = _dtypes.get(k, None)
        if dtype is not None:
            abc.register(dtype, v)
            abc.alias(k, v)
        else:
            abc.register(k, v)


# _Key, _Field, and _SchemaItem are {str: class} dicts populated by the C++
# wrappers.  The keys are a superset of those in _dtypes.
_registerInstantiations(Key, _Key)
_registerInstantiations(Field, _Field)
_registerInstantiations(SchemaItem, _SchemaItem)

# Also register `float->D` and 'futurestr->String' as aliases; can't include
# in _dtypes because we have (and prefer) np.float64/str there.
Key.alias(float, _Key["D"])
Field.alias(float, _Field["D"])
SchemaItem.alias(float, _SchemaItem["D"])
if futurestr is not str:
    Key.alias(futurestr, _Key["String"])
    Field.alias(futurestr, _Field["String"])
    SchemaItem.alias(futurestr, _SchemaItem["String"])


@continueClass  # noqa F811
class Schema:

    def getOrderedNames(self):
        """Return a list of field names in the order the fields were added to the Schema.
        """
        names = []

        def func(item):
            names.append(item.field.getName())
        self.forEach(func)
        return names

    def __iter__(self):
        """Iterate over the items in the Schema.
        """
        items = []
        self.forEach(items.append)
        return iter(items)

    def checkUnits(self, parse_strict='raise'):
        """Check that all units in the Schema are valid Astropy unit strings.
        """
        def func(item):
            astropy.units.Unit(item.field.getUnits(),
                               parse_strict=parse_strict)
        self.forEach(func)

    def addField(self, field, type=None, doc="", units="", size=None,
                 doReplace=False, parse_strict="raise"):
        """Add a field to the Schema.

        Parameters
        ----------
        field : str,Field
            The string name of the Field, or a fully-constructed Field object.
            If the latter, all other arguments besides doReplace are ignored.
        type\n : str,type
            The type of field to create.  Valid types are the keys of the
            afw.table.Field dictionary.
        doc : str
            Documentation for the field.
        unit : str
            Units for the field, or an empty string if unitless.
        size : int
            Size of the field; valid for string and array fields only.
        doReplace : bool
            If a field with this name already exists, replace it instead of
            raising pex.exceptions.InvalidParameterError.
        parse_strict : str
            One of 'raise' (default), 'warn', or 'strict', indicating how to
            handle unrecognized unit strings.  See also astropy.units.Unit.
        """
        if isinstance(field, basestring):
            field = Field[type](field, doc=doc, units=units,
                                size=size, parse_strict=parse_strict)
        return field._addTo(self, doReplace)

    def extract(self, *patterns, **kwds):
        """
        Extract a dictionary of {<name>: <schema-item>} in which the field names
        match the given shell-style glob pattern(s).

        Any number of glob patterns may be passed; the result will be the union of all
        the result of each glob considered separately.

        Additional Keyword Arguments
        ----------------------------
        regex : `str` or `re` pattern
            A regular expression to be used in addition to any glob patterns
            passed as positional arguments.  Note that this will be compared
            with re.match, not re.search.

        sub : `str`
            A replacement string (see re.MatchObject.expand) used to set the
            dictionary keys of any fields matched by regex.

        ordered : `bool`
            If True, a collections.OrderedDict will be returned instead of a
            standard dict, with the order corresponding to the definition
            order of the Schema. Default is False.
        """
        if kwds.pop("ordered", False):
            d = collections.OrderedDict()
        else:
            d = dict()
        regex = kwds.pop("regex", None)
        sub = kwds.pop("sub", None)
        if sub is not None and regex is None:
            raise ValueError(
                "'sub' keyword argument to extract is invalid without 'regex' argument")
        if kwds:
            raise ValueError(
                "Unrecognized keyword arguments for extract: %s" % ", ".join(kwds.keys()))
        for item in self:
            trueName = item.field.getName()
            names = [trueName]
            for alias, target in self.getAliasMap().items():
                if trueName.startswith(target):
                    names.append(trueName.replace(target, alias, 1))
            for name in names:
                if regex is not None:
                    m = re.match(regex, name)
                    if m is not None:
                        if sub is not None:
                            name = m.expand(sub)
                        d[name] = item
                        continue  # continue middle loop so we don't match the same name twice
                for pattern in patterns:
                    if fnmatch.fnmatchcase(name, pattern):
                        d[name] = item
                        break  # break inner loop so we don't match the same name twice
        return d
