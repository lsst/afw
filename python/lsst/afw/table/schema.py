from __future__ import absolute_import, division, print_function

from past.builtins import basestring
from builtins import str as futurestr
import collections
import fnmatch
import re
import warnings

import numpy as np
import astropy.units

from ..geom import Angle
from . import _fieldBase
from ._flag import FieldBase_Flag
from . import detail
import lsst.pex.exceptions

from ._schema import Schema, SubSchema

__all__ = []  # import only for the side effects

# FieldBase_Flag is defined in the flag module, but to make the code more uniform we add it to fieldBase
_fieldBase.FieldBase_Flag = FieldBase_Flag

# Types defined for FieldBase objects
_suffixes = {getattr(_fieldBase, k).getTypeString(): k.split('_')[1] for k in dir(_fieldBase)
             if k.startswith('FieldBase')}

# Map python types to C++ type identifiers
aliases = {
    str: "String",
    futurestr: "String",
    np.uint16: "U",
    np.int32: "I",
    np.int64: "L",
    np.float32: "F",
    np.float64: "D",
    Angle: "Angle",
}


def addSchemaMethods(cls):
    """Add pure Python methods to the Schema class
    """

    def addField(self, field, type=None, doc="", units="", size=None, doReplace=False, parse_strict='raise'):
        """
        The C++ Schema class has a templated addField function, which is exposed to python via
        Schema._addField_Suffix, where suffix is the type exposed to python, for example
        Schema._addField_I for 32 bit integers.

        This method finds the correct type for the current field and calls the appropriate C++ member.
        """
        # Check for astropy compatible unit string
        astropy.units.Unit(units, parse_strict=parse_strict)
        if type is None:
            try:
                prefix, suffix = __builtins__['type'](field).__name__.split("_")
            except Exception:
                raise TypeError("First argument to Schema.addField must be a Field if 'type' is not given.")
            if prefix != "Field":
                raise TypeError("First argument to Schema.addField must be a Field if 'type' is not given.")
            attr = "_addField_" + suffix
            method = getattr(self, attr)
            return method(field)
        if not isinstance(type, basestring):
            type = aliases[type]
        suffix = _suffixes[type]
        attr = "_addField_" + suffix
        method = getattr(self, attr)
        if size is None:
            size = getattr(_fieldBase, "FieldBase_" + suffix)()
        else:
            size = getattr(_fieldBase, "FieldBase_" + suffix)(size)
        return method(field, doc, units, size, doReplace)
    cls.addField = addField

    def find(self, k):
        """
        The C++ Schema class has a templated find function, which is exposed to python via
        Schema._find_Suffix, where suffix is the type exposed to python, for example
        Schema._find_I for 32 bit integers.

        This method finds the correct type for the current field and calls the appropriate C++ member.
        """
        if not isinstance(k, basestring):

            try:
                prefix, suffix = type(k).__name__.split("_")
            except Exception:
                raise TypeError("Argument to Schema.find must be a string or Key.")
            if prefix != "Key":
                raise TypeError("Argument to Schema.find must be a string or Key.")
            attr = "_find_" + suffix
            method = getattr(self, attr)
            return method(k)
        for suffix in _suffixes.values():
            attr = "_find_" + suffix
            method = getattr(self, attr)
            try:
                return method(k)
            except (lsst.pex.exceptions.TypeError, lsst.pex.exceptions.NotFoundError):
                pass
        raise KeyError("Field '%s' not found in Schema." % k)
    cls.find = find

    def checkUnits(self, parse_strict='raise'):
        """
        Check all of the SchemaItems in a Schema have valid units
        """
        for schemaItem in self:
            astropy.units.Unit(schemaItem.getField().getUnits(), parse_strict=parse_strict)
    cls.checkUnits = checkUnits

    def __contains__(self, key):
        """
        Check whether or not a key is found in a Schema.
        """
        try:
            self.find(key)
            return True
        except:
            return False
    cls.__contains__ = __contains__

    def schemaComparisonAnd(self, other):
        """
        Enumerated types do not have __and__ and __rand__ operators defined, so for
        SchemaComparisons we cast the Schema.ComparisonFlag enumerated types into integers
        to do a bitwise comparison.
        """
        compare = other
        if isinstance(other, Schema.ComparisonFlags):
            compare = int(other)
        return int(self) & compare
    cls.ComparisonFlags.__and__ = schemaComparisonAnd

    def schemaComparisonRand(self, other):
        """
        Enumerated types do not have __and__ and __rand__ operators defined, so for
        SchemaComparisons we cast the Schema.ComparisonFlag enumerated types into integers
        to do a bitwise comparison.
        """
        compare = other
        if isinstance(other, Schema.ComparisonFlags):
            compare = int(other)
        return compare & int(self)
    cls.ComparisonFlags.__rand__ = schemaComparisonRand

    def schemaComparisonInvert(self):
        """
        Enumerated types does not have the unary ~ (__invert__) operator defined.
        For SchemaComparisons we cast a Schema.ComparisonFlags enumerated type into an integer,
        invert it, and keep only the first 8 bits (since Schema.ComparisonFlags is an 8 bit bitmask).
        """
        return ~int(self) & 0xFF
    cls.ComparisonFlags.__invert__ = schemaComparisonInvert

    class _FieldNameExtractor:

        """
        Schema.forEach requires a functor, which is applied to each item.
        In this case the method "getOrderedNames" uses this functor to extract the
        ordered field names from the Schema.
        """

        def __init__(self):
            self.fieldNames = []

        def __call__(self, item):
            return self.fieldNames.append(item.field.getName())

    class _ItemExtractor:

        """
        Schema.forEach requires a functor, which is applied to each item.
        This functor creates a list of all the items in the Schema.
        """

        def __init__(self):
            self.items = []

        def __call__(self, item):
            return self.items.append(item)

    def getOrderedNames(self):
        """
        Extract the ordered field names from a Schema. This uses the _FieldNameExtractor
        functor in Schema._forEach to get the name of each field.
        """
        nameFunctor = _FieldNameExtractor()
        self._forEach(nameFunctor)
        return nameFunctor.fieldNames
    cls.getOrderedNames = getOrderedNames

    def __iter__(self):
        """
        Iterator for a schema that returns a list of SchemaItems
        """
        itemFunctor = _ItemExtractor()
        self._forEach(itemFunctor)
        return iter(itemFunctor.items)
    cls.__iter__ = __iter__

    def _schemaExtract(self, *patterns, **kwds):
        """
        Extract a dictionary of {<name>: <schema-item>} in which the field names
        match the given shell-style glob pattern(s).
        Any number of glob patterns may be passed; the result will be the union of all
        the result of each glob considered separately.
        Additional optional arguments may be passed as keywords:
          regex ------ A regular expression to be used in addition to any glob patterns passed
                       as positional arguments.  Note that this will be compared with re.match,
                       not re.search.
          sub -------- A replacement string template (see re.MatchObject.expand) used to set the
                       dictionary keys of any fields matched by regex.  The field name in the
                       SchemaItem is not modified.
          ordered----- If True, a collections.OrderedDict will be returned instead of a standard
                       dict, with the order corresponding to the definition order of the Schema.
        """
        if kwds.pop("ordered", False):
            d = collections.OrderedDict()
        else:
            d = dict()
        regex = kwds.pop("regex", None)
        sub = kwds.pop("sub", None)
        if sub is not None and regex is None:
            raise ValueError("'sub' keyword argument to extract is invalid without 'regex' argument")
        if kwds:
            raise ValueError("Unrecognized keyword arguments for extract: %s" % ", ".join(kwds.keys()))
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
    cls.extract = _schemaExtract


def addSubSchemaMethods(cls):
    """Add pure Python methods to the SubSchema class
    """
    def runSubTemplateFunc(subSchema, func, *args):
        """
        The C++ SubSchema class has templated functions, which are exposed to python via
        SubSchema._function_suffix, where function is the C++ function and
        suffix is the type exposed to python.
        For example, SubSchema._addField_I funs SubSchema.addField for 32 bit integers.

        This function tries all of the available template types for a given function and
        raises an Exception if none of the functions have matching types.

        Parameters
        ----------
        subSchema: lsst.afw.table.SubSchema
            Subschema containing the method
        func: string
            Name of the function to run. For example to execute ``SubSchema.find`` use
            ``runSubTemplateFunc(subSchema, "find")``
        args: tuple (optional)
            Arguments to pass to the function.
        """
        for suffix in _suffixes.values():
            attr = "_"+func+"_" + suffix
            method = getattr(subSchema, attr)
            try:
                return method(*args)
            except (lsst.pex.exceptions.TypeError, lsst.pex.exceptions.NotFoundError):
                pass
        raise KeyError("Field '%s' not found in Schema." % subSchema.getPrefix())

    def find(self, key):
        """
        The C++ SubSchema class has a templated find function, which is exposed to python via
        SubSchema._find_Suffix, where suffix is the type exposed to python, for example
        SubSchema._find_I for 32 bit integers.

        SubSchema.find takes a lsst.afw.table.Key as an input and returns the lsst.afw.table.SchemaItem
        corresponding to the key.
        """
        return runSubTemplateFunc(self, "find", key)
    cls.find = find

    def asField(self):
        """
        The C++ SubSchema class has a templated find function, which is exposed to python via
        SubSchema._asField_suffix, where suffix is the type exposed to python, for example
        SubSchema._asField_I for 32 bit integers.

        SubSchema.asField casts the Schema as an lsst.afw.table.Field.
        """
        return runSubTemplateFunc(self, "asField")
    cls.asField = asField

    def asKey(self):
        """
        The C++ SubSchema class has a templated find function, which is exposed to python via
        SubSchema._asKey_Suffix, where suffix is the type exposed to python, for example
        SubSchema._asKey_I for 32 bit integers.

        SubSchema.asKey casts the Schema as an lsst.afw.table.Key.
        """
        return runSubTemplateFunc(self, "asKey")
    cls.asKey = asKey


def addSchemaItemMethods(cls):
    """Add pure python methods to a SchemaItem<X> class
    """

    def __getitem__(self, i):
        if i == 0:
            return self.key
        elif i == 1:
            return self.field
        raise IndexError("SchemaItem index must be 0 or 1")
    cls.__getitem__ = __getitem__

    def __str__(self):
        return str(tuple(self))
    cls.__str__ = __str__

    def __repr__(self):
        return "SchemaItem(%r, %r)" % (self.key, self.field)
    cls.__repr_ = __repr__

    def getKey(self):
        warnings.warn("getKey() is deprecated; use key instead", DeprecationWarning)
        return self.key
    cls.getKey = getKey

    def getField(self):
        warnings.warn("getField() is deprecated; use field instead", DeprecationWarning)
        return self.field
    cls.getField = getField

addSchemaMethods(Schema)

addSubSchemaMethods(SubSchema)

_SchemaItemDict = {getattr(_fieldBase, k).getTypeString(): getattr(detail, "SchemaItem_"+k.split('_')[1])
                   for k in dir(_fieldBase) if k.startswith('FieldBase')}
for _k, _v in aliases.items():
    _SchemaItemDict[_k] = _SchemaItemDict[_v]

for schemaItemClass in _SchemaItemDict.values():
    addSchemaItemMethods(schemaItemClass)
