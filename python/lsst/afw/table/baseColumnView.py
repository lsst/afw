from __future__ import absolute_import, division, print_function

from past.builtins import basestring

import numpy as np

from ._flag import Key_Flag
from ._baseColumnView import BaseColumnView, BitsColumn

__all__ = []  # importing this module adds methods to BaseColumnView


def addBaseColumnViewMethods(cls):
    def getBits(self, keys=None):
        """Get the bits associated with the specified keys

        Unlike the C++ version, each key may be a field name or a key,
        and if keys is None then all bits are returned.
        """
        if keys is None:
            return self.getAllBits()
        arg = []
        for k in keys:
            if isinstance(k, basestring):
                arg.append(self.schema.find(k).key)
            else:
                arg.append(k)
        return self._getBits(arg)
    cls.getBits = getBits

    def __getitem__(self, key):
        """Get a column view; key may be a key object or the name of a field.
        """
        if isinstance(key, basestring):
            keyobj = self.schema.find(key).key
        else:
            keyobj = key
        return self._basicget(keyobj)
    cls.get = __getitem__
    cls.__getitem__ = __getitem__

    def __setitem__(self, key, value):
        """Set a full column to an array or scalar; key may be a key object or the name of a field.
        """
        self.get(key)[:] = value
    cls.set = __setitem__
    cls.__setitem__ = __setitem__

    def get_bool_array(self, key):
        """Get the value of a flag column as a boolean array; key must be a key object or the name of a field.
        """
        if isinstance(key, Key_Flag):
            return self[key]
        raise TypeError("key={} not an lsst.afw.table.Key_Flag".format(key))
    cls.get_bool_array = get_bool_array

    def extract(self, *patterns, **kwds):
        """Extract a dictionary of {<name>: <column-array>} in which the field names
        match the given shell-style glob pattern(s).

        Any number of glob patterns may be passed (including none); the result will be the union of all
        the result of each glob considered separately.
        Note that extract("*", copy=True) provides an easy way to transform a row-major
        ColumnView into a possibly more efficient set of contiguous NumPy arrays.
        This routines unpacks Flag columns into full boolean arrays and covariances into dense
        (i.e. non-triangular packed) arrays with dimension (N,M,M), where N is the number of
        records and M is the dimension of the covariance matrix.  Fields with named subfields
        (e.g. points) are always split into separate dictionary items, as is done in
        BaseRecord.extract(..., split=True).  String fields are silently ignored.
        Additional optional arguments may be passed as keywords:
          items ------ The result of a call to self.schema.extract(); this will be used instead
                       of doing any new matching, and allows the pattern matching to be reused
                       to extract values from multiple records.  This keyword is incompatible
                       with any position arguments and the regex, sub, and ordered keyword
                       arguments.
          where ------ Any expression that can be passed as indices to a NumPy array, including
                       slices, boolean arrays, and index arrays, that will be used to index
                       each column array.  This is applied before arrays are copied when
                       copy is True, so if the indexing results in an implicit copy no
                       unnecessary second copy is performed.
          copy ------- If True, the returned arrays will be contiguous copies rather than strided
                       views into the catalog.  This ensures that the lifetime of the catalog is
                       not tied to the lifetime of a particular catalog, and it also may improve
                       the performance if the array is used repeatedly.
                       Default is False.
          regex ------ A regular expression to be used in addition to any glob patterns passed
                       as positional arguments.  Note that this will be compared with re.match,
                       not re.search.
          sub -------- A replacement string (see re.MatchObject.expand) used to set the
                       dictionary keys of any fields matched by regex.
          ordered----- If True, a collections.OrderedDict will be returned instead of a standard
                       dict, with the order corresponding to the definition order of the Schema.
                       Default is False.
        """
        copy = kwds.pop("copy", False)
        where = kwds.pop("where", None)
        d = kwds.pop("items", None)
        # If ``items`` is given as a kwd, an extraction has already been performed and there shouldn't be
        # any additional keywords. Otherwise call schema.extract to load the dictionary.
        if d is None:
            d = self.schema.extract(*patterns, **kwds).copy()
        elif kwds:
            raise ValueError("kwd 'items' was specified, which is not compatible with additional keywords")

        def processArray(a):
            if where is not None:
                a = a[where]
            if copy:
                a = np.ascontiguousarray(a)
            return a

        for name, schemaItem in list(d.items()):  # must use list because we might be adding/deleting elements
            key = schemaItem.key
            if key.getTypeString() == "String":
                del d[name]
            else:
                d[name] = processArray(self.get(schemaItem.key))
        return d
    cls.extract = extract

    cls.array = property(BitsColumn.getArray)

addBaseColumnViewMethods(BaseColumnView)
