from __future__ import absolute_import, division, print_function
import numpy as np

from past.builtins import basestring

import lsst.pex.exceptions
from .schema import _suffixes

__all__ = ["addCatalogMethods", "searchTemplate"]


def addCatalogMethods(cls):
    """Add pure python methods to CatalogT classes or subclasses

    This should be called for BaseCatalog and other unsorted catalog classes (if any).
    For sorted catalogs such as SimpleCatalog and SourceCatalog,
    call addSortedCatalogMethods (which calls this) instead.
    """
    def getColumnView(self):
        self._columns = self._getColumnView()
        return self._columns
    cls.getColumnView = getColumnView

    def __getColumns(self):
        if not hasattr(self, "_columns") or self._columns is None:
            self._columns = self._getColumnView()
        return self._columns
    cls.columns = property(__getColumns, doc="a column view of the catalog")

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
    cls.__iter__ = __iter__

    def __getitem__(self, key):
        """Return the record at index key if key is an integer,
        return a column if key is a string field name or Key,
        or return a subset of the catalog if key is a slice
        or boolean NumPy array.
        """
        if type(key) is slice:
            (start, stop, step) = (key.start, key.stop, key.step)
            if step is None:
                step = 1
            if start is None:
                start = 0
            if stop is None:
                stop = len(self)
            return self.subset(start, stop, step)
        elif isinstance(key, np.ndarray):
            return self.subset(key)
        try:
            # this works only for integer arguments (single record access)
            return self._getitem_(key)
        except TypeError:
            # when record access fails, try column access
            return self.columns[key]
    cls.__getitem__ = __getitem__

    def __setitem__(self, key, value):
        """
        If ``key`` is an integer, set ``catalog[key]`` to ``value``. Otherwise select column ``key``
        and set it to ``value``.
        """
        self._columns = None
        try:
            # this works only for integer arguments (single record access)
            return self.set(key, value)
        except TypeError:
            self.columns[key] = value
    cls.__setitem__ = __setitem__

    def __delitem__(self, key):
        self._columns = None
        self._delitem_(key)
    cls.__delitem__ = __delitem__

    def append(self, record):
        self._columns = None
        self._append(record)
    cls.append = append

    def insert(self, key, value):
        self._columns = None
        self._insert(key, value)
    cls.insert = insert

    def clear(self):
        self._columns = None
        self._clear()
    cls.clear = clear

    def addNew(self):
        self._columns = None
        return self._addNew()
    cls.addNew = addNew

    def cast(self, type_, deep=False):
        """Return a copy of the catalog with the given type, optionally
        cloning the table and deep-copying all records if deep==True.
        """
        if deep:
            table = self.table.clone()
            table.preallocate(len(self))
        else:
            table = self.table
        newTable = table.cast(type_.Table)
        copy = type_(newTable)
        copy.extend(self, deep=deep)
        return copy
    cls.cast = cast

    def copy(self, deep=False):
        """
        Copy a catalog (default is not a deep copy).
        """
        return self.cast(type(self), deep)
    cls.copy = copy

    def __getattribute__(self, name):
        # Catalog forwards unknown method calls to its table and column view
        # for convenience.  (Feature requested by RHL; complaints about magic
        # should be directed to him.)
        # We have to use __getattribute__ because SWIG overrides __getattr__.
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            # self._columns is created the when self.columns is accessed -
            # looking for it in self.columns below would trigger infinite
            # recursion.
            # Test for "this" as a temporary workaround for ticket 7377
            if name in ("_columns", "this"):
                raise
        try:
            return getattr(self.table, name)
        except AttributeError:
            return getattr(self.columns, name)
    cls.__getattribute__ = __getattribute__

    def extend(self, iterable, deep=False, mapper=None):
        """Append all records in the given iterable to the catalog.

        Arguments:
          iterable ------ any Python iterable containing records
          deep ---------- if True, the records will be deep-copied; ignored
                          if mapper is not None (that always implies True).
          mapper -------- a SchemaMapper object used to translate records
        """
        self._columns = None
        # We can't use isinstance here, because the SchemaMapper symbol isn't available
        # when this code is part of a subclass of Catalog in another package.
        if type(deep).__name__ == "SchemaMapper":
            mapper = deep
            deep = None
        if isinstance(iterable, type(self)):
            if mapper is not None:
                self._extend(iterable, mapper)
            else:
                self._extend(iterable, deep)
        else:
            for record in iterable:
                if mapper is not None:
                    self._append(self.table.copyRecord(record, mapper))
                elif deep:
                    self._append(self.table.copyRecord(record))
                else:
                    self._append(record.cast(self.Record))
    cls.extend = extend

    def __reduce__(self):
        import lsst.afw.fits
        return lsst.afw.fits.reduceToFits(self)
    cls.__reduce__ = __reduce__

    def find(self, value, key):
        """Return the record for which record.get(key) == value

        If no such record is found, return None; if multiple such records are found,
        return one of them.

        The catalog must be sorted by this key before this method is called.
        """
        return searchTemplate(self, "_find_", value, key)
    cls.find = find

    def lower_bound(self, value, key):
        """Return the index of the first record for which record.get(key) >= value.

        If all elements in the catalog column are greater than or equal to the given
        value, returns 0; if all elements in the catalog are less than the given value,
        returns len(self).

        The catalog must be sorted by this key before this method is called.
        """
        return searchTemplate(self, "_lower_bound_", value, key)
    cls.lower_bound = lower_bound

    def upper_bound(self, value, key):
        """Return the record for which record.get(key) == value

        If all elements in the catalog column are greater than the given value,
        returns 0; if all elements in the catalog are less or equal to the given value,
        returns len(self).

        The catalog must be sorted by this key before this method is called.
        """
        return searchTemplate(self, "_upper_bound_", value, key)
    cls.upper_bound = upper_bound

    def between(self, lower, upper, key):
        """Return a slice object representing the records for which record.get(key)
        is between lower (inclusive) and upper(exclusive).

        The catalog must be sorted by this key before this method is called.
        """
        return slice(self.lower_bound(lower, key), self.upper_bound(upper, key))
    cls.between = between

    def equal_range(self, value, key):
        """Return a slice object representing the records for which record.get(key)
        is equal to the given value

        The catalog must be sorted by this key before this method is called.
        """
        lower, upper = searchTemplate(self, '_equal_range_', value, key)
        return slice(lower, upper)
    cls.equal_range = equal_range


def searchTemplate(self, func, value, key):
    if not isinstance(key, basestring):
        try:
            prefix, suffix = type(key).__name__.split("_")
        except Exception:
            raise TypeError("Argument to Catalog.find must be a string or Key.")
        if prefix != "Key":
            raise TypeError("Argument to Catalog.find must be a string or Key.")
        attr = func + suffix
        method = getattr(self, attr)
        return method(value, key)
    for suffix in _suffixes.values():
        attr = func + suffix
        method = getattr(self, attr)
        try:
            return method(value, key)
        except (lsst.pex.exceptions.TypeError, lsst.pex.exceptions.NotFoundError):
            pass
    raise KeyError("Record '%s' not found in Catalog." % key)
