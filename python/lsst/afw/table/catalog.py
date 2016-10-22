from __future__ import absolute_import, division, print_function
import numpy as np

from past.builtins import basestring

import lsst.pex.exceptions
from .schema import _suffixes

__all__ = ["addCatalogMethods"]


def addCatalogMethods(cls):
    """Add pure python methods to CatalogT classes or subclasses

    This should be called for BaseCatalog and other unsorted catalog classes (if any).
    For sorted catalogs such as SimpleCatalog and SourceCatalog,
    call addSortedCatalogMethods (which calls this) instead.
    """
    def _iter(self):
        for i in range(len(self)):
            yield self[i]

    def _getitem(self, key):
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

    def _setitem(self, key, value):
        """
        If ``key`` is an integer, set ``catalog[key]`` to ``value``. Otherwise select column ``key``
        and set it to ``value``.
        """
        try:
            # this works only for integer arguments (single record access)
            return self.set(key, value)
        except TypeError:
            self.columns[key] = value

    def _getColumns(self):
        """
        Call ``getColmnview()`` method.
        """
        return self.getColumnView()

    def _getattribute(self, name):
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

    def _extend(self, iterable, deep=False, mapper=None):
        """Append all records in the given iterable to the catalog.

        Arguments:
          iterable ------ any Python iterable containing records
          deep ---------- if True, the records will be deep-copied; ignored
                          if mapper is not None (that always implies True).
          mapper -------- a SchemaMapper object used to translate records
        """
        # self._columns = None
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
                    self.append(self.table.copyRecord(record, mapper))
                elif deep:
                    self.append(self.table.copyRecord(record))
                else:
                    self.append(record)

    def _copy(self, deep=False):
        """
        Copy a catalog (default is not a deep copy).
        """
        _type = type(self)
        if deep:
            table = self.table.clone()
            table.preallocate(len(self))
        else:
            table = self.table
        copy = _type(table)
        copy.extend(self, deep=deep)
        return copy

    def _searchTemplate(self, func, value, key):
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

    def _find(self, value, key):
        """Return the record for which record.get(key) == value

        If no such record is found, return None; if multiple such records are found,
        return one of them.

        The catalog must be sorted by this key before this method is called.
        """
        return _searchTemplate(self, "_find_", value, key)

    def _lower_bound(self, value, key):
        """Return the index of the first record for which record.get(key) >= value.

        If all elements in the catalog column are greater than or equal to the given
        value, returns 0; if all elements in the catalog are less than the given value,
        returns len(self).

        The catalog must be sorted by this key before this method is called.
        """
        return _searchTemplate(self, "_lower_bound_", value, key)

    def _upper_bound(self, value, key):
        """Return the record for which record.get(key) == value

        If all elements in the catalog column are greater than the given value,
        returns 0; if all elements in the catalog are less or equal to the given value,
        returns len(self).

        The catalog must be sorted by this key before this method is called.
        """
        return _searchTemplate(self, "_upper_bound_", value, key)

    def _between(self, lower, upper, key):
        """Return a slice object representing the records for which record.get(key)
        is between lower (inclusive) and upper(exclusive).

        The catalog must be sorted by this key before this method is called.
        """
        return slice(self.lower_bound(lower, key), self.upper_bound(upper, key))

    def _equal_range(self, value, key):
        """Return a slice object representing the records for which record.get(key)
        is equal to the given value

        The catalog must be sorted by this key before this method is called.
        """
        lower, upper = _searchTemplate(self, '_equal_range_', value, key)
        return slice(lower, upper)

    cls.__iter__ = _iter
    cls.__getitem__ = _getitem
    cls.__setitem__ = _setitem
    cls.__getattribute__ = _getattribute
    cls.extend = _extend
    cls.copy = _copy
    cls.find = _find
    cls.lower_bound = _lower_bound
    cls.upper_bound = _upper_bound
    cls.equal_range = _equal_range
    cls.between = _between
    cls.columns = property(_getColumns, doc="a column view of the catalog")
    cls.schema = property(cls.getSchema)
    cls.table = property(cls.getTable)


def _asAstropy(self, cls=None, copy=False, unviewable="copy"):
    """!
    Return an astropy.table.Table (or subclass thereof) view into this catalog.

    @param[in]   cls        Table subclass to use; None implies astropy.table.Table itself.
                            Use astropy.table.QTable to get Quantity columns.

    @param[in]  copy        Whether to copy data from the LSST catalog to the astropy table.
                            Not copying is usually faster, but can keep memory from being
                            freed if columns are later removed from the Astropy view.

    @param[in]  unviewable  One of the following options, indicating how to handle field types
                            (string and Flag) for which views cannot be constructed:
                              - 'copy' (default): copy only the unviewable fields.
                              - 'raise': raise ValueError if unviewable fields are present.
                              - 'skip': do not include unviewable fields in the Astropy Table.
                            This option is ignored if copy=True.
    """
    import astropy.table
    if cls is None:
        cls = astropy.table.Table
    if unviewable not in ("copy", "raise", "skip"):
        raise ValueError("'unviewable' must be one of 'copy', 'raise', or 'skip'")
    ps = self.getMetadata()
    meta = ps.toOrderedDict() if ps is not None else None
    columns = []
    items = self.schema.extract("*", ordered=True)
    for name, item in items.items():
        key = item.key
        unit = item.field.getUnits() or None  # use None instead of "" when empty
        if key.getTypeString() == "String":
            if not copy:
                if unviewable == "raise":
                    raise ValueError(
                        "Cannot extract string unless copy=True or unviewable='copy' or 'skip'.")
                elif unviewable == "skip":
                    continue
            data = np.zeros(len(self), dtype=np.dtype((str, key.getSize())))
            for i, record in enumerate(self):
                data[i] = record.get(key)
        elif key.getTypeString() == "Flag":
            if not copy:
                if unviewable == "raise":
                    raise ValueError(
                        "Cannot extract packed bit columns unless copy=True or unviewable='copy' or 'skip'."
                    )
                elif unviewable == "skip":
                    continue
            data = self.columns.get_bool_array(key)
        elif key.getTypeString() == "Angle":
            data = self.columns.get(key)
            unit = "radian"
            if copy:
                data = data.copy()
        else:
            data = self.columns.get(key)
            if copy:
                data = data.copy()
        columns.append(
            astropy.table.Column(
                data,
                name=item.field.getName(),
                unit=unit,
                description=item.field.getDoc()
            )
        )
    return cls(columns, meta=meta, copy=False)
