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

__all__ = ["Catalog"]

from future.utils import with_metaclass

import numpy as np

from lsst.utils import continueClass, TemplateMeta
from .base import BaseRecord, BaseCatalog


@continueClass
class BaseRecord:

    def extract(self, *patterns, **kwds):
        """Extract a dictionary of {<name>: <field-value>} in which the field names
        match the given shell-style glob pattern(s).

        Any number of glob patterns may be passed; the result will be the union of all
        the result of each glob considered separately.

        Parameters
        ----------

        items : `dict`
            The result of a call to self.schema.extract(); this will be used
            instead of doing any new matching, and allows the pattern matching
            to be reused to extract values from multiple records.  This
            keyword is incompatible with any position arguments and the regex,
            sub, and ordered keyword arguments.

        split : `bool`
            If True, fields with named subfields (e.g. points) will be split
            into separate items in the dict; instead of {"point":
            lsst.afw.geom.Point2I(2,3)}, for instance, you'd get {"point.x":
            2, "point.y": 3}. Default is False.

        regex : `str` or `re` pattern object
            A regular expression to be used in addition to any glob patterns
            passed as positional arguments.  Note that this will be compared
            with re.match, not re.search.

        sub : `str`
            A replacement string (see `re.MatchObject.expand`) used to set the
            dictionary keys of any fields matched by regex.

        ordered : `bool`
            If `True`, a `collections.OrderedDict` will be returned instead of
            a standard dict, with the order corresponding to the definition
            order of the `Schema`. `Default is False`.
        """
        d = kwds.pop("items", None)
        split = kwds.pop("split", False)
        if d is None:
            d = self.schema.extract(*patterns, **kwds).copy()
        elif kwds:
            raise ValueError("Unrecognized keyword arguments for extract: %s" % ", ".join(kwds.keys()))
        for name, schemaItem in list(d.items()):  # must use list because we might be adding/deleting elements
            key = schemaItem.key
            if split and key.HAS_NAMED_SUBFIELDS:
                for subname, subkey in zip(key.subfields, key.subkeys):
                    d["%s.%s" % (name, subname)] = self.get(subkey)
                del d[name]
            else:
                d[name] = self.get(schemaItem.key)
        return d


class Catalog(with_metaclass(TemplateMeta, object)):

    def getColumnView(self):
        self._columns = self._getColumnView()
        return self._columns

    def __getColumns(self):
        if not hasattr(self, "_columns") or self._columns is None:
            self._columns = self._getColumnView()
        return self._columns
    columns = property(__getColumns, doc="a column view of the catalog")

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
            pass
        # when record access fails, try column access
        return self.columns[key]

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
            pass
        self.columns[key] = value

    def __delitem__(self, key):
        self._columns = None
        self._delitem_(key)

    def append(self, record):
        self._columns = None
        self._append(record)

    def insert(self, key, value):
        self._columns = None
        self._insert(key, value)

    def clear(self):
        self._columns = None
        self._clear()

    def addNew(self):
        self._columns = None
        return self._addNew()

    def cast(self, type_, deep=False):
        """Return a copy of the catalog with the given type, optionally
        cloning the table and deep-copying all records if deep==True.
        """
        if deep:
            table = self.table.clone()
            table.preallocate(len(self))
        else:
            table = self.table
        copy = type_(table)
        copy.extend(self, deep=deep)
        return copy

    def copy(self, deep=False):
        """
        Copy a catalog (default is not a deep copy).
        """
        return self.cast(type(self), deep)

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
                    self._append(record)

    def __reduce__(self):
        import lsst.afw.fits
        return lsst.afw.fits.reduceToFits(self)

    def asAstropy(self, cls=None, copy=False, unviewable="copy"):
        """!Return an astropy.table.Table (or subclass thereof) view into this catalog.

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
            raise ValueError("'unviewable'=%r must be one of 'copy', 'raise', or 'skip'" % (unviewable,))
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
                        raise ValueError("Cannot extract string "
                                         "unless copy=True or unviewable='copy' or 'skip'.")
                    elif unviewable == "skip":
                        continue
                data = np.zeros(len(self), dtype=np.dtype((str, key.getSize())))
                for i, record in enumerate(self):
                    data[i] = record.get(key)
            elif key.getTypeString() == "Flag":
                if not copy:
                    if unviewable == "raise":
                        raise ValueError("Cannot extract packed bit columns "
                                         "unless copy=True or unviewable='copy' or 'skip'.")
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
                    name=name,
                    unit=unit,
                    description=item.field.getDoc()
                )
            )
        return cls(columns, meta=meta, copy=False)

    def __getattr__(self, name):
        # Catalog forwards unknown method calls to its table and column view
        # for convenience.  (Feature requested by RHL; complaints about magic
        # should be directed to him.)
        if name == "_columns":
            self._columns = None
            return None
        try:
            return getattr(self.table, name)
        except AttributeError:
            return getattr(self.columns, name)


Catalog.register("Base", BaseCatalog)
