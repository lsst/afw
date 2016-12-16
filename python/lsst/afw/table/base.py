from __future__ import absolute_import, division, print_function

from past.builtins import basestring

from .record import addRecordMethods
from .table import addTableMethods
from .catalog import addCatalogMethods
from ._base import BaseRecord, BaseTable, BaseCatalog

import numpy as np

__all__ = []  # import this module only for its side effects


def addBaseRecordMethods(cls):
    """Add pure Python methods to BaseRecord
    """
    addRecordMethods(cls)

    # The leading underscore avoids shadowing the standard Python `set` class.
    def set_(self, key, value):
        """Given a string or `lsst.afw.table.Key`, set the value of the field.
        """
        if isinstance(key, basestring):
            funcKey = self.getSchema().find(key).key
        else:
            funcKey = key

        try:
            prefix, suffix = type(funcKey).__name__.split("_")
        except ValueError:
            # Try using a functor
            return key.set(self, value)
        except Exception:
            raise TypeError("Argument to BaseRecord.set must be a string or Key.")

        if prefix != "Key":
            raise TypeError("Argument to BaseRecord.set must be a string or Key.")
        method = getattr(self, 'set'+suffix)
        return method(funcKey, value)
    cls.set = set_
    cls.__setitem__ = set_

    def get(self, key):
        """Given a string or `lsst.afw.table.Key`, get the value of the field.
        """
        if isinstance(key, basestring):
            funcKey = self.getSchema().find(key).key
        else:
            funcKey = key

        try:
            prefix, suffix = type(funcKey).__name__.split("_")
        except ValueError:
            # Try using a functor
            return key.get(self)
        except Exception:
            raise TypeError("Argument to BaseRecord.get must be a string or Key.")

        if prefix != "Key":
            raise TypeError("Argument to BaseRecord.get must be a string or Key.")
        method = getattr(self, '_get_'+suffix)
        return method(funcKey)
    cls.get = get

    def __getitem__(self, key):
        """Given a string or `lsst.afw.table.Key`, get the `BaseRecord` for the given key and return it.
        """
        if isinstance(key, basestring):
            return self[self.getSchema().find(key).key]

        # Try to get the item
        try:
            return self._getitem_(key)
        except TypeError:
            # Try to get a functor
            return key.get(self)
    cls.__getitem__ = __getitem__

    def extract(self, *patterns, **kwds):
        """Extract a dictionary of {<name>: <field-value>} in which the field names
        match the given shell-style glob pattern(s).

        Any number of glob patterns may be passed; the result will be the union of all
        the result of each glob considered separately.
        Additional optional arguments may be passed as keywords:
          items ------ The result of a call to self.schema.extract(); this will be used instead
                       of doing any new matching, and allows the pattern matching to be reused
                       to extract values from multiple records.  This keyword is incompatible
                       with any position arguments and the regex, sub, and ordered keyword
                       arguments.
          split ------ If True, fields with named subfields (e.g. points) will be split into
                       separate items in the dict; instead of {"point": lsst.afw.geom.Point2I(2,3)},
                       for instance, you'd get {"point.x": 2, "point.y": 3}.
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
    cls.extract = extract

    cls.schema = property(cls.getSchema)


def addBaseCatalogMethods(cls):
    """Add pure Python methods to BaseCatalog
    """
    addCatalogMethods(cls)

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
                    name=item.field.getName(),
                    unit=unit,
                    description=item.field.getDoc()
                )
            )
        return cls(columns, meta=meta, copy=False)
    cls.asAstropy = asAstropy


addTableMethods(BaseTable)
addBaseRecordMethods(BaseRecord)
addBaseCatalogMethods(BaseCatalog)
