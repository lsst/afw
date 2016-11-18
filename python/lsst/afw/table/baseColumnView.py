from __future__ import absolute_import, division, print_function

from past.builtins import basestring

import numpy as np

from ._baseColumnView import BaseColumnView

def _get(self, key):
    """ If the key is a string, search the Schema for the correct key object and return the 
    column view
    """
    if isinstance(key, basestring):
        return self[self.schema.find(key).key]
    return self[key]

def _setitem_(self, key, value):
    """ Set the values of a column
    """
    self.get(key)[:] = value

def _set(self, key, value):
    """
    Set the value of a column
    """
    self[key] = value

def _extract(self, *patterns, **kwds):
    """ Extract a dictionary of {<name>: <column-array>} in which the field names
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

BaseColumnView.table = property(BaseColumnView.getTable)
BaseColumnView.schema = property(BaseColumnView.getSchema)
BaseColumnView.get = _get
BaseColumnView.set = _set
BaseColumnView.__setitem__ = _setitem_
BaseColumnView.extract = _extract