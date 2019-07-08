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

__all__ = []  # importing this module adds methods to BaseColumnView

import numpy as np

from lsst.utils import continueClass
from .._table import KeyFlag
from .baseColumnView import _BaseColumnViewBase

# We can't call this "BaseColumnView" because that's the typedef for
# "ColumnViewT<BaseRecord>". This is just a mostly-invisible implementation
# base class, so we use the same naming convention we use for those.


@continueClass  # noqa: F811
class _BaseColumnViewBase:

    def getBits(self, keys=None):
        """Get the bits associated with the specified keys.

        Parameters
        ----------
        key : `str`
            Key to retrieve. Unlike the C++ version, each key may be a
            field name or a key, and if keys is `None` then all bits
            are returned.

        Returns
        -------
        bits : `int`
             Integer array of the requested bitmask.
        """
        if keys is None:
            return self.getAllBits()
        arg = []
        for k in keys:
            if isinstance(k, str):
                arg.append(self.schema.find(k).key)
            else:
                arg.append(k)
        return self._getBits(arg)

    def __getitem__(self, key):
        """Get a column view; key may be a key object or the name of a field.
        """
        if isinstance(key, str):
            keyobj = self.schema.find(key).key
        else:
            keyobj = key
        return self._basicget(keyobj)

    get = __getitem__

    def __setitem__(self, key, value):
        """Set a full column to an array or scalar; key may be a key object or
        the name of a field.
        """
        self.get(key)[:] = value

    set = __setitem__

    def get_bool_array(self, key):
        """Get the value of a flag column as a boolean array; key must be a
        key object or the name of a field.

        Parameters
        ----------
        key : `lsst.afw.table.KeyFlag`
            Flag column to search for.

        Returns
        -------
        value : `list` of `bool`
            Array of booleans corresponding to the flag.

        Raises
        ------
        TypeError
            Raised if the key is not a KeyFlag.
        """
        if isinstance(key, KeyFlag):
            return self[key]
        raise TypeError("key={} not an lsst.afw.table.KeyFlag".format(key))

    def extract(self, *patterns, **kwds):
        """Extract a dictionary of {<name>: <column-array>} in which the field
        names match the given shell-style glob pattern(s).

        Any number of glob patterns may be passed (including none); the result
        will be the union of all the result of each glob considered
        separately.

        Note that extract("*", copy=True) provides an easy way to transform a
        row-major ColumnView into a possibly more efficient set of contiguous
        NumPy arrays.

        This routines unpacks `Flag` columns into full boolean arrays and
        covariances into dense (i.e. non-triangular packed) arrays with
        dimension (N,M,M), where N is the number of records and M is the
        dimension of the covariance matrix.  String fields are silently
        ignored.

        Parameters
        ----------
        patterns : Array of `str`
            List of glob patterns to use to select field names.
        kwds : `dict`
            Dictionary of additional keyword arguments.  May contain:
            - ``items`` : `list`
                The result of a call to self.schema.extract(); this
                will be used instead of doing any new matching, and
                allows the pattern matching to be reused to extract
                values from multiple records.  This keyword is
                incompatible with any position arguments and the
                regex, sub, and ordered keyword arguments.
            - ``where`` : array index expression
                Any expression that can be passed as indices to a
                NumPy array, including slices, boolean arrays, and
                index arrays, that will be used to index each column
                array.  This is applied before arrays are copied when
                copy is True, so if the indexing results in an
                implicit copy no unnecessary second copy is performed.
            - ``copy`` : `bool`
                If True, the returned arrays will be contiguous copies
                rather than strided views into the catalog.  This
                ensures that the lifetime of the catalog is not tied
                to the lifetime of a particular catalog, and it also
                may improve the performance if the array is used
                repeatedly. Default is False.
            - ``regex`` : `str` or `re` pattern
                A regular expression to be used in addition to any
                glob patterns passed as positional arguments.  Note
                that this will be compared with re.match, not
                re.search.
            - ``sub`` : `str`
                A replacement string (see re.MatchObject.expand) used
                to set the dictionary keys of any fields matched by
                regex.
            - ``ordered`` : `bool`
                If True, a collections.OrderedDict will be returned
                instead of a standard dict, with the order
                corresponding to the definition order of the
                Schema. Default is False.

        Returns
        -------
        d : `dict`
            Dictionary of extracted name-column array sets.

        Raises
        ------
        ValueError
            Raised if a list of ``items`` is supplied with additional
            keywords.
        """
        copy = kwds.pop("copy", False)
        where = kwds.pop("where", None)
        d = kwds.pop("items", None)
        # If ``items`` is given as a kwd, an extraction has already been performed and there shouldn't be
        # any additional keywords. Otherwise call schema.extract to load the
        # dictionary.
        if d is None:
            d = self.schema.extract(*patterns, **kwds).copy()
        elif kwds:
            raise ValueError(
                "kwd 'items' was specified, which is not compatible with additional keywords")

        def processArray(a):
            if where is not None:
                a = a[where]
            if copy:
                a = np.ascontiguousarray(a)
            return a

        # must use list because we might be adding/deleting elements
        for name, schemaItem in list(d.items()):
            key = schemaItem.key
            if key.getTypeString() == "String":
                del d[name]
            else:
                d[name] = processArray(self.get(schemaItem.key))
        return d
