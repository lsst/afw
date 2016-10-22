from __future__ import absolute_import, division, print_function

from ._aliasMap import AliasMap

def _items(self):
    """
    pybind11 has a ``make_iterator`` method that greatly reduces the amount of code needed to
    iterate over an AliasMap and it's keys. This function remains to avoid changes to the AliasMap API
    but in the future could be removed if desired.
    """
    return [a for a in self]

AliasMap.items = _items