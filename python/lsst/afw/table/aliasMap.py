from __future__ import absolute_import, division, print_function

from ._aliasMap import AliasMap


def _keys(self):
    """Return an iterator over AliasMap keys"""
    for key, value in self.items():
        yield key


def _values(self):
    """Return an iterator over AliasMap values"""
    for key, value in self.items():
        yield value

AliasMap.__iter__ = _keys

AliasMap.keys = _keys

AliasMap.values = _values
