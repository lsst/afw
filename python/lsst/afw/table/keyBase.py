from __future__ import absolute_import, division, print_function

from . import _keyBase

def kbGetItem(self, key):
    if isinstance(key, slice):
        return self.slice(key.start, key.stop)
    else:
        return self._getitem_(key)

for _kb in dir(_keyBase):
    if _kb.startswith("KeyBase"):
        getattr(_keyBase, _kb).subfields = None
        getattr(_keyBase, _kb).subkeys = None
        getattr(_keyBase, _kb).HAS_NAMED_SUBFIELDS = False
        getattr(_keyBase, _kb).__getitem__ = kbGetItem
