from __future__ import absolute_import, division, print_function

from past.builtins import basestring

from . import _fieldBase as fieldBase
from ._flag import Key_Flag
from .schema import aliases
from . import _key

# KeyBase_Flag is defined in the flag module, but to make the code more uniform we add it to the key module
_key.Key_Flag = Key_Flag

def key_eq(self, other):
    """
    Compare two keys. If the key types are the same they are checked for equality using the
    C++ layer. Otherwise return NotImplemented.
    """
    if type(other) != type(self):
        return NotImplemented
    return self._eq_impl(other)

def key_ne(self, other):
    """
    The inverse of key_eq
    """
    return not self == other

def init_keys(Key):
    """
    lsst::afw::table.Key<T> templates are wrapped as different objects in python, for example
    lsst::afw::table::Key<int> -> lsst.afw.table.Key_I, and
    lsst::afw::table::Key<Float> -> lsst.afw.table.Key_F.
    This function sets __eq__ and __ne__ for all of the key objects.
    """
    for k in Key:
        Key[k].__eq__ = key_eq
        Key[k].__ne__ = key_ne
        if isinstance(k, basestring) and k.startswith('Array'):
            Key[k].subfields = property(lambda self: tuple(range(self.getSize())))
            Key[k].subkeys = property(lambda self: tuple(self[i] for i in range(self.getSize())))
            Key[k].HAS_NAMED_SUBFIELDS = False

Key = {getattr(fieldBase, k).getTypeString(): getattr(_key, "Key_"+k.split('_')[1]) for k in dir(fieldBase)
    if k.startswith('FieldBase')}
for _k, _v in aliases.items():
    Key[_k] = Key[_v]
init_keys(Key)