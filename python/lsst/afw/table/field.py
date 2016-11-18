from __future__ import absolute_import, division, print_function

from . import _fieldBase
from .schema import aliases
from . import _field

# Map python data types to C++ Field_* objects
Field = {getattr(_fieldBase, k).getTypeString(): getattr(_field,"Field_"+k.split('_')[1])
            for k in dir(_fieldBase) if k.startswith('FieldBase')}

for _k, _v in aliases.items():
    Field[_k] = Field[_v]