from __future__ import absolute_import, division, print_function

from ._apCorrMap import ApCorrMap

ApCorrMap.values = lambda self: [self[name] for name in self.keys()]
ApCorrMap.items = lambda self: [(name, self[name]) for name in self.keys()]
ApCorrMap.__iter__ = lambda self: iter(self.keys())
