from __future__ import absolute_import, division, print_function

from lsst.utils import continueClass
from ._apCorrMap import ApCorrMap

@continueClass
class ApCorrMap:

    def keys(self):
        for item in self.items():
            yield item[0]

    def values(self):
        for item in self.items():
            yield item[1]

    __iter__ = keys
