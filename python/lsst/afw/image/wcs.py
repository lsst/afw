from __future__ import absolute_import

import lsst.afw.table.io

from ._exposure import *
from ._wcs import *


def unpickleWcs(pick):
    import pickle
    exposure = pickle.loads(pick)
    return exposure.getWcs()

def __reduce__(self):
    import pickle
    exposure = ExposureU(1, 1)
    exposure.setWcs(self)
    return (unpickleWcs, (pickle.dumps(exposure),))

Wcs.__reduce__ = __reduce__
del __reduce__

