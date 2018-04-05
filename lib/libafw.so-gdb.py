import os
import sys
import lsst.utils
import gdb
#
# Adjust the load path to include lsst.gdb, bypassing the regular lsstimport mechanism as
# the version of python running within gdb may not be the same as we are using for lsst processing
#
try:
    afwDir = lsst.utils.getPackageDir('afw')
except Exception:
    pass
else:
    printerDir = os.path.join(afwDir, "python", "lsst", "gdb")
    if printerDir not in sys.path:
        sys.path.append(printerDir)

import afw.printers

afw.printers.register(gdb.current_objfile())
