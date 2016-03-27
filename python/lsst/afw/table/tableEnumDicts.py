#
# LSST Data Management System
# See the COPYRIGHT and LICENSE files in the top-level directory of this
# package for notices and licensing terms.
#
from __future__ import absolute_import, division
from . import tableLib

ReadoutCornerValNameDict = {
    tableLib.LL: "LL",
    tableLib.LR: "LR",
    tableLib.UR: "UR",
    tableLib.UL: "UL",
}
ReadoutCornerNameValDict = dict((val, key) for key, val in ReadoutCornerValNameDict.iteritems())
