#
# LSST Data Management System
# See the COPYRIGHT and LICENSE files in the top-level directory of this
# package for notices and licensing terms.
#

import detectionLib as afwDetect

def writeFootprintAsDefects(fd, foot):
    """Write foot as a set of Defects to fd"""

    bboxes = afwDetect.footprintToBBoxList(foot)
    for bbox in bboxes:
        print >> fd, """\
Defects: {
    x0:     %4d                         # Starting column
    width:  %4d                         # number of columns
    y0:     %4d                         # Starting row
    height: %4d                         # number of rows
}""" % (bbox.getMinX(), bbox.getWidth(), bbox.getMinY(), bbox.getHeight())
