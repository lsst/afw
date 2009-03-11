import lsst.afw.detection.detectionLib as afwDetection

def writeFootprintAsDefects(fd, foot):
    """Write foot as a set of Defects to fd"""

    bboxes = afwDetection.footprintToBBoxList(foot)
    for bbox in bboxes:
        x0, y0, x1, y1 = bbox.getX0(), bbox.getY0(), bbox.getX1(), bbox.getY1()

        print >> fd, """\
Defects: {
    x0:     %4d                         # Starting column
    width:  %4d                         # number of columns
    y0:     %4d                         # Starting row
    height: %4d                         # number of rows
}""" % (bbox.getX0(), bbox.getWidth(), bbox.getY0(), bbox.getHeight())

