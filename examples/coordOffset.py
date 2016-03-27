#!/usr/bin/env python

#
# LSST Data Management System
# See the COPYRIGHT and LICENSE files in the top-level directory of this
# package for notices and licensing terms.
#

#
import lsst.afw.coord as afwCoord

def main():

    # trace out a great circle
    long0 = 44.0
    lat0 = 0.0
    dArc = 1.0
    phi = 45.0

    # axis
    axLon = 90.0
    axLat = 0.0
    
    n = 360.0/dArc

    for i in range(n):

        # try the offset() method
        arc = i*dArc*afwCoord.degToRad
        c = afwCoord.Fk5Coord(long0, lat0)
        c.offset(phi*afwCoord.degToRad, arc)
        print c.getLongitude(afwCoord.DEGREES), c.getLatitude(afwCoord.DEGREES),

        # try the rotate() method
        c = afwCoord.Fk5Coord(long0, lat0)
        ax = afwCoord.Fk5Coord(axLon, axLat)

        c.rotate(ax, arc)
        print c.getLongitude(afwCoord.DEGREES), c.getLatitude(afwCoord.DEGREES)

if __name__ == '__main__':
    main()
