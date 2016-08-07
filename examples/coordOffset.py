#!/usr/bin/env python

#
# LSST Data Management System
# Copyright 2008, 2009, 2010 LSST Corporation.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#

#
from __future__ import print_function
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
        print(c.getLongitude(afwCoord.DEGREES), c.getLatitude(afwCoord.DEGREES), end=' ')

        # try the rotate() method
        c = afwCoord.Fk5Coord(long0, lat0)
        ax = afwCoord.Fk5Coord(axLon, axLat)

        c.rotate(ax, arc)
        print(c.getLongitude(afwCoord.DEGREES), c.getLatitude(afwCoord.DEGREES))

if __name__ == '__main__':
    main()
