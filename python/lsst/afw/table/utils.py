#
# LSST Data Management System
# Copyright 2016 AURA/LSST.
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
# see <https://www.lsstcorp.org/LegalNotices/>.
#
from __future__ import absolute_import, division, print_function

from .tableLib import CoordKey, Point2DKey

__all__ = ["updateRefCentroids", "updateSourceCoords"]


def updateRefCentroids(wcs, refList):
    """Update centroids in a collection of reference objects

    This code supports any kind of collection, instead of requiring a catalog,
    to make it easy to use with match lists. For example:

        updateRefCentroids(wcs, refList=[match.first for match in matches])

    @param[in] wcs  WCS to map from sky to pixels; an lsst.afw.image.Wcs
    @param[in,out] refList  collection of reference objects (lsst.afw.table.SimpleRecords); for each:
                            - read field "coords", an lsst.afw.coord.Coord
                            - write field "centroid", an lsst.afw.geom.Point2D
    """
    if len(refList) < 1:
        return
    schema = refList[0].schema
    coordKey = CoordKey(schema["coord"])
    centroidKey = Point2DKey(schema["centroid"])
    for refObj in refList:
        refObj.set(centroidKey, wcs.skyToPixel(refObj.get(coordKey)))


def updateSourceCoords(wcs, sourceList):
    """Update coords in a collection of sources

    This code supports any kind of collection, instead of requiring a catalog,
    to make it easy to use with match lists. For example:

        updateSourceCoords(wcs, sourceList=[match.second for match in matches])

    @param[in] wcs  WCS to map from pixels to sky; an lsst.afw.image.Wcs
    @param[in,out] sourceList   collection of sources (lsst.afw.table.SourceRecords); for each:
                                - read centroid using getCentroid()
                                - write field "coord", an lsst.afw.coord.Coord
    """
    if len(sourceList) < 1:
        return
    schema = sourceList[0].schema
    srcCoordKey = CoordKey(schema["coord"])
    for src in sourceList:
        src.set(srcCoordKey, wcs.pixelToSky(src.getCentroid()))
