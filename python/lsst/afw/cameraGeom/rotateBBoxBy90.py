from __future__ import absolute_import, division
# 
# LSST Data Management System
# Copyright 2014 LSST Corporation.
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
import numpy
import lsst.afw.geom as afwGeom

__all__ = ["rotateBBoxBy90"]

def rotateBBoxBy90(bbox, n90, dimensions):
    """ This is a direct translation of the method by the same name in the previous camera geom
    @param bbox -- bbox to rotate
    @param n90 -- number of quarter rotations to perform
    @param dimensions -- The dimensions of the parent grid.  If None, rotate about LLC
    @return rotated bounding box
    """
    while n90 < 0:
        n90 += 4
    n90 %= 4
    
    # sin/cos of the rotation angle
    s = 0
    c = 0                          
    if n90 == 0:
        s = 0
        c = 1
    elif n90 == 1:
        s = 1
        c = 0
    elif n90 == 2:
        s = 0
        c = -1
    elif n90 == 3:
        s = -1
        c = 0
    else:
        raise ValueError("n90 must be an integer")

    centerPixel = afwGeom.Point2I(int(dimensions[0]/2), int(dimensions[1]/2))

    xCorner = numpy.array([(corner.getX() - centerPixel[0]) for corner in bbox.getCorners()])
    yCorner = numpy.array([(corner.getY() - centerPixel[1]) for corner in bbox.getCorners()])
    x0 = min(c*xCorner - s*yCorner)
    y0 = min(s*xCorner + c*yCorner)
    x1 = max(c*xCorner - s*yCorner)
    y1 = max(s*xCorner + c*yCorner)

    # Fiddle things a little if the detector has an even number of pixels so that square BBoxes
    # will map into themselves

    if n90 == 1 :
        if  dimensions[0]%2 == 0:
            x0 -= 1 
            x1 -= 1
    elif n90 == 2:
        if dimensions[0]%2 == 0:
            x0 -= 1 
            x1 -= 1
        if dimensions[1]%2 == 0:
            y0 -= 1
            y1 -= 1
    elif n90 == 3:
        if dimensions[1]%2 == 0:
            y0 -= 1
            y1 -= 1

    LLC = afwGeom.Point2I(centerPixel[0] + x0, centerPixel[1] + y0)
    URC = afwGeom.Point2I(centerPixel[0] + x1, centerPixel[1] + y1)
 
    newBbox = afwGeom.Box2I(LLC, URC)
        
    dxy0 = centerPixel[0] - centerPixel[1]
    if n90%2 == 1 and not dxy0 == 0:
        newBbox.shift(afwGeom.Extent2I(-dxy0, dxy0))
        
    return newBbox;
