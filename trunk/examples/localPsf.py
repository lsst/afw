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

import lsst.afw.detection as detection
import lsst.afw.geom as geom
import lsst.afw.geom.ellipses
import lsst.afw.image as image
import lsst.afw.math.shapelets as shapelets
import lsst.afw.display.ds9 as ds9

import numpy


def run():
    order = 1
    sigma = 3.0
    size = shapelets.computeSize(order)
    coeff = [10, 2, 3]
    dim = geom.ExtentI(15)
    bbox = geom.BoxI(geom.PointI(0,0), dim)
    center = geom.PointD(7,7)

    sf = shapelets.ShapeletFunction(order, shapelets.LAGUERRE, sigma, center, coeff)
    msf = shapelets.MultiShapeletFunction(sf)
    shapeletLocalPsf = detection.ShapeletLocalPsf(center, msf)

    fp = detection.Footprint(bbox)
        
    flatarray = numpy.zeros(bbox.getArea())
    shapeletLocalPsf.evaluatePointSource(fp, flatarray)
    norm = numpy.sum(flatarray)
    flatarray /= norm
    flatarray.resize((dim.getY(), dim.getX()))
    original = image.ImageD(flatarray)


    imageLocalPsf = detection.ImageLocalPsf(center, original)

    sfFromImage = imageLocalPsf.computeShapelet(shapelets.LAGUERRE, 10, shapeletLocalPsf.computeMoments())
    msfFromImage = shapelets.MultiShapeletFunction(sfFromImage)
    roundtripShapeletLocalPsf = detection.ShapeletLocalPsf(center, msfFromImage)


    flatarray = numpy.zeros(bbox.getArea())
    roundtripShapeletLocalPsf.evaluatePointSource(fp, flatarray)
    norm = numpy.sum(flatarray)
    flatarray /= norm
    flatarray.resize((dim.getY(), dim.getX()))
    roundtrip = image.ImageD(flatarray)

    ds9.mtv(original, frame=0, title="original shapeletImage")
    e = shapeletLocalPsf.computeMoments()
    m = geom.ellipses.Quadrupole(e.getCore())
    p = e.getCenter()
    ds9.dot("@:%f,%f,%f"%(m.getIXX(), m.getIXY(), m.getIYY()), p.getX(), p.getY(), frame=0)
    ds9.mtv(roundtrip, frame=1, title="roundtripped shapeletImage")
    e = roundtripShapeletLocalPsf.computeMoments()
    m = geom.ellipses.Quadrupole(e.getCore())
    p = e.getCenter()
    ds9.dot("@:%f,%f,%f"%(m.getIXX(), m.getIXY(), m.getIYY()), p.getX(), p.getY(),frame=1)

if __name__ == "__main__":
    run()






