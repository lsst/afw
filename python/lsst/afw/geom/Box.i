/* 
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
 * 
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the LSST License Statement and 
 * the GNU General Public License along with this program.  If not, 
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */
 

%{
#include "lsst/afw/geom/Box.h"
%}

%ignore lsst::afw::geom::Box2I::getSlices;
%rename(set) lsst::afw::geom::Box2I::operator=;
%useValueEquality(lsst::afw::geom::Box2I);
%copyctor lsst::afw::geom::Box2I;
%rename(set) lsst::afw::geom::Box2D::operator=;
%useValueEquality(lsst::afw::geom::Box2D);
%copyctor lsst::afw::geom::Box2D;

%include "lsst/afw/geom/Box.h"

%extend lsst::afw::geom::Box2I {
    %pythoncode %{

    Extent = Extent2I
    Point = Point2I

    __swig_getmethods__["min"] = getMin
    __swig_getmethods__["max"] = getMax
    __swig_getmethods__["begin"] = getBegin
    __swig_getmethods__["end"] = getEnd
    __swig_getmethods__["dimensions"] = getDimensions
    __swig_getmethods__["width"] = getWidth
    __swig_getmethods__["height"] = getHeight

    def __repr__(self):
        return "Box2I(%r, %r)" % (self.getMin(), self.getDimensions())

    def __reduce__(self):
        return (Box2I, (self.getMin(), self.getMax()))

    def __str__(self):
        return "Box2I(%s, %s)" % (self.getMin(), self.getMax())

    def getSlices(self):
        return (slice(self.getBeginY(), self.getEndY()), slice(self.getBeginX(), self.getEndX()))

    def getCorners(self):
        return (
            self.getMin(),
            self.Point(self.getMaxX(), self.getMinY()),
            self.getMax(),
            self.Point(self.getMinX(), self.getMaxY())
        )

    %}
}

%extend lsst::afw::geom::Box2D {
    %pythoncode %{

    Extent = Extent2D
    Point = Point2D

    __swig_getmethods__["min"] = getMin
    __swig_getmethods__["max"] = getMax
    __swig_getmethods__["dimensions"] = getDimensions
    __swig_getmethods__["width"] = getWidth
    __swig_getmethods__["height"] = getHeight


    def __repr__(self):
        return "Box2D(%r, %r)" % (self.getMin(), self.getDimensions())

    def __reduce__(self):
        return (Box2D, (self.getMin(), self.getDimensions()))

    def __str__(self):
        return "Box2D(%s, %s)" % (self.getMin(), self.getMax())

    def getCorners(self):
        return (
            self.getMin(),
            self.Point(self.getMaxX(), self.getMinY()),
            self.getMax(),
            self.Point(self.getMinX(), self.getMaxY())
        )

    %}
}
