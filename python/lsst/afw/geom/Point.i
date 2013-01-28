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

%module(package="lsst.afw.geom") geomLib
%include "CoordinateBase.i"

%{
#include "lsst/afw/geom/Extent.h"
#include "lsst/afw/geom/Point.h"
%}

// This doesn't wrap "scalar OP extent" versions, but that's okay.
%define %Point_PREINCLUDE(T,N)
%copyctor lsst::afw::geom::Point<T,N>;
%useValueEquality(lsst::afw::geom::Point<T,N>)
%rename(__add__) lsst::afw::geom::Point<T,N>::operator+;
%rename(__sub__) lsst::afw::geom::Point<T,N>::operator-;
%rename(__iadd__) lsst::afw::geom::Point<T,N>::operator+=;
%rename(__isub__) lsst::afw::geom::Point<T,N>::operator-=;
%enddef

%define %Point_POSTINCLUDE(T,N,SUFFIX)
%template(PointCoordinateBase ## N ## SUFFIX) lsst::afw::geom::CoordinateBase<lsst::afw::geom::Point<T,N>,T,N>;
%template(PointBase ## N ## SUFFIX) lsst::afw::geom::PointBase<T,N>;
%template(Point ## N ## SUFFIX) lsst::afw::geom::Point<T,N>;
%CoordinateBase_POSTINCLUDE(T, N, lsst::afw::geom::Point<T,N>);
%enddef

%Point_PREINCLUDE(int,2);
%Point_PREINCLUDE(int,3);
%Point_PREINCLUDE(double,2);
%Point_PREINCLUDE(double,3);

%include "lsst/afw/geom/Point.h"

%Point_POSTINCLUDE(int,2,I);
%Point_POSTINCLUDE(int,3,I);
%Point_POSTINCLUDE(double,2,D);
%Point_POSTINCLUDE(double,3,D);


%extend lsst::afw::geom::Point<int,2> {
    %template(Point2I) Point<double>;
    %pythoncode {
    def __reduce__(self):
        return (Point2I, (self.getX(), self.getY()))
    }
};

%extend lsst::afw::geom::Point<int,3> {
    %template(Point3I) Point<double>;
    %pythoncode {
    def __reduce__(self):
        return (Point3I, (self.getX(), self.getY(), self.getZ()))
    }
};

%extend lsst::afw::geom::Point<double,2> {
    %template(Point2D) Point<int>;
    %pythoncode {
    def __reduce__(self):
        return (Point2D, (self.getX(), self.getY()))
    }
};

%extend lsst::afw::geom::Point<double,3> {
    %template(Point3D) Point<int>;
    %pythoncode {
    def __reduce__(self):
        return (Point3D, (self.getX(), self.getY(), self.getZ()))
    }
};
