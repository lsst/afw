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
#include "lsst/afw/geom/Extent.h"
#include "lsst/afw/geom/Point.h"
%}

%define %Point_PREINCLUDE(T,N)
%copyctor lsst::afw::geom::Point<T,N>;
%useValueEquality(lsst::afw::geom::Point<T,N>)
%ignore lsst::afw::geom::Point<T,N>::operator+;
%ignore lsst::afw::geom::Point<T,N>::operator-;
%ignore lsst::afw::geom::Point<T,N>::operator+=;
%ignore lsst::afw::geom::Point<T,N>::operator-=;
%enddef

%define %Point_POSTINCLUDE(T,N,SUFFIX)
%template(PointCoordinateBase ## N ## SUFFIX) lsst::afw::geom::CoordinateBase<lsst::afw::geom::Point<T,N>,T,N>;
%template(PointBase ## N ## SUFFIX) lsst::afw::geom::PointBase<T,N>;
%template(Point ## N ## SUFFIX) lsst::afw::geom::Point<T,N>;
%CoordinateBase_POSTINCLUDE(T, N, lsst::afw::geom::Point<T,N>);

%extend lsst::afw::geom::Point<T,N> {

    lsst::afw::geom::Extent<T,N> __sub__(lsst::afw::geom::Point<T,N> const & other) const {
        return *self - other;
    }

    lsst::afw::geom::Point<T,N> __add__(lsst::afw::geom::Extent<T,N> const & other) const {
        return *self + other;
    }

    lsst::afw::geom::Point<T,N> __sub__(lsst::afw::geom::Extent<T,N> const & other) const {
        return *self - other;
    }

    PyObject * __iadd__(PyObject** PYTHON_SELF, lsst::afw::geom::Extent<T,N> const & other) {
        *self += other;
        Py_INCREF(*PYTHON_SELF);
        return *PYTHON_SELF;
    }

    PyObject * __isub__(PyObject** PYTHON_SELF, lsst::afw::geom::Extent<T,N> const & other) {
        *self -= other;
        Py_INCREF(*PYTHON_SELF);
        return *PYTHON_SELF;
    }
}
%enddef

%define %PointD_POSTINCLUDE(N)
%Point_POSTINCLUDE(double,N,D)

%extend lsst::afw::geom::Point<double,N> {

    lsst::afw::geom::Point<double,N> __add__(lsst::afw::geom::Extent<int,N> const & other) const {
        return *self + other;
    }

    PyObject * __iadd__(PyObject** PYTHON_SELF, lsst::afw::geom::Extent<int,N> const & other) {
        *self += other;
        Py_INCREF(*PYTHON_SELF);
        return *PYTHON_SELF;
    }

    lsst::afw::geom::Point<double,N> __sub__(lsst::afw::geom::Extent<int,N> const & other) const {
        return *self - other;
    }

    PyObject * __isub__(PyObject** PYTHON_SELF, lsst::afw::geom::Extent<int,N> const & other) {
        *self -= other;
        Py_INCREF(*PYTHON_SELF);
        return *PYTHON_SELF;
    }

    lsst::afw::geom::Extent<double,N> __sub__(lsst::afw::geom::Point<int,N> const & other) const {
        return *self - other;
    }

}
%enddef

%define %PointI_POSTINCLUDE(N)
%Point_POSTINCLUDE(int,N,I)
%extend lsst::afw::geom::Point<int,N> {
    lsst::afw::geom::Point<double,N> __add__(lsst::afw::geom::Extent<double,N> const & other) const {
        return *self + other;
    }
    lsst::afw::geom::Point<double,N> __sub__(lsst::afw::geom::Extent<double,N> const & other) const {
        return *self - other;
    }
    lsst::afw::geom::Extent<double,N> __sub__(lsst::afw::geom::Point<double,N> const & other) const {
        return *self - other;
    }
}
%enddef
