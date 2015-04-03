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

%define %Extent_PREINCLUDE(T,N)
%copyctor lsst::afw::geom::Extent<T,N>;
%useValueEquality(lsst::afw::geom::Extent<T,N>)
%ignore lsst::afw::geom::ExtentBase<T,N>::operator+;
%ignore lsst::afw::geom::ExtentBase<T,N>::operator-;
%ignore lsst::afw::geom::ExtentBase<T,N>::operator+=;
%ignore lsst::afw::geom::ExtentBase<T,N>::operator-=;
%rename(__pos__) lsst::afw::geom::ExtentBase<T,N>::operator+() const;
%rename(__neg__) lsst::afw::geom::ExtentBase<T,N>::operator-() const;
%ignore lsst::afw::geom::ExtentBase<T,N>::operator*;
%ignore lsst::afw::geom::ExtentBase<T,N>::operator*=;
%ignore lsst::afw::geom::ExtentBase<T,N>::operator/;
%ignore lsst::afw::geom::ExtentBase<T,N>::operator/=;
%enddef

%define %Extent_POSTINCLUDE(T,N,SUFFIX)
%template(ExtentCoordinateBase ## N ## SUFFIX) lsst::afw::geom::CoordinateBase<lsst::afw::geom::Extent<T,N>,T,N>;
%template(ExtentBase ## N ## SUFFIX) lsst::afw::geom::ExtentBase<T,N>;
%template(Extent ## N ## SUFFIX) lsst::afw::geom::Extent<T,N>;
%CoordinateBase_POSTINCLUDE(T, N, lsst::afw::geom::Extent<T,N>);
%extend lsst::afw::geom::Extent<T,N> {
    lsst::afw::geom::Point<T,N> __add__(lsst::afw::geom::Point<T,N> const & other) const {
        return *self + other;
    }
    lsst::afw::geom::Extent<T,N> __add__(lsst::afw::geom::Extent<T,N> const & other) const {
        return *self + other;
    }
    lsst::afw::geom::Extent<T,N> __sub__(lsst::afw::geom::Extent<T,N> const & other) const {
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
    lsst::afw::geom::Extent<T,N> __mul__(T scalar) const {
        return *self * scalar;
    }
    PyObject * __imul__(PyObject** PYTHON_SELF, T scalar) {
        *self *= scalar;
        Py_INCREF(*PYTHON_SELF);
        return *PYTHON_SELF;
    }
    lsst::afw::geom::Extent<T,N> __rmul__(T scalar) const {
        return scalar * (*self);
    }
}
%enddef

%define %ExtentD_POSTINCLUDE(N)
%Extent_POSTINCLUDE(double, N, D)
%extend lsst::afw::geom::Extent<double,N> {
    lsst::afw::geom::Extent<int,N> truncate() const { return truncate(*self); }
    lsst::afw::geom::Extent<int,N> floor() const { return floor(*self); }
    lsst::afw::geom::Extent<int,N> ceil() const { return ceil(*self); }
    lsst::afw::geom::Extent<double,N> __add__(lsst::afw::geom::Extent<int,N> const & other) const {
        return *self + other;
    }
    lsst::afw::geom::Point<double,N> __add__(lsst::afw::geom::Point<int,N> const & other) const {
        return other + *self;
    }
    PyObject * __iadd__(PyObject** PYTHON_SELF, lsst::afw::geom::Extent<int,N> const & other) {
        *self += other;
        Py_INCREF(*PYTHON_SELF);
        return *PYTHON_SELF;
    }
    lsst::afw::geom::Extent<double,N> __sub__(lsst::afw::geom::Extent<int,N> const & other) const {
        return *self - other;
    }
    PyObject * __isub__(PyObject** PYTHON_SELF, lsst::afw::geom::Extent<int,N> const & other) {
        *self -= other;
        Py_INCREF(*PYTHON_SELF);
        return *PYTHON_SELF;
    }
    // no need for overloads that take int scalars, because int scalars will automatically match the
    // overloads that take double
    lsst::afw::geom::Extent<double,N> __truediv__(double scalar) const {
        return *self / scalar;
    }
    PyObject * __itruediv__(PyObject** PYTHON_SELF, double scalar) {
        *self /= scalar;
        Py_INCREF(*PYTHON_SELF);
        return *PYTHON_SELF;
    }
    %pythoncode %{
# support code not using "__from__ future import division"
# warning: this indentation level is required for SWIG 3.0.2, at least
__div__ = __truediv__
__idiv__ = __itruediv__
    %}
}
%template(truncate) lsst::afw::geom::truncate<N>;
%template(floor) lsst::afw::geom::floor<N>;
%template(ceil) lsst::afw::geom::ceil<N>;
%enddef

%define %ExtentI_POSTINCLUDE(N)
%Extent_POSTINCLUDE(int, N, I)
%extend lsst::afw::geom::Extent<int,N> {
    lsst::afw::geom::Extent<double,N> __add__(lsst::afw::geom::Extent<double,N> const & other) const {
        return *self + other;
    }
    lsst::afw::geom::Extent<double,N> __sub__(lsst::afw::geom::Extent<double,N> const & other) const {
        return *self - other;
    }
    lsst::afw::geom::Point<double,N> __add__(lsst::afw::geom::Point<double,N> const & other) const {
        return *self + other;
    }
    lsst::afw::geom::Extent<double,N> __mul__(double scalar) const {
        return *self * scalar;
    }
    lsst::afw::geom::Extent<double,N> __rmul__(double scalar) const {
        return scalar * (*self);
    }
    // __floordiv__ (invoked by //) always takes ints, returns ExtentI.
    lsst::afw::geom::Extent<int,N> __floordiv__(int scalar) const {
        // Python's integer division works differently that C++'s for negative numbers - Python
        // uses floor (rounds towards more negative), while C++ truncates (rounds towards zero).
        return floor(lsst::afw::geom::Extent<double,N>(*self) / scalar);
    }
    PyObject * __ifloordiv__(PyObject** PYTHON_SELF, int scalar) {
        *self = floor(lsst::afw::geom::Extent<double,N>(*self) / scalar);
        Py_INCREF(*PYTHON_SELF);
        return *PYTHON_SELF;
    }
    // __truediv__ (invoked by / with future division) always takes doubles, returns ExtentD
    lsst::afw::geom::Extent<double,N> __truediv__(double scalar) const {
        return *self / scalar;
    }
    // Don't want __itruediv__; would convert type in-place - but we have to add one that throws,
    // because otherwise Python uses __div__, which is incorrect.
    PyObject * __itruediv__(double scalar) const {
        PyErr_SetString(PyExc_TypeError, "In-place true division not supported for Extent<int,N>.");
        return NULL;
    }
    // __div__ (invoked by / without future division) mimics __floordiv__ for ints, __truediv__ for floats
    lsst::afw::geom::Extent<int,N> __div__(int scalar) const {
        return floor(lsst::afw::geom::Extent<double,N>(*self) / scalar);
    }
    PyObject * __idiv__(PyObject** PYTHON_SELF, int scalar) {
        *self = floor(lsst::afw::geom::Extent<double,N>(*self) / scalar);
        Py_INCREF(*PYTHON_SELF);
        return *PYTHON_SELF;
    }
    lsst::afw::geom::Extent<double,N> __div__(double scalar) const {
        return *self / scalar;
    }
    // No __idiv__ taking doubles; would convert type in-place.
}
%enddef
