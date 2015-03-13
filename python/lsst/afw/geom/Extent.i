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

// This doesn't wrap "scalar OP extent" versions, but that's okay.
%define %Extent_PREINCLUDE(T,N)
%copyctor lsst::afw::geom::Extent<T,N>;
%useValueEquality(lsst::afw::geom::Extent<T,N>)
%rename(__add__) lsst::afw::geom::ExtentBase<T,N>::operator+;
%rename(__sub__) lsst::afw::geom::ExtentBase<T,N>::operator-;
%rename(__iadd__) lsst::afw::geom::ExtentBase<T,N>::operator+=;
%rename(__isub__) lsst::afw::geom::ExtentBase<T,N>::operator-=;
%rename(__pos__) lsst::afw::geom::ExtentBase<T,N>::operator+() const;
%rename(__neg__) lsst::afw::geom::ExtentBase<T,N>::operator-() const;
%rename(__mul__) lsst::afw::geom::ExtentBase<T,N>::operator*;
%rename(__imul__) lsst::afw::geom::ExtentBase<T,N>::operator*=;
%rename(__div__) lsst::afw::geom::ExtentBase<T,N>::operator/;
%rename(__idiv__) lsst::afw::geom::ExtentBase<T,N>::operator/=;
%enddef

%define %Extent_POSTINCLUDE(T,N,SUFFIX)
%template(ExtentCoordinateBase ## N ## SUFFIX) lsst::afw::geom::CoordinateBase<lsst::afw::geom::Extent<T,N>,T,N>;
%template(ExtentBase ## N ## SUFFIX) lsst::afw::geom::ExtentBase<T,N>;
%template(Extent ## N ## SUFFIX) lsst::afw::geom::Extent<T,N>;
%CoordinateBase_POSTINCLUDE(T, N, lsst::afw::geom::Extent<T,N>);
%extend lsst::afw::geom::ExtentBase<T,N> {
    %pythoncode %{
# support "__from__ future import division" in Python 2; not needed for Python 3
# warning: this indentation level is required for SWIG 3.0.2, at least
__truediv__ = __div__
__itruediv__ = __idiv__
    %}
}
%enddef
