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
%rename(__eq__) lsst::afw::geom::Extent<T,N>::operator==;
%rename(__ne__) lsst::afw::geom::Extent<T,N>::operator!=;
%rename(__add__) lsst::afw::geom::Extent<T,N>::operator+;
%rename(__sub__) lsst::afw::geom::Extent<T,N>::operator-;
%rename(__iadd__) lsst::afw::geom::Extent<T,N>::operator+=;
%rename(__isub__) lsst::afw::geom::Extent<T,N>::operator-=;
%rename(__pos__) lsst::afw::geom::Extent<T,N>::operator+() const;
%rename(__neg__) lsst::afw::geom::Extent<T,N>::operator-() const;
%rename(__mul__) lsst::afw::geom::Extent<T,N>::operator*;
%rename(__imul__) lsst::afw::geom::Extent<T,N>::operator*=;
%rename(__div__) lsst::afw::geom::Extent<T,N>::operator/;
%rename(__idiv__) lsst::afw::geom::Extent<T,N>::operator/=;
%enddef

%CoordinateBase_PREINCLUDE_2(int, lsst::afw::geom::Extent<int,2>);
%CoordinateBase_PREINCLUDE_3(int, lsst::afw::geom::Extent<int,3>);
%Extent_PREINCLUDE(int,2);
%Extent_PREINCLUDE(int,3);

%CoordinateBase_PREINCLUDE_2(double, lsst::afw::geom::Extent<double,2>);
%CoordinateBase_PREINCLUDE_3(double, lsst::afw::geom::Extent<double,3>);
%Extent_PREINCLUDE(double,2);
%Extent_PREINCLUDE(double,3);
