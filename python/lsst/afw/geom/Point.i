
%{
#include "lsst/afw/geom/Extent.h"
#include "lsst/afw/geom/Point.h"
%}

// This doesn't wrap "scalar OP extent" versions, but that's okay.
%define %Point_PREINCLUDE(T,N)
%rename(__eq__) lsst::afw::geom::Point<T,N>::operator==;
%rename(__ne__) lsst::afw::geom::Point<T,N>::operator!=;
%rename(__add__) lsst::afw::geom::Point<T,N>::operator+;
%rename(__sub__) lsst::afw::geom::Point<T,N>::operator-;
%rename(__iadd__) lsst::afw::geom::Point<T,N>::operator+=;
%rename(__isub__) lsst::afw::geom::Point<T,N>::operator-=;
%enddef

%CoordinateBase_PREINCLUDE_2(int, lsst::afw::geom::Point<int,2>);
%CoordinateBase_PREINCLUDE_3(int, lsst::afw::geom::Point<int,3>);
%Point_PREINCLUDE(int,2);
%Point_PREINCLUDE(int,3);

%CoordinateBase_PREINCLUDE_2(double, lsst::afw::geom::Point<double,2>);
%CoordinateBase_PREINCLUDE_3(double, lsst::afw::geom::Point<double,3>);
%Point_PREINCLUDE(double,2);
%Point_PREINCLUDE(double,3);
