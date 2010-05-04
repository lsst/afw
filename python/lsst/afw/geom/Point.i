
%{
#include "lsst/afw/geom/Extent.h"
#include "lsst/afw/geom/Point.h"
%}

// This doesn't wrap "scalar OP extent" versions, but that's okay.
%define %Point_PREINCLUDE(T,N)
%rename(__eq__) lsst::afw::geom::PointBase<T,N>::operator==;
%rename(__ne__) lsst::afw::geom::PointBase<T,N>::operator!=;
%rename(__add__) lsst::afw::geom::PointBase<T,N>::operator+;
%rename(__sub__) lsst::afw::geom::PointBase<T,N>::operator-;
%rename(__iadd__) lsst::afw::geom::PointBase<T,N>::operator+=;
%rename(__isub__) lsst::afw::geom::PointBase<T,N>::operator-=;
%enddef

%Point_PREINCLUDE(int,2);
%Point_PREINCLUDE(int,3);

%Point_PREINCLUDE(double,2);
%Point_PREINCLUDE(double,3);

%define %Point_POSTINCLUDE(T,N,SUFFIX)
%template(PointCoordinateBase ## N ## SUFFIX) lsst::afw::geom::CoordinateBase<lsst::afw::geom::Point<T,N>,T,N>;
%template(PointBase ## N ## SUFFIX) lsst::afw::geom::PointBase<T,N>;
%template(Point ## N ## SUFFIX) lsst::afw::geom::Point<T,N>;
%CoordinateBase_POSTINCLUDE(T, N, lsst::afw::geom::Point<T,N>);
%enddef

