
%{
#include "lsst/afw/geom/Extent.h"
#include "lsst/afw/geom/Point.h"
%}

// This doesn't wrap "scalar OP extent" versions, but that's okay.
%define %Extent_PREINCLUDE(T,N)
%rename(__eq__) lsst::afw::geom::Extent<T,N>::operator==;
%rename(__ne__) lsst::afw::geom::Extent<T,N>::operator!=;
%rename(__lt__) lsst::afw::geom::Extent<T,N>::operator<;
%rename(__le__) lsst::afw::geom::Extent<T,N>::operator<=;
%rename(__gt__) lsst::afw::geom::Extent<T,N>::operator>;
%rename(__ge__) lsst::afw::geom::Extent<T,N>::operator>=;
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
