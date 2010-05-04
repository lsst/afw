
%{
#include "lsst/afw/geom/Extent.h"
#include "lsst/afw/geom/Point.h"
%}

// This doesn't wrap "scalar OP extent" versions, but that's okay.
%define %Extent_PREINCLUDE(T,N)
%rename(__eq__) lsst::afw::geom::ExtentBase<T,N>::operator==;
%rename(__ne__) lsst::afw::geom::ExtentBase<T,N>::operator!=;
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

%Extent_PREINCLUDE(int,2);
%Extent_PREINCLUDE(int,3);

%Extent_PREINCLUDE(double,2);
%Extent_PREINCLUDE(double,3);

%define %Extent_POSTINCLUDE(T,N,SUFFIX)
%template(ExtentCoordinateBase ## N ## SUFFIX) lsst::afw::geom::CoordinateBase<lsst::afw::geom::Extent<T,N>,T,N>;
%template(ExtentBase ## N ## SUFFIX) lsst::afw::geom::ExtentBase<T,N>;
%template(Extent ## N ## SUFFIX) lsst::afw::geom::Extent<T,N>;
%CoordinateBase_POSTINCLUDE(T, N, lsst::afw::geom::Extent<T,N>);
%enddef
