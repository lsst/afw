
%{
#include "lsst/afw/geom/CoordinateExpr.h"
%}

// These wrap && -> &, because Python 'and' cannot be overloaded (etc)
%define %CoordinateExpr_PREINCLUDE(N)
%rename(__and__) lsst::afw::geom::CoordinateExpr<N>::operator&&;
%rename(__or__) lsst::afw::geom::CoordinateExpr<N>::operator||;
%rename(__not__) lsst::afw::geom::CoordinateExpr<N>::operator!;
%enddef

%CoordinateBase_PREINCLUDE_2(bool, lsst::afw::geom::CoordinateExpr<2>);
%CoordinateBase_PREINCLUDE_3(bool, lsst::afw::geom::CoordinateExpr<3>);
%CoordinateExpr_PREINCLUDE(2);
%CoordinateExpr_PREINCLUDE(3);

// Note: any() and all() not wrapped; use numpy.any() and numpy.all().
