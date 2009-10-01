
%{
#include "lsst/afw/math/Interpolate.h"
%}

%include "lsst/afw/math/Interpolate.h"

%define %declareInterp(TYPE1, TYPE2, SUFFIX)
	%template(Interpolate ## SUFFIX) lsst::afw::math::Interpolate<TYPE1,TYPE2>;
	%template(LinearInterpolate ## SUFFIX) lsst::afw::math::LinearInterpolate<TYPE1,TYPE2>;
	%template(SplineInterpolate ## SUFFIX) lsst::afw::math::SplineInterpolate<TYPE1,TYPE2>;
%enddef

%declareInterp(double, double, DD);
%declareInterp(float, float, FF);
%declareInterp(int, double, ID);
%declareInterp(int, float, IF);
%declareInterp(int, int, II);

