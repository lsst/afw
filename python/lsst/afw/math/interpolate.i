
%{
#include "lsst/afw/math/Interpolate.h"
%}

%include "lsst/afw/math/Interpolate.h"

%template(InterpolateDD) lsst::afw::math::Interpolate<double,double>;
%template(InterpolateFF) lsst::afw::math::Interpolate<float,float>;
%template(InterpolateID) lsst::afw::math::Interpolate<int,double>;
%template(InterpolateIF) lsst::afw::math::Interpolate<int,float>;
%template(InterpolateII) lsst::afw::math::Interpolate<int,int>;

%template(LinearInterpolateDD) lsst::afw::math::LinearInterpolate<double,double>;
%template(LinearInterpolateFF) lsst::afw::math::LinearInterpolate<float,float>;
%template(LinearInterpolateID) lsst::afw::math::LinearInterpolate<int,double>;
%template(LinearInterpolateIF) lsst::afw::math::LinearInterpolate<int,float>;
%template(LinearInterpolateII) lsst::afw::math::LinearInterpolate<int,int>;

%template(SplineInterpolateDD) lsst::afw::math::SplineInterpolate<double,double>;
%template(SplineInterpolateFF) lsst::afw::math::SplineInterpolate<float,float>;
%template(SplineInterpolateID) lsst::afw::math::SplineInterpolate<int,double>;
%template(SplineInterpolateIF) lsst::afw::math::SplineInterpolate<int,float>;
%template(SplineInterpolateII) lsst::afw::math::SplineInterpolate<int,int>;

