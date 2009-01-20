
%{
#include "lsst/afw/math/Statistics.h"
%}

%include "lsst/afw/math/Statistics.h"

%define %declareStats(PIXTYPE, SUFFIX)
    %template(make_Statistics) lsst::afw::math::make_Statistics<lsst::afw::image::Image<PIXTYPE> >;
    %template(Statistics ## SUFFIX) lsst::afw::math::Statistics::Statistics<lsst::afw::image::Image<PIXTYPE> >;
%enddef

%declareStats(double, D)
%declareStats(float, F)
%declareStats(int, I)
