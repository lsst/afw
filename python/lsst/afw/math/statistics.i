
%{
#include "lsst/afw/math/Statistics.h"
%}

%include "lsst/afw/math/Statistics.h"

%define %stats(NAME, PIXEL_TYPE)
%template(NAME) lsst::afw::math::Statistics<lsst::afw::image::Image<PIXEL_TYPE> >;
%template(make_Statistics) lsst::afw::math::make_Statistics<lsst::afw::image::Image<PIXEL_TYPE> >;
%enddef

%stats(StatisticsD, double);
%stats(StatisticsF, float);
%stats(StatisticsI, int);
