
%{
#include "lsst/afw/math/Statistics.h"
%}

%include "lsst/afw/math/Statistics.h"

%template(StatisticsD) lsst::afw::math::Statistics<lsst::afw::image::Image<double> >;
%template(StatisticsF) lsst::afw::math::Statistics<lsst::afw::image::Image<float> >;
%template(StatisticsI) lsst::afw::math::Statistics<lsst::afw::image::Image<int> >;
