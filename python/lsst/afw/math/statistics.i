
%{
#include "lsst/afw/math/Statistics.h"
%}

%include "lsst/afw/math/Statistics.h"

%define %declareStats(PIXTYPE, SUFFIX)
    %template(makeStatistics) lsst::afw::math::makeStatistics<lsst::afw::image::Image<PIXTYPE> >;
    %template(Statistics ## SUFFIX) lsst::afw::math::Statistics::Statistics<lsst::afw::image::Image<PIXTYPE> >;
%enddef

%declareStats(unsigned short, U)
%declareStats(double, D)
%declareStats(float, F)
%declareStats(int, I)
