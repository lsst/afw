
%{
#include "lsst/afw/math/Statistics.h"
%}

%include "lsst/afw/math/Statistics.h"


%define %declareStats(PIXTYPE, SUFFIX)
    %template(makeStatistics) lsst::afw::math::makeStatistics<PIXTYPE>;
    %template(Statistics ## SUFFIX) lsst::afw::math::Statistics::Statistics<lsst::afw::image::Image<PIXTYPE>,lsst::afw::image::Mask<lsst::afw::image::MaskPixel> >;
%enddef

%declareStats(unsigned short, U)
%declareStats(double, D)
%declareStats(float, F)
%declareStats(int, I)

// We also support Mask<MaskPixel>
%rename(makeStatisticsMU) lsst::afw::math::makeStatistics(lsst::afw::image::Mask<lsst::afw::image::MaskPixel>, int, StatisticsControl const&);
%template(StatisticsMU) lsst::afw::math::Statistics::Statistics<lsst::afw::image::Mask<lsst::afw::image::MaskPixel>, lsst::afw::image::Mask<lsst::afw::image::MaskPixel> >;

