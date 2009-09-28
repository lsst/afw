
%{
#include "lsst/afw/math/Statistics.h"
%}

%include "lsst/afw/math/Statistics.h"

%define %declareStats(PIXTYPE, SUFFIX)
    %template(makeStatistics ## SUFFIX) lsst::afw::math::makeStatistics<lsst::afw::image::Image<PIXTYPE> >;
    // %template(makeStatistics ## SUFFIX) lsst::afw::math::makeStatistics<lsst::afw::image::MaskedImage<PIXTYPE> >;
    // %template(makeStatistics ## SUFFIX) lsst::afw::math::makeStatistics<std::vector<PIXTYPE> >;
    %template(Statistics ## SUFFIX) lsst::afw::math::Statistics::Statistics<lsst::afw::image::Image<PIXTYPE>, lsst::afw::image::Mask<lsst::afw::image::MaskPixel> >;
    // %template(Statistics ## SUFFIX) lsst::afw::math::Statistics::Statistics<lsst::afw::image::Image<PIXTYPE>, lsst::afw::math::MaskImposter<lsst::afw::image::MaskPixel> >;
    // %template(Statistics ## SUFFIX) lsst::afw::math::Statistics::Statistics<lsst::afw::math::ImageImposter<PIXTYPE>, lsst::afw::math::MaskImposter<lsst::afw::image::MaskPixel> >;
    
%enddef

%declareStats(unsigned short, U)
%declareStats(double, D)
%declareStats(float, F)
%declareStats(int, I)

// We also support Mask<MaskPixel>
%template(makeStatisticsMU) lsst::afw::math::makeStatistics<lsst::afw::image::Mask<lsst::afw::image::MaskPixel> >;
%template(StatisticsMU) lsst::afw::math::Statistics::Statistics<lsst::afw::image::Mask<lsst::afw::image::MaskPixel>, lsst::afw::image::Mask<lsst::afw::image::MaskPixel> >;

