%{
#include "lsst/afw/math/warpExposure.h"
%}

//
// Additional kernel subclasses
//
// These definitions must go before you %include the .h file; the %templates must go after
//
SWIG_SHARED_PTR_DERIVED(LanczosWarpingKernel, lsst::afw::math::SeparableKernel, lsst::afw::math::LanczosWarpingKernel);
SWIG_SHARED_PTR_DERIVED(BilinearWarpingKernel, lsst::afw::math::SeparableKernel, lsst::afw::math::BilinearWarpingKernel);

%include "lsst/afw/math/warpExposure.h"

%define %warpExp(DESTIMAGEPIXELT, SRCIMAGEPIXELT)
%template(warpExposure) lsst::afw::math::warpExposure<
    lsst::afw::image::Exposure<DESTIMAGEPIXELT, lsst::afw::image::MaskPixel, lsst::afw::image::VariancePixel>,
    lsst::afw::image::Exposure<SRCIMAGEPIXELT, lsst::afw::image::MaskPixel, lsst::afw::image::VariancePixel> >;
%enddef

%warpExp(float, int);
%warpExp(double, int);
%warpExp(float, boost::uint16_t);
%warpExp(double, boost::uint16_t);
%warpExp(float, float);
%warpExp(double, float);
%warpExp(double, double);
