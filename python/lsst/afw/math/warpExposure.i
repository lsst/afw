// -*- lsst-c++ -*-
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

%define %WarpFuncsByType(DESTIMAGEPIXELT, SRCIMAGEPIXELT)
%template(warpExposure) lsst::afw::math::warpExposure<
    lsst::afw::image::Exposure<DESTIMAGEPIXELT, lsst::afw::image::MaskPixel, lsst::afw::image::VariancePixel>,
    lsst::afw::image::Exposure<SRCIMAGEPIXELT, lsst::afw::image::MaskPixel, lsst::afw::image::VariancePixel> >;
%template(warpImage) lsst::afw::math::warpImage<
    lsst::afw::image::MaskedImage<DESTIMAGEPIXELT, lsst::afw::image::MaskPixel, lsst::afw::image::VariancePixel>,
    lsst::afw::image::MaskedImage<SRCIMAGEPIXELT, lsst::afw::image::MaskPixel, lsst::afw::image::VariancePixel> >;
%template(warpImage) lsst::afw::math::warpImage<
    lsst::afw::image::Image<DESTIMAGEPIXELT>,
    lsst::afw::image::Image<SRCIMAGEPIXELT> >;
%enddef

%WarpFuncsByType(float, boost::uint16_t);
%WarpFuncsByType(double, boost::uint16_t);
%WarpFuncsByType(float, int);
%WarpFuncsByType(double, int);
%WarpFuncsByType(float, float);
%WarpFuncsByType(double, float);
%WarpFuncsByType(double, double);

//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

%{
#include "lsst/afw/math/offsetImage.h"
%}

%include "lsst/afw/math/offsetImage.h"

%template(offsetImage) lsst::afw::math::offsetImage<lsst::afw::image::Image<double> >;
%template(offsetImage) lsst::afw::math::offsetImage<lsst::afw::image::Image<float> >;
