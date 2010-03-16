// -*- lsst-c++ -*-
%{
#include "lsst/afw/math/warpExposure.h"
%}

//
// Additional kernel subclasses
//
// These definitions must go before you %include the .h file; the %templates must go after
//
SWIG_SHARED_PTR_DERIVED(BilinearWarpingKernel, lsst::afw::math::SeparableKernel,
    lsst::afw::math::BilinearWarpingKernel);
SWIG_SHARED_PTR_DERIVED(LanczosWarpingKernel, lsst::afw::math::SeparableKernel,
    lsst::afw::math::LanczosWarpingKernel);
SWIG_SHARED_PTR_DERIVED(NearestWarpingKernel, lsst::afw::math::SeparableKernel,
    lsst::afw::math::NearestWarpingKernel);

%include "lsst/afw/math/warpExposure.h"

%define %WarpFuncsByType(DESTIMAGEPIXEL, SRCIMAGEPIXEL)
%template(warpExposure) lsst::afw::math::warpExposure<
    lsst::afw::image::Exposure<DESTIMAGEPIXEL, lsst::afw::image::MaskPixel, lsst::afw::image::VariancePixel>,
    lsst::afw::image::Exposure<SRCIMAGEPIXEL, lsst::afw::image::MaskPixel, lsst::afw::image::VariancePixel> >;
%template(warpImage) lsst::afw::math::warpImage<
    lsst::afw::image::MaskedImage<DESTIMAGEPIXEL, lsst::afw::image::MaskPixel, lsst::afw::image::VariancePixel>,
    lsst::afw::image::MaskedImage<SRCIMAGEPIXEL, lsst::afw::image::MaskPixel, lsst::afw::image::VariancePixel> >;
%template(warpImage) lsst::afw::math::warpImage<
    lsst::afw::image::Image<DESTIMAGEPIXEL>,
    lsst::afw::image::Image<SRCIMAGEPIXEL> >;
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

%define rotateImageBy90(PIXELT)
%template(rotateImageBy90) lsst::afw::math::rotateImageBy90<lsst::afw::image::Image<PIXELT> >;
#if 0
%template(rotateImageBy90) lsst::afw::math::rotateImageBy90<
    lsst::afw::image::MaskedImage<PIXELT, lsst::afw::image::MaskPixel, lsst::afw::image::VariancePixel> >;
#endif
%enddef

rotateImageBy90(boost::uint16_t);
rotateImageBy90(int);
rotateImageBy90(float);
rotateImageBy90(double);
