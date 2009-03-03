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

%define %warpExposureFuncByType(DESTIMAGEPIXELT, SRCIMAGEPIXELT)
%template(warpExposure) lsst::afw::math::warpExposure<
    lsst::afw::image::Exposure<DESTIMAGEPIXELT>,
    lsst::afw::image::Exposure<SRCIMAGEPIXELT> >;
%enddef

%warpExposureFuncByType(float, boost::uint16_t);
%warpExposureFuncByType(double, boost::uint16_t);
%warpExposureFuncByType(float, int);
%warpExposureFuncByType(double, int);
%warpExposureFuncByType(float, float);
%warpExposureFuncByType(double, float);
%warpExposureFuncByType(double, double);

//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

%{
#include "lsst/afw/math/offsetImage.h"
%}

%include "lsst/afw/math/offsetImage.h"

%template(offsetImage) lsst::afw::math::offsetImage<lsst::afw::image::Image<double> >;
%template(offsetImage) lsst::afw::math::offsetImage<lsst::afw::image::Image<float> >;
