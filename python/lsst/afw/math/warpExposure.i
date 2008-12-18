
%{
#include "lsst/afw/math/warpExposure.h"
%}

%include "lsst/afw/math/warpExposure.h"

%define %warpExp(IMAGE_PIXEL_TYPE)
%template(warpExposure) lsst::afw::math::warpExposure<IMAGE_PIXEL_TYPE, lsst::afw::image::MaskPixel, lsst::afw::image::VariancePixel>;
%enddef

%warpExp(int);
%warpExp(boost::uint16_t);
%warpExp(float);
%warpExp(double);
