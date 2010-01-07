%{
#include "lsst/afw/math/Random.h"
%}

%include "lsst/afw/math/Random.h"

%define %randomImage(TYPE)
%template(randomUniformImage)    lsst::afw::math::randomUniformImage<lsst::afw::image::Image<TYPE> >;
%template(randomUniformPosImage) lsst::afw::math::randomUniformPosImage<lsst::afw::image::Image<TYPE> >;
%template(randomUniformIntImage) lsst::afw::math::randomUniformIntImage<lsst::afw::image::Image<TYPE> >;
%template(randomFlatImage)       lsst::afw::math::randomFlatImage<lsst::afw::image::Image<TYPE> >;
%template(randomGaussianImage)   lsst::afw::math::randomGaussianImage<lsst::afw::image::Image<TYPE> >;
%template(randomChisqImage)      lsst::afw::math::randomChisqImage<lsst::afw::image::Image<TYPE> >;
%template(randomPoissonImage)    lsst::afw::math::randomPoissonImage<lsst::afw::image::Image<TYPE> >;
%enddef

%randomImage(double)
%randomImage(float)
