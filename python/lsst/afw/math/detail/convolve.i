// -*- lsst-c++ -*-
%{
#include "lsst/afw/math/detail/Convolve.h"
%}

%include "lsst/afw/math/detail/Convolve.h"
//
// Functions to convolve a MaskedImage or Image with a Kernel.
// There are a lot of these, so write a set of macros to do the instantiations
//
// First a couple of macros (%IMAGE and %MASKEDIMAGE) to provide MaskedImage's default arguments,
%define %IMAGE(PIXTYPE)
lsst::afw::image::Image<PIXTYPE>
%enddef

%define %MASKEDIMAGE(PIXTYPE)
lsst::afw::image::MaskedImage<PIXTYPE, lsst::afw::image::MaskPixel, lsst::afw::image::VariancePixel>
%enddef

// Next a macro to generate needed instantiations for IMAGE (e.g. %MASKEDIMAGE) and the specified pixel types
//
// @todo put convolveWith... functions in lsst.afw.math.detail instead of lsst.afw.math
//
// Note that IMAGE is a macro, not a class name
%define %templateConvolveByType(IMAGE, PIXTYPE1, PIXTYPE2)
    %template(basicConvolve) lsst::afw::math::basicConvolve<
        IMAGE(PIXTYPE1), IMAGE(PIXTYPE2)>;
    %template(convolveWithInterpolation)
        lsst::afw::math::detail::convolveWithInterpolation<IMAGE(PIXTYPE1), IMAGE(PIXTYPE2)>;
    %template(convolveRegionWithRecursiveInterpolation)
        lsst::afw::math::detail::convolveRegionWithRecursiveInterpolation<IMAGE(PIXTYPE1), IMAGE(PIXTYPE2)>;
    %template(convolveRegionWithInterpolation)
        lsst::afw::math::detail::convolveRegionWithInterpolation<IMAGE(PIXTYPE1), IMAGE(PIXTYPE2)>;
%enddef
//
// Now a macro to specify Image and MaskedImage
//
%define %templateConvolve(PIXTYPE1, PIXTYPE2)
    %convolutionFuncsByType(%IMAGE,       PIXTYPE1, PIXTYPE2);
    %convolutionFuncsByType(%MASKEDIMAGE, PIXTYPE1, PIXTYPE2);
%enddef
//
// Finally, specify the functions we want
//
%convolutionFuncs(double, double);
%convolutionFuncs(double, float);
%convolutionFuncs(double, int);
%convolutionFuncs(double, boost::uint16_t);
%convolutionFuncs(float, float);
%convolutionFuncs(float, int);
%convolutionFuncs(float, boost::uint16_t);
%convolutionFuncs(int, int);
%convolutionFuncs(boost::uint16_t, boost::uint16_t);
