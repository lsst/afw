// -*- lsst-c++ -*-
%{
#include "lsst/afw/math/detail/Convolve.h"
%}

SWIG_SHARED_PTR_DERIVED(KernelImagesForRegion,
    lsst::daf::data::LsstBase, lsst::afw::math::detail::KernelImagesForRegion);

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
    %template(basicConvolve) lsst::afw::math::detail::basicConvolve<IMAGE(PIXTYPE1), IMAGE(PIXTYPE2)>;
    %template(convolveWithBruteForce)
        lsst::afw::math::detail::convolveWithBruteForce<IMAGE(PIXTYPE1), IMAGE(PIXTYPE2)>;
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
    %templateConvolveByType(%IMAGE,       PIXTYPE1, PIXTYPE2);
    %templateConvolveByType(%MASKEDIMAGE, PIXTYPE1, PIXTYPE2);
%enddef
//
// Finally, specify the functions we want
//
%templateConvolve(double, double);
%templateConvolve(double, float);
%templateConvolve(double, int);
%templateConvolve(double, boost::uint16_t);
%templateConvolve(float, float);
%templateConvolve(float, int);
%templateConvolve(float, boost::uint16_t);
%templateConvolve(int, int);
%templateConvolve(boost::uint16_t, boost::uint16_t);

// Note: cannot swig std::vector<lsst::afw::math::detail::KernelImagesForRegion> in the obvious way:
// %template(KernelImagesForRegionList) std::vector<lsst::afw::math::detail::KernelImagesForRegion>;
// because there is no KernelImagesForRegionList default constructor. Sigh.
