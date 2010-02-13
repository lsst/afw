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

namespace afwImage = lsst::afw::image;

%define %WarpFuncsByType(DESTIMAGEPIXELT, SRCIMAGEPIXELT)
%template(warpExposure) lsst::afw::math::warpExposure<
    afwImage::Exposure<DESTIMAGEPIXELT, afwImage::MaskPixel, afwImage::VariancePixel>,
    afwImage::Exposure<SRCIMAGEPIXELT, afwImage::MaskPixel, afwImage::VariancePixel> >;
%template(warpImage) lsst::afw::math::warpImage<
    afwImage::MaskedImage<DESTIMAGEPIXELT, afwImage::MaskPixel, afwImage::VariancePixel>,
    afwImage::MaskedImage<SRCIMAGEPIXELT, afwImage::MaskPixel, afwImage::VariancePixel> >;
%template(warpImage) lsst::afw::math::warpImage<
    afwImage::Image<DESTIMAGEPIXELT>,
    afwImage::Image<SRCIMAGEPIXELT> >;
%template(warpImageToSkyMap) lsst::afw::math::warpImageToSkyMap<
    afwImage::SkyMapImage<
        afwImage::HealPixId,
        afwImage::MaskedImage<DESTIMAGEPIXELT, afwImage::MaskPixel, afwImage::VariancePixel>::SinglePixel
    >,
    afwImage::MaskedImage<SRCIMAGEPIXELT, afwImage::MaskPixel, afwImage::VariancePixel>
>;
%template(warpImageToSkyMap) lsst::afw::math::warpImageToSkyMap<
    afwImage::SkyMapImage<afwImage::HealPixId, afwImage::Image<DESTIMAGEPIXELT>::SinglePixel >,
    afwImage::Image<SRCIMAGEPIXELT> >;
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

%template(offsetImage) lsst::afw::math::offsetImage<afwImage::Image<double> >;
%template(offsetImage) lsst::afw::math::offsetImage<afwImage::Image<float> >;

%define rotateImageBy90(PIXELT)
%template(rotateImageBy90) lsst::afw::math::rotateImageBy90<afwImage::Image<PIXELT> >;
#if 0
%template(rotateImageBy90) lsst::afw::math::rotateImageBy90<
    afwImage::MaskedImage<PIXELT, afwImage::MaskPixel, afwImage::VariancePixel> >;
#endif
%enddef

rotateImageBy90(boost::uint16_t);
rotateImageBy90(int);
rotateImageBy90(float);
rotateImageBy90(double);
