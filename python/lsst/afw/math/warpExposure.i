// -*- lsst-c++ -*-

/* 
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
 * 
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the LSST License Statement and 
 * the GNU General Public License along with this program.  If not, 
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */
 
%{
#include "lsst/afw/math/warpExposure.h"
#include "lsst/afw/image/Mask.h"
%}

//
// Additional kernel subclasses
//
// These definitions must go before you %include the .h file; the %templates must go after
//
%shared_ptr(lsst::afw::math::BilinearWarpingKernel);
%shared_ptr(lsst::afw::math::LanczosWarpingKernel);
%shared_ptr(lsst::afw::math::NearestWarpingKernel);

// No idea why Swig doesn't want these to be fully-qualified, but it doesn't work if they are
%warnfilter(325) BilinearFunction1;
%warnfilter(325) NearestFunction1;

%import "lsst/afw/gpu/DevicePreference.h"
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
%template(warpCenteredImage) lsst::afw::math::warpCenteredImage<
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

%define imageTransforms(PIXELT, FLOATING)
%template(binImage) lsst::afw::math::binImage<lsst::afw::image::Image<PIXELT> >;
%template(binImage) lsst::afw::math::binImage<lsst::afw::image::MaskedImage<PIXELT, lsst::afw::image::MaskPixel, lsst::afw::image::VariancePixel> >;

%template(flipImage) lsst::afw::math::flipImage<lsst::afw::image::Image<PIXELT> >;

#if FLOATING
%template(offsetImage) lsst::afw::math::offsetImage<lsst::afw::image::Image<PIXELT> >;
%template(offsetImage) lsst::afw::math::offsetImage<lsst::afw::image::MaskedImage<PIXELT, lsst::afw::image::MaskPixel, lsst::afw::image::VariancePixel> >;
#endif

%template(rotateImageBy90) lsst::afw::math::rotateImageBy90<lsst::afw::image::Image<PIXELT> >;
%template(rotateImageBy90) lsst::afw::math::rotateImageBy90<
    lsst::afw::image::MaskedImage<PIXELT, lsst::afw::image::MaskPixel, lsst::afw::image::VariancePixel> >;
%template(flipImage) lsst::afw::math::flipImage<
    lsst::afw::image::MaskedImage<PIXELT, lsst::afw::image::MaskPixel, lsst::afw::image::VariancePixel> >;
%enddef

imageTransforms(boost::uint16_t, 0);
imageTransforms(int, 0);
imageTransforms(float, 1);
imageTransforms(double, 1);

%template(rotateImageBy90) lsst::afw::math::rotateImageBy90<lsst::afw::image::Mask<boost::uint16_t> >;
%template(flipImage) lsst::afw::math::flipImage<lsst::afw::image::Mask<boost::uint16_t> >;
