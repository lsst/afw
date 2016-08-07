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
#include <cstdint>

#include "lsst/afw/math/detail/Convolve.h"
%}

%shared_ptr(lsst::afw::math::detail::KernelImagesForRegion);

%include "lsst/afw/math/detail/Convolve.h"

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
// Note that IMAGE is a macro, not a class name
%define %templateConvolveByType(IMAGE, PIXTYPE1, PIXTYPE2)
    %template(basicConvolve) lsst::afw::math::detail::basicConvolve<IMAGE(PIXTYPE1), IMAGE(PIXTYPE2)>;
    %template(convolveWithBruteForce)
        lsst::afw::math::detail::convolveWithBruteForce<IMAGE(PIXTYPE1), IMAGE(PIXTYPE2)>;
    %template(convolveWithInterpolation)
        lsst::afw::math::detail::convolveWithInterpolation<IMAGE(PIXTYPE1), IMAGE(PIXTYPE2)>;
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
%templateConvolve(double, std::uint16_t);
%templateConvolve(float, float);
%templateConvolve(float, int);
%templateConvolve(float, std::uint16_t);
%templateConvolve(int, int);
%templateConvolve(std::uint16_t, std::uint16_t);
