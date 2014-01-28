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
#include "lsst/afw/math/Kernel.h"
#include "lsst/afw/math/KernelFunctions.h"
#include "lsst/afw/formatters/KernelFormatter.h"
%}

%include "std_complex.i"
%include "../boost_picklable.i"

%import "lsst/afw/table/io/ioLib.i"

//
// Kernel classes (every template of a class must have a unique name)
//
// These definitions must go Before you include Kernel.h; the %templates must go After
//
%define %kernelPtr(TYPE...)
%declareTablePersistable(TYPE, lsst::afw::math::TYPE)
%lsst_persistable(lsst::afw::math::TYPE)
%boost_picklable(lsst::afw::math::TYPE)
%enddef

%kernelPtr(Kernel);
%kernelPtr(AnalyticKernel);
%kernelPtr(DeltaFunctionKernel);
%kernelPtr(FixedKernel);
%kernelPtr(LinearCombinationKernel);
%kernelPtr(SeparableKernel);

%include "lsst/afw/math/Kernel.h"

%include "lsst/afw/math/KernelFunctions.h"

%include "lsst/afw/math/ConvolveImage.h"
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
%define %templateKernelByType(IMAGE, PIXTYPE1, PIXTYPE2)
    %template(convolve) lsst::afw::math::convolve<
        IMAGE(PIXTYPE1), IMAGE(PIXTYPE2), lsst::afw::math::Kernel>;
    %template(convolve) lsst::afw::math::convolve<
        IMAGE(PIXTYPE1), IMAGE(PIXTYPE2), lsst::afw::math::AnalyticKernel>;
    %template(convolve) lsst::afw::math::convolve<
        IMAGE(PIXTYPE1), IMAGE(PIXTYPE2), lsst::afw::math::DeltaFunctionKernel>;
    %template(convolve) lsst::afw::math::convolve<
        IMAGE(PIXTYPE1), IMAGE(PIXTYPE2), lsst::afw::math::FixedKernel>;
    %template(convolve) lsst::afw::math::convolve<
        IMAGE(PIXTYPE1), IMAGE(PIXTYPE2), lsst::afw::math::LinearCombinationKernel>;
    %template(convolve) lsst::afw::math::convolve<
        IMAGE(PIXTYPE1), IMAGE(PIXTYPE2), lsst::afw::math::SeparableKernel>;
    %template(scaledPlus) lsst::afw::math::scaledPlus<IMAGE(PIXTYPE1), IMAGE(PIXTYPE2)>;
%enddef
//
// Now a macro to specify Image and MaskedImage
//
%define %templateKernel(PIXTYPE1, PIXTYPE2)
    %templateKernelByType(%IMAGE,       PIXTYPE1, PIXTYPE2);
    %templateKernelByType(%MASKEDIMAGE, PIXTYPE1, PIXTYPE2);
%enddef
//
// Finally, specify the functions we want
//
%templateKernel(double, double);
%templateKernel(double, float);
%templateKernel(double, int);
%templateKernel(double, boost::uint16_t);
%templateKernel(float, float);
%templateKernel(float, int);
%templateKernel(float, boost::uint16_t);
%templateKernel(int, int);
%templateKernel(boost::uint16_t, boost::uint16_t);
  
//-------------------------------------------------------------------------
// THIS CAST INTERFACE NOW DEPRECATED IN FAVOR OF %castShared
%define %dynamic_cast(KERNEL_TYPE)
%inline %{
    lsst::afw::math::KERNEL_TYPE *
        cast_##KERNEL_TYPE(lsst::afw::math::Kernel *candidate) {
        return dynamic_cast<lsst::afw::math::KERNEL_TYPE *>(candidate);
    }
%}
%enddef
%dynamic_cast(AnalyticKernel);
%dynamic_cast(DeltaFunctionKernel);
%dynamic_cast(FixedKernel);
%dynamic_cast(LinearCombinationKernel);
%dynamic_cast(SeparableKernel);
//-------------------------------------------------------------------------

%castShared(lsst::afw::math::AnalyticKernel, lsst::afw::math::Kernel)
%castShared(lsst::afw::math::DeltaFunctionKernel, lsst::afw::math::Kernel)
%castShared(lsst::afw::math::FixedKernel, lsst::afw::math::Kernel)
%castShared(lsst::afw::math::LinearCombinationKernel, lsst::afw::math::Kernel)
%castShared(lsst::afw::math::SeparableKernel, lsst::afw::math::Kernel)
