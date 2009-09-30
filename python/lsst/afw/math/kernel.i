// -*- lsst-c++ -*-
%{
#include "lsst/afw/math/Kernel.h"
#include "lsst/afw/math/KernelFunctions.h"
%}

// I doubt newobject is needed; the code seems to work just as well without it.
%newobject lsst::afw::math::convolve;
%newobject lsst::afw::math::Kernel::getKernelParameters;
%newobject lsst::afw::math::Kernel::getSpatialParameters;
//
// Kernel classes (every template of a class must have a unique name)
//
// These definitions must go Before you include Kernel.h; the %templates must go After
//
%define %kernelPtr(TYPE...)
SWIG_SHARED_PTR_DERIVED(TYPE, lsst::afw::math::Kernel, lsst::afw::math::TYPE);
%lsst_persistable(lsst::afw::math::TYPE)
%enddef

SWIG_SHARED_PTR_DERIVED(Kernel, lsst::daf::data::LsstBase, lsst::afw::math::Kernel); // the base class
%lsst_persistable(lsst::afw::math::Kernel)

%kernelPtr(AnalyticKernel);
%kernelPtr(DeltaFunctionKernel);
%kernelPtr(FixedKernel);
%kernelPtr(LinearCombinationKernel);
%kernelPtr(SeparableKernel);

%include "lsst/afw/math/Kernel.h"

%template(KernelList) std::vector<boost::shared_ptr<lsst::afw::math::Kernel> >;

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
// Note that IMAGE is a macro, not a class name
%define %convolutionFuncsByType(IMAGE, PIXTYPE1, PIXTYPE2)
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
%enddef
//
// Now a macro to specify Image and MaskedImage
//
%define %convolutionFuncs(PIXTYPE1, PIXTYPE2)
    %convolutionFuncsByType(%IMAGE,       PIXTYPE1, PIXTYPE2);
    %convolutionFuncsByType(%MASKEDIMAGE, PIXTYPE1, PIXTYPE2);
%enddef
//
// Finally, specify the functions we want
//
%convolutionFuncs(double, double);
%convolutionFuncs(double, float);
%convolutionFuncs(float, float);
%convolutionFuncs(boost::uint16_t, boost::uint16_t);
         
//
// When swig sees a Kernel it doesn't know about KERNEL_TYPE; all it knows is that it
// has a Kernel, and Kernels don't know about e.g. LinearCombinationKernel's getKernelParameters()
//
// We therefore provide a cast to KERNEL_TYPE* and swig can go from there
//
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
