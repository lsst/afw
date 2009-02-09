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
%enddef

SWIG_SHARED_PTR_DERIVED(Kernel, lsst::daf::data::LsstBase, lsst::afw::math::Kernel); // the base class

%kernelPtr(AnalyticKernel);
%kernelPtr(DeltaFunctionKernel);
%kernelPtr(FixedKernel);
%kernelPtr(LinearCombinationKernel);
%kernelPtr(SeparableKernel);

%include "lsst/afw/math/Kernel.h"
%include "lsst/afw/math/KernelFunctions.h"

%template(VectorKernel)         std::vector<lsst::afw::math::Kernel::PtrT>;
%template(VectorKernelA)        std::vector<lsst::afw::math::AnalyticKernel::PtrT>;
%template(VectorKernelDF)       std::vector<lsst::afw::math::DeltaFunctionKernel::PtrT>;
%template(KernelListD_)         lsst::afw::math::KernelList<>;
%template(KernelListD)          lsst::afw::math::KernelList<lsst::afw::math::Kernel>;
%template(AnalyticKernelListD)  lsst::afw::math::KernelList<lsst::afw::math::AnalyticKernel>;
%template(DeltaFunctionKernelListD)  lsst::afw::math::KernelList<lsst::afw::math::DeltaFunctionKernel>;

// Create conversion constructors 
%extend lsst::afw::math::KernelList<lsst::afw::math::Kernel> {
    %template(KernelListDD) KernelList<lsst::afw::math::AnalyticKernel>; // Conversion constructor
    %template(KernelListDD) KernelList<lsst::afw::math::DeltaFunctionKernel>; // Conversion constructor
};

//
// Functions to convolve a (Masked)?Image with a Kernel.  There are a lot of these,
// so write a set of macros to do the instantiations
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
    %template(convolve)       lsst::afw::math::convolve<IMAGE(PIXTYPE1), IMAGE(PIXTYPE2),
                                                        lsst::afw::math::Kernel>;
    %template(convolve)       lsst::afw::math::convolve<IMAGE(PIXTYPE1), IMAGE(PIXTYPE2),
                                                        lsst::afw::math::AnalyticKernel>;
    %template(convolve)       lsst::afw::math::convolve<IMAGE(PIXTYPE1), IMAGE(PIXTYPE2),
                                                        lsst::afw::math::DeltaFunctionKernel>;
    %template(convolve)       lsst::afw::math::convolve<IMAGE(PIXTYPE1), IMAGE(PIXTYPE2),
                                                        lsst::afw::math::LinearCombinationKernel>;
    %template(convolveLinear) lsst::afw::math::convolveLinear<IMAGE(PIXTYPE1), IMAGE(PIXTYPE2)>;
    %template(convolve)       lsst::afw::math::convolve<IMAGE(PIXTYPE1), IMAGE(PIXTYPE2),
                                                        lsst::afw::math::SeparableKernel>;
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
namespace lsst { namespace afw { namespace image {
             //typedef unsigned short MaskPixel;
             typedef float VariancePixel;
}}}
         
%include "lsst/afw/math/ConvolveImage.h"

%convolutionFuncs(double, double);
%convolutionFuncs(double, float);
%convolutionFuncs(float, float);
%convolutionFuncs(boost::uint16_t, boost::uint16_t);

