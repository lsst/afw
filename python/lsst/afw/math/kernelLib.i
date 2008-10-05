// -*- lsst-c++ -*-
%{
#include "lsst/afw/math/Kernel.h"
#include "lsst/afw/math/KernelFunctions.h"
%}

// I doubt newobject is needed; the code seems to work just as well without it.
%newobject lsst::afw::math::convolve;
%newobject lsst::afw::math::Kernel::computeNewImage;
%newobject lsst::afw::math::Kernel::getKernelParameters;
%newobject lsst::afw::math::Kernel::getSpatialParameters;

// Handle return-by-reference argument.
%apply double& OUTPUT { double& imSum };

%include "lsst/afw/math/Kernel.h"
//
// Kernel classes (every template of a class must have a unique name)
//
%boost_shared_ptr(KernelPtr,   lsst::afw::math::Kernel);
%boost_shared_ptr(LinearCombinationKernelPtr,  lsst::afw::math::LinearCombinationKernel);

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
// functions to convolve a (Masked)?Image with a Kernel that work in-place
//
// First a macro to generate needed instantiations for IMAGE (e.g. MaskedImage) and the specified pixel types
//
%define %convolutionFuncsByType(IMAGE, PIXTYPE1, PIXTYPE2)
    %template(convolve)       lsst::afw::math::convolve<lsst::afw::image::IMAGE<PIXTYPE1>,
                                                        lsst::afw::image::IMAGE<PIXTYPE2>,
                                                        lsst::afw::math::Kernel>;
    %template(convolve)       lsst::afw::math::convolve<lsst::afw::image::IMAGE<PIXTYPE1>,
                                                        lsst::afw::image::IMAGE<PIXTYPE2>,
                                                        lsst::afw::math::DeltaFunctionKernel>;
    %template(convolve)       lsst::afw::math::convolve<lsst::afw::image::IMAGE<PIXTYPE1>,
                                                        lsst::afw::image::IMAGE<PIXTYPE2>,
                                                        lsst::afw::math::SeparableKernel>;
    %template(convolveLinear) lsst::afw::math::convolveLinear<lsst::afw::image::IMAGE<PIXTYPE1>,
                                                              lsst::afw::image::IMAGE<PIXTYPE2> >;
%enddef
//
// Now a macro to specify Image and MaskedImage
//
%define %convolutionFuncs(PIXTYPE1, PIXTYPE2)
    %convolutionFuncsByType(Image, PIXTYPE1, PIXTYPE2);
    %convolutionFuncsByType(MaskedImage, PIXTYPE1, PIXTYPE2);
%enddef
//
// Finally, specify the functions we want
//
%include "lsst/afw/math/ConvolveImage.h"

%convolutionFuncs(double, double);
%convolutionFuncs(double, float);
%convolutionFuncs(float, float);
%convolutionFuncs(boost::uint16_t, boost::uint16_t);

/******************************************************************************/
// Local Variables: ***
// eval: (setq indent-tabs-mode nil) ***
// End: ***


