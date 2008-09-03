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

%include "lsst/afw/math/ConvolveMaskedImage.h"
//
// functions to convolve a MaskedImage with a Kernel that return a new MaskedImage
//
%template(convolve)             lsst::afw::math::convolve<double, lsst::afw::image::maskPixelType, lsst::afw::math::Kernel>;
%template(convolve)             lsst::afw::math::convolve<double, lsst::afw::image::maskPixelType, lsst::afw::math::DeltaFunctionKernel>;
%template(convolve)             lsst::afw::math::convolve<double, lsst::afw::image::maskPixelType, lsst::afw::math::SeparableKernel>;
%template(convolve)             lsst::afw::math::convolve<float,  lsst::afw::image::maskPixelType, lsst::afw::math::Kernel>;
%template(convolve)             lsst::afw::math::convolve<float,  lsst::afw::image::maskPixelType, lsst::afw::math::DeltaFunctionKernel>;
%template(convolve)             lsst::afw::math::convolve<float,  lsst::afw::image::maskPixelType, lsst::afw::math::SeparableKernel>;
%template(convolve)             lsst::afw::math::convolve<float,  lsst::afw::image::maskPixelType, lsst::afw::math::Kernel>;
%template(convolve)             lsst::afw::math::convolve<float,  lsst::afw::image::maskPixelType, lsst::afw::math::DeltaFunctionKernel>;
%template(convolve)             lsst::afw::math::convolve<float,  lsst::afw::image::maskPixelType, lsst::afw::math::SeparableKernel>;
%template(convolve)             lsst::afw::math::convolve<boost::uint16_t, lsst::afw::image::maskPixelType, lsst::afw::math::Kernel>;
%template(convolve)             lsst::afw::math::convolve<boost::uint16_t, lsst::afw::image::maskPixelType, lsst::afw::math::DeltaFunctionKernel>;
%template(convolve)             lsst::afw::math::convolve<boost::uint16_t, lsst::afw::image::maskPixelType, lsst::afw::math::SeparableKernel>;

%template(convolveLinear)       lsst::afw::math::convolveLinear<double, lsst::afw::image::maskPixelType>;
%template(convolveLinear)       lsst::afw::math::convolveLinear<float,  lsst::afw::image::maskPixelType>;
%template(convolveLinear)       lsst::afw::math::convolveLinear<boost::uint16_t, lsst::afw::image::maskPixelType>;

//
// functions to convolve a MaskedImage with a Kernel that work in-place
//
%template(convolve)             lsst::afw::math::convolve<double, double, lsst::afw::image::maskPixelType, lsst::afw::math::Kernel>;
%template(convolve)             lsst::afw::math::convolve<double, double, lsst::afw::image::maskPixelType, lsst::afw::math::DeltaFunctionKernel>;
%template(convolve)             lsst::afw::math::convolve<double, double, lsst::afw::image::maskPixelType, lsst::afw::math::SeparableKernel>;
%template(convolve)             lsst::afw::math::convolve<double, float,  lsst::afw::image::maskPixelType, lsst::afw::math::Kernel>;
%template(convolve)             lsst::afw::math::convolve<double, float,  lsst::afw::image::maskPixelType, lsst::afw::math::DeltaFunctionKernel>;
%template(convolve)             lsst::afw::math::convolve<double, float,  lsst::afw::image::maskPixelType, lsst::afw::math::SeparableKernel>;
%template(convolve)             lsst::afw::math::convolve<float,  float,  lsst::afw::image::maskPixelType, lsst::afw::math::Kernel>;
%template(convolve)             lsst::afw::math::convolve<float,  float,  lsst::afw::image::maskPixelType, lsst::afw::math::DeltaFunctionKernel>;
%template(convolve)             lsst::afw::math::convolve<float,  float,  lsst::afw::image::maskPixelType, lsst::afw::math::SeparableKernel>;
%template(convolve)             lsst::afw::math::convolve<boost::uint16_t, boost::uint16_t, lsst::afw::image::maskPixelType, lsst::afw::math::Kernel>;
%template(convolve)             lsst::afw::math::convolve<boost::uint16_t, boost::uint16_t, lsst::afw::image::maskPixelType, lsst::afw::math::DeltaFunctionKernel>;
%template(convolve)             lsst::afw::math::convolve<boost::uint16_t, boost::uint16_t, lsst::afw::image::maskPixelType, lsst::afw::math::SeparableKernel>;

%template(convolveLinear)       lsst::afw::math::convolveLinear<double, double, lsst::afw::image::maskPixelType>;
%template(convolveLinear)       lsst::afw::math::convolveLinear<double, float,  lsst::afw::image::maskPixelType>;
%template(convolveLinear)       lsst::afw::math::convolveLinear<float,  float, lsst::afw::image::maskPixelType>;
%template(convolveLinear)       lsst::afw::math::convolveLinear<boost::uint16_t, boost::uint16_t, lsst::afw::image::maskPixelType>;


/* The following conflicts with the above (MaskedImage functions that return a new MI are ambiguous with Image in-place functions)
The likely solution is to rename the functions that return a new MI or new Image in C++, but first I've asked aroud.
*/
// %include "lsst/afw/math/ConvolveImage.h"
// //
// // functions to convolve an Image with a Kernel that return a new Image
// //
// %template(convolve)             lsst::afw::math::convolve<double, lsst::afw::math::DeltaFunctionKernel>;
// %template(convolve)             lsst::afw::math::convolve<double, lsst::afw::math::SeparableKernel>;
// %template(convolve)             lsst::afw::math::convolve<float, lsst::afw::math::Kernel>;
// %template(convolve)             lsst::afw::math::convolve<float, lsst::afw::math::DeltaFunctionKernel>;
// %template(convolve)             lsst::afw::math::convolve<float, lsst::afw::math::SeparableKernel>;
// %template(convolve)             lsst::afw::math::convolve<float, lsst::afw::math::Kernel>;
// %template(convolve)             lsst::afw::math::convolve<float, lsst::afw::math::DeltaFunctionKernel>;
// %template(convolve)             lsst::afw::math::convolve<float, lsst::afw::math::SeparableKernel>;
// %template(convolve)             lsst::afw::math::convolve<boost::uint16_t, lsst::afw::math::Kernel>;
// %template(convolve)             lsst::afw::math::convolve<boost::uint16_t, lsst::afw::math::DeltaFunctionKernel>;
// %template(convolve)             lsst::afw::math::convolve<boost::uint16_t, lsst::afw::math::SeparableKernel>;
// 
// %template(convolveLinear)       lsst::afw::math::convolveLinear<double>;
// %template(convolveLinear)       lsst::afw::math::convolveLinear<float>;
// %template(convolveLinear)       lsst::afw::math::convolveLinear<boost::uint16_t>;
// 
// //
// // functions to convolve an Image with a Kernel that work in-place
// //
// %template(convolve)             lsst::afw::math::convolve<double, double, lsst::afw::math::Kernel>;
// %template(convolve)             lsst::afw::math::convolve<double, double, lsst::afw::math::DeltaFunctionKernel>;
// %template(convolve)             lsst::afw::math::convolve<double, double, lsst::afw::math::SeparableKernel>;
// %template(convolve)             lsst::afw::math::convolve<double, float, lsst::afw::math::Kernel>;
// %template(convolve)             lsst::afw::math::convolve<double, float, lsst::afw::math::DeltaFunctionKernel>;
// %template(convolve)             lsst::afw::math::convolve<double, float, lsst::afw::math::SeparableKernel>;
// %template(convolve)             lsst::afw::math::convolve<float,  float, lsst::afw::math::Kernel>;
// %template(convolve)             lsst::afw::math::convolve<float,  float, lsst::afw::math::DeltaFunctionKernel>;
// %template(convolve)             lsst::afw::math::convolve<float,  float, lsst::afw::math::SeparableKernel>;
// %template(convolve)             lsst::afw::math::convolve<boost::uint16_t, boost::uint16_t, lsst::afw::math::Kernel>;
// %template(convolve)             lsst::afw::math::convolve<boost::uint16_t, boost::uint16_t, lsst::afw::math::DeltaFunctionKernel>;
// %template(convolve)             lsst::afw::math::convolve<boost::uint16_t, boost::uint16_t, lsst::afw::math::SeparableKernel>;
// 
// %template(convolveLinear)       lsst::afw::math::convolveLinear<double, double>;
// %template(convolveLinear)       lsst::afw::math::convolveLinear<double, float>;
// %template(convolveLinear)       lsst::afw::math::convolveLinear<float,  float>;
// %template(convolveLinear)       lsst::afw::math::convolveLinear<boost::uint16_t, boost::uint16_t>;


/******************************************************************************/
// Local Variables: ***
// eval: (setq indent-tabs-mode nil) ***
// End: ***


