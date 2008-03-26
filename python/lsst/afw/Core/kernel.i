// -*- lsst-c++ -*-
%{
#include "lsst/fw/Kernel.h"
#include "lsst/fw/KernelFunctions.h"
%}

// I doubt newobject is needed; the code seems to work just as well without it.
%newobject lsst::fw::kernel::convolve;
%newobject lsst::fw::Kernel::computeNewImage;
%newobject lsst::fw::Kernel::getKernelParameters;
%newobject lsst::fw::Kernel::getSpatialParameters;

// Handle return-by-reference argument.
%apply double& OUTPUT { double& imSum };
%include "lsst/fw/Kernel.h"
%include "lsst/fw/KernelFunctions.h"
//
// classes (every template must have a unique name)
//
%boost_shared_ptr(KernelPtr,   lsst::fw::Kernel);
%boost_shared_ptr(LinearCombinationKernelPtr,  lsst::fw::LinearCombinationKernel);

%template(VectorKernel)         std::vector<lsst::fw::Kernel::PtrT>;
%template(VectorKernelA)        std::vector<lsst::fw::AnalyticKernel::PtrT>;
%template(VectorKernelDF)       std::vector<lsst::fw::DeltaFunctionKernel::PtrT>;
%template(KernelListD_)         lsst::fw::KernelList<>;
%template(KernelListD)          lsst::fw::KernelList<lsst::fw::Kernel>;
%template(AnalyticKernelListD)  lsst::fw::KernelList<lsst::fw::AnalyticKernel>;
%template(DeltaFunctionKernelListD)  lsst::fw::KernelList<lsst::fw::DeltaFunctionKernel>;

// Create conversion constructors 
%extend lsst::fw::KernelList<lsst::fw::Kernel> {
    %template(KernelListDD) KernelList<lsst::fw::AnalyticKernel>; // Conversion constructor
    %template(KernelListDD) KernelList<lsst::fw::DeltaFunctionKernel>; // Conversion constructor
};

//%extend lsst::fw::KernelList {
//    %template(KernelList) KernelList<lsst::fw::AnalyticKernel>; // Conversion constructor
//};

//
// functions (every template can have the same name)
//
%template(convolve)             lsst::fw::kernel::convolve<double, lsst::fw::maskPixelType, lsst::fw::Kernel>;
%template(convolve)             lsst::fw::kernel::convolve<double, lsst::fw::maskPixelType, lsst::fw::DeltaFunctionKernel>;
%template(convolve)             lsst::fw::kernel::convolve<float, lsst::fw::maskPixelType, lsst::fw::Kernel>;
%template(convolve)             lsst::fw::kernel::convolve<float, lsst::fw::maskPixelType, lsst::fw::DeltaFunctionKernel>;
%template(convolve)             lsst::fw::kernel::convolve<boost::uint16_t, lsst::fw::maskPixelType, lsst::fw::Kernel>;
%template(convolve)             lsst::fw::kernel::convolve<boost::uint16_t, lsst::fw::maskPixelType, lsst::fw::DeltaFunctionKernel>;

%template(convolveLinear)       lsst::fw::kernel::convolveLinear<double, lsst::fw::maskPixelType>;
%template(convolveLinear)       lsst::fw::kernel::convolveLinear<float, lsst::fw::maskPixelType>;
%template(convolveLinear)       lsst::fw::kernel::convolveLinear<boost::uint16_t, lsst::fw::maskPixelType>;

/******************************************************************************/
// Local Variables: ***
// eval: (setq indent-tabs-mode nil) ***
// End: ***

















