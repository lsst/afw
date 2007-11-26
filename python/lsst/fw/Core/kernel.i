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
%template(KernelD)              lsst::fw::Kernel<double>;
%template(LinearCombinationKernelD) lsst::fw::LinearCombinationKernel<double>;
%boost_shared_ptr(KernelDPtr,   lsst::fw::Kernel<double>);
%template(vectorKernelDPtr)     std::vector<boost::shared_ptr<lsst::fw::Kernel<double> > >;

%template(FixedKernelD)         lsst::fw::FixedKernel<double>;
%template(AnalyticKernelD)      lsst::fw::AnalyticKernel<double>;
%template(LinearCombinationKernelD) lsst::fw::LinearCombinationKernel<double>;

//
// functions (every template can have the same name)
//
%template(printKernel)          lsst::fw::kernel::printKernel<double>;

%template(convolve)             lsst::fw::kernel::convolve<double, lsst::fw::maskPixelType, double>;
%template(convolve)             lsst::fw::kernel::convolve<float, lsst::fw::maskPixelType, double>;
%template(convolve)             lsst::fw::kernel::convolve<boost::uint16_t, lsst::fw::maskPixelType, double>;

%template(convolveLinear)       lsst::fw::kernel::convolveLinear<double, lsst::fw::maskPixelType, double>;
%template(convolveLinear)       lsst::fw::kernel::convolveLinear<float, lsst::fw::maskPixelType, double>;
%template(convolveLinear)       lsst::fw::kernel::convolveLinear<boost::uint16_t, lsst::fw::maskPixelType, double>;

