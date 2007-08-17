
%{
#include "lsst/fw/Kernel.h"
#include "lsst/fw/KernelFunctions.h"
%}

%include "lsst/fw/Kernel.h"
%include "lsst/fw/KernelFunctions.h"

%template(KernelF)              lsst::fw::Kernel<float>;
%template(FixedKernelF)         lsst::fw::FixedKernel<float>;
%template(AnalyticKernelF)      lsst::fw::AnalyticKernel<float>;
%template(LinearCombinationKernelF) lsst::fw::LinearCombinationKernel<float>;

%template(Function2PtrTypeF)    boost::shared_ptr<lsst::fw::function::Function2<float> >;
%template(KernelPtrTypeF)       boost::shared_ptr<lsst::fw::Kernel<float> >;

%template(printKernelF)         lsst::fw::kernel::printKernel<float>;

%template(convolveF)            lsst::fw::kernel::convolve<ImagePixelType, MaskPixelType, float>;

%template(KernelD)              lsst::fw::Kernel<double>;
%template(FixedKernelD)         lsst::fw::FixedKernel<double>;
%template(AnalyticKernelD)      lsst::fw::AnalyticKernel<double>;
%template(LinearCombinationKernelD) lsst::fw::LinearCombinationKernel<double>;

%template(Function2PtrTypeD)    boost::shared_ptr<lsst::fw::function::Function2<double> >;
%template(KernelPtrTypeD)       boost::shared_ptr<lsst::fw::Kernel<double> >;

%template(printKernelD)         lsst::fw::kernel::printKernel<double>;

// define kernel-related vectors
%template(kernelPtrVectorF)     std::vector<boost::shared_ptr<lsst::fw::Kernel<float> > >;
%template(kernelPtrVectorD)     std::vector<boost::shared_ptr<lsst::fw::Kernel<double> > >;
