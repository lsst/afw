%{
#include "lsst/fw/Function.h"
#include "lsst/fw/FunctionLibrary.h"
#include "lsst/fw/Kernel.h"
#include "lsst/fw/KernelFunctions.h"
%}

%include "lsst/fw/Function.h"
%include "lsst/fw/FunctionLibrary.h"
%include "lsst/fw/Kernel.h"
%include "lsst/fw/KernelFunctions.h"

%inline %{
    typedef float pixelType;
%}

%template(KernelD)         lsst::fw::Kernel<pixelType>;
%template(FixedKernelD)    lsst::fw::FixedKernel<pixelType>;
%template(AnalyticKernelD) lsst::fw::AnalyticKernel<pixelType>;
%template(LinearCombinationKernelD) lsst::fw::LinearCombinationKernel<pixelType>;

%template(FunctionD)          lsst::fw::function::Function<pixelType>;
%template(Function1D)         lsst::fw::function::Function1<pixelType>;
%template(Function2D)         lsst::fw::function::Function2<pixelType>;
%template(Chebyshev1Function1D) lsst::fw::function::Chebyshev1Function1<pixelType>;
%template(GaussianFunction1D) lsst::fw::function::GaussianFunction1<pixelType>;
%template(GaussianFunction2D) lsst::fw::function::GaussianFunction2<pixelType>;
%template(IntegerDeltaFunction2D) lsst::fw::function::IntegerDeltaFunction2<pixelType>;
%template(LanczosFunction1D) lsst::fw::function::LanczosFunction1<pixelType>;
%template(LanczosFunction2D) lsst::fw::function::LanczosFunction2<pixelType>;
%template(PolynomialFunction1D) lsst::fw::function::PolynomialFunction1<pixelType>;
%template(PolynomialFunction2D) lsst::fw::function::PolynomialFunction2<pixelType>;

%template(Function2PtrTypeD)  boost::shared_ptr<lsst::fw::function::Function2<pixelType> >;

%template(printKernelD)       lsst::fw::kernel::printKernel<pixelType>;
