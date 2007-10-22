
%{
#include "lsst/fw/Function.h"
#include "lsst/fw/FunctionLibrary.h"
%}

// I'm not sure newobject is needed (the memory leak test works without it)
%newobject lsst::fw::function::Function::getParameters;

%include "lsst/fw/Function.h"
%include "lsst/fw/FunctionLibrary.h"

%template(FunctionF)          lsst::fw::function::Function<float>;
%template(Function1F)         lsst::fw::function::Function1<float>;
%template(Function2F)         lsst::fw::function::Function2<float>;
%template(Function2PtrTypeF)    boost::shared_ptr<lsst::fw::function::Function2<float> >;

%template(Chebyshev1Function1F) lsst::fw::function::Chebyshev1Function1<float>;
%template(GaussianFunction1F) lsst::fw::function::GaussianFunction1<float>;
%template(GaussianFunction2F) lsst::fw::function::GaussianFunction2<float>;
%template(IntegerDeltaFunction2F) lsst::fw::function::IntegerDeltaFunction2<float>;
%template(LanczosFunction1F) lsst::fw::function::LanczosFunction1<float>;
%template(LanczosFunction2F) lsst::fw::function::LanczosFunction2<float>;
%template(LanczosSeparableFunction2F) lsst::fw::function::LanczosSeparableFunction2<float>;
%template(PolynomialFunction1F) lsst::fw::function::PolynomialFunction1<float>;
%template(PolynomialFunction2F) lsst::fw::function::PolynomialFunction2<float>;

%template(FunctionD)          lsst::fw::function::Function<double>;
%template(Function1D)         lsst::fw::function::Function1<double>;
%template(Function2D)         lsst::fw::function::Function2<double>;
%template(Function2PtrTypeD)    boost::shared_ptr<lsst::fw::function::Function2<double> >;

%template(Chebyshev1Function1D) lsst::fw::function::Chebyshev1Function1<double>;
%template(GaussianFunction1D) lsst::fw::function::GaussianFunction1<double>;
%template(GaussianFunction2D) lsst::fw::function::GaussianFunction2<double>;
%template(IntegerDeltaFunction2D) lsst::fw::function::IntegerDeltaFunction2<double>;
%template(LanczosFunction1D) lsst::fw::function::LanczosFunction1<double>;
%template(LanczosFunction2D) lsst::fw::function::LanczosFunction2<double>;
%template(LanczosSeparableFunction2D) lsst::fw::function::LanczosSeparableFunction2<double>;
%template(PolynomialFunction1D) lsst::fw::function::PolynomialFunction1<double>;
%template(PolynomialFunction2D) lsst::fw::function::PolynomialFunction2<double>;
