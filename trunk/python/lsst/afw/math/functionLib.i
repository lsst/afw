// -*- lsst-c++ -*-
%{
#include "lsst/afw/math/Function.h"
#include "lsst/afw/math/FunctionLibrary.h"
%}

// I'm not sure newobject is needed (the memory leak test works without it)
%newobject lsst::afw::math::Function::getParameters;

// Must come before %include
%define %baseFunctionPtr(TYPE, CTYPE)
SWIG_SHARED_PTR_DERIVED(Function##TYPE, lsst::daf::data::LsstBase, lsst::afw::math::Function<CTYPE>);
%enddef

%define %baseFunctionNPtr(N, TYPE, CTYPE)
SWIG_SHARED_PTR_DERIVED(Function##N##TYPE, lsst::afw::math::Function<CTYPE>, lsst::afw::math::Function##N<CTYPE>);
%enddef

%define %functionPtr(NAME, N, TYPE, CTYPE)
SWIG_SHARED_PTR_DERIVED(NAME##N##TYPE, lsst::afw::math::Function##N<CTYPE>, lsst::afw::math::NAME##N<CTYPE>);
%enddef

// Must come after %include
%define %baseFunction(TYPE, CTYPE)
%template(Function##TYPE) lsst::afw::math::Function<CTYPE>;
%enddef

%define %baseFunctionN(N, TYPE, CTYPE)
%template(Function##N##TYPE) lsst::afw::math::Function##N<CTYPE>;
%enddef

%define %function(NAME, N, TYPE, CTYPE)
%template(NAME##N##TYPE) lsst::afw::math::NAME##N<CTYPE>;
%enddef


%baseFunctionPtr(F, float);
%baseFunctionNPtr(1, F, float);
%baseFunctionNPtr(2, F, float);

%functionPtr(Chebyshev1Function, 1, F, float);
%functionPtr(DoubleGaussianFunction, 2, F, float);
%functionPtr(GaussianFunction, 1, F, float);
%functionPtr(GaussianFunction, 2, F, float);
%functionPtr(IntegerDeltaFunction, 2, F, float);
%functionPtr(LanczosFunction, 1, F, float);
%functionPtr(LanczosFunction, 2, F, float);
%functionPtr(PolynomialFunction, 1, F, float);
%functionPtr(PolynomialFunction, 2, F, float);

%baseFunctionPtr(D, double);
%baseFunctionNPtr(1, D, double);
%baseFunctionNPtr(2, D, double);

%functionPtr(Chebyshev1Function, 1, D, double);
%functionPtr(DoubleGaussianFunction, 2, D, double);
%functionPtr(GaussianFunction, 1, D, double);
%functionPtr(GaussianFunction, 2, D, double);
%functionPtr(IntegerDeltaFunction, 2, D, double);
%functionPtr(LanczosFunction, 1, D, double);
%functionPtr(LanczosFunction, 2, D, double);
%functionPtr(PolynomialFunction, 1, D, double);
%functionPtr(PolynomialFunction, 2, D, double);

%include "lsst/afw/math/Function.h"
%include "lsst/afw/math/FunctionLibrary.h"

%baseFunction(F, float);
%baseFunctionN(1, F, float);
%baseFunctionN(2, F, float);

%function(Chebyshev1Function, 1, F, float);
%function(DoubleGaussianFunction, 2, F, float);
%function(GaussianFunction, 1, F, float);
%function(GaussianFunction, 2, F, float);
%function(IntegerDeltaFunction, 2, F, float);
%function(LanczosFunction, 1, F, float);
%function(LanczosFunction, 2, F, float);
%function(PolynomialFunction, 1, F, float);
%function(PolynomialFunction, 2, F, float);

%baseFunction(D, double);
%baseFunctionN(1, D, double);
%baseFunctionN(2, D, double);

%function(Chebyshev1Function, 1, D, double);
%function(DoubleGaussianFunction, 2, D, double);
%function(GaussianFunction, 1, D, double);
%function(GaussianFunction, 2, D, double);
%function(IntegerDeltaFunction, 2, D, double);
%function(LanczosFunction, 1, D, double);
%function(LanczosFunction, 2, D, double);
%function(PolynomialFunction, 1, D, double);
%function(PolynomialFunction, 2, D, double);

