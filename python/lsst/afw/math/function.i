// -*- lsst-c++ -*-
%{
#include <vector>

#include "boost/shared_ptr.hpp"

#include "lsst/afw/math/Function.h"
#include "lsst/afw/math/FunctionLibrary.h"
%}

// I'm not sure newobject is needed (the memory leak test works without it)
%newobject lsst::afw::math::Function::getParameters;

// Must be used before %include
%define %baseFunctionPtr(TYPE, CTYPE)
SWIG_SHARED_PTR_DERIVED(Function##TYPE, lsst::daf::data::LsstBase, lsst::afw::math::Function<CTYPE>);
%enddef

%define %baseFunctionNPtr(N, TYPE, CTYPE)
SWIG_SHARED_PTR_DERIVED(Function##N##TYPE, lsst::afw::math::Function<CTYPE>, lsst::afw::math::Function##N<CTYPE>);
%enddef

%define %functionPtr(NAME, N, TYPE, CTYPE)
SWIG_SHARED_PTR_DERIVED(NAME##N##TYPE, lsst::afw::math::Function##N<CTYPE>, lsst::afw::math::NAME##N<CTYPE>);
%enddef

// Must be used after %include
%define %baseFunction(TYPE, CTYPE)
%template(Function##TYPE) lsst::afw::math::Function<CTYPE>;
%enddef

%define %baseFunctionN(N, TYPE, CTYPE)
%template(Function##N##TYPE) lsst::afw::math::Function##N<CTYPE>;
%enddef

%define %function(NAME, N, TYPE, CTYPE)
%template(NAME##N##TYPE) lsst::afw::math::NAME##N<CTYPE>;
%enddef
//
// Macros to define float or double versions of things
//
%define %definePointers(NAME, TYPE)
    // Must be called BEFORE %include
    %baseFunctionPtr(NAME, TYPE);
    %baseFunctionNPtr(1, NAME, TYPE);
    %baseFunctionNPtr(2, NAME, TYPE);

    %functionPtr(Chebyshev1Function, 1, NAME, TYPE);
    %functionPtr(DoubleGaussianFunction, 2, NAME, TYPE);
    %functionPtr(GaussianFunction, 1, NAME, TYPE);
    %functionPtr(GaussianFunction, 2, NAME, TYPE);
    %functionPtr(IntegerDeltaFunction, 2, NAME, TYPE);
    %functionPtr(LanczosFunction, 1, NAME, TYPE);
    %functionPtr(LanczosFunction, 2, NAME, TYPE);
    %functionPtr(NullFunction, 1, NAME, TYPE);
    %functionPtr(NullFunction, 2, NAME, TYPE);
    %functionPtr(PolynomialFunction, 1, NAME, TYPE);
    %functionPtr(PolynomialFunction, 2, NAME, TYPE);
%enddef

%define %defineTemplates(NAME, TYPE)
    // Must be called AFTER %include
    %baseFunction(NAME, TYPE);
    %baseFunctionN(1, NAME, TYPE);
    %baseFunctionN(2, NAME, TYPE);

    %function(Chebyshev1Function, 1, NAME, TYPE);
    %function(DoubleGaussianFunction, 2, NAME, TYPE);
    %function(GaussianFunction, 1, NAME, TYPE);
    %function(GaussianFunction, 2, NAME, TYPE);
    %function(IntegerDeltaFunction, 2, NAME, TYPE);
    %function(LanczosFunction, 1, NAME, TYPE);
    %function(LanczosFunction, 2, NAME, TYPE);
    %function(NullFunction, 1, NAME, TYPE);
    %function(NullFunction, 2, NAME, TYPE);
    %function(PolynomialFunction, 1, NAME, TYPE);
    %function(PolynomialFunction, 2, NAME, TYPE);
%enddef

/************************************************************************************************************/

%definePointers(D, double);
%definePointers(F, float);

%include "lsst/afw/math/Function.h"
%include "lsst/afw/math/FunctionLibrary.h"

%defineTemplates(D, double)
%defineTemplates(F, float)
