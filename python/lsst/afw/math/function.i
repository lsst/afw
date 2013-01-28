// -*- lsst-c++ -*-

/* 
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
 * 
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the LSST License Statement and 
 * the GNU General Public License along with this program.  If not, 
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */

%import "lsst/daf/base/baseLib.i" // for Citizen
 
%{
#include <vector>

#include "boost/shared_ptr.hpp"

#include "lsst/afw/math/Function.h"
#include "lsst/afw/math/FunctionLibrary.h"
%}

%import "lsst/afw/table/io/Persistable.i"

// Must be used before %include
%define %baseFunctionPtrs(TYPE, CTYPE)
%declareTablePersistable(Function##TYPE, lsst::afw::math::Function<CTYPE>);
%declareTablePersistable(Function1##TYPE, lsst::afw::math::Function1<CTYPE>);
%declareTablePersistable(Function2##TYPE, lsst::afw::math::Function2<CTYPE>);
%shared_ptr(lsst::afw::math::BasePolynomialFunction2<CTYPE>);
%enddef

%define %functionPtr(NAME, N, CTYPE)
%shared_ptr(lsst::afw::math::NAME##N<CTYPE>);
%enddef

// Must be used after %include
%define %baseFunctions(TYPE, CTYPE)
%template(Function##TYPE) lsst::afw::math::Function<CTYPE>;
%template(Function1##TYPE) lsst::afw::math::Function1<CTYPE>;
%template(Function2##TYPE) lsst::afw::math::Function2<CTYPE>;
%template(BasePolynomialFunction2##TYPE) lsst::afw::math::BasePolynomialFunction2<CTYPE>;
%enddef

%define %function(NAME, N, TYPE, CTYPE)
%template(NAME##N##TYPE) lsst::afw::math::NAME##N<CTYPE>;
%enddef
//
// Macros to define float or double versions of things
//
%define %definePointers(TYPE, CTYPE)
    // Must be called BEFORE %include
    %baseFunctionPtrs(TYPE, CTYPE);

    %functionPtr(Chebyshev1Function, 1, CTYPE);
    %functionPtr(Chebyshev1Function, 2, CTYPE);
    %functionPtr(DoubleGaussianFunction, 2, CTYPE);
    %functionPtr(GaussianFunction, 1, CTYPE);
    %functionPtr(GaussianFunction, 2, CTYPE);
    %functionPtr(IntegerDeltaFunction, 2, CTYPE);
    %functionPtr(LanczosFunction, 1, CTYPE);
    %functionPtr(LanczosFunction, 2, CTYPE);
    %functionPtr(NullFunction, 1, CTYPE);
    %functionPtr(NullFunction, 2, CTYPE);
    %functionPtr(PolynomialFunction, 1, CTYPE);
    %functionPtr(PolynomialFunction, 2, CTYPE);
%enddef

%define %defineTemplates(TYPE, CTYPE)
    // Must be called AFTER %include
    %baseFunctions(TYPE, CTYPE);

    %function(Chebyshev1Function, 1, TYPE, CTYPE);
    %function(Chebyshev1Function, 2, TYPE, CTYPE);
    %function(DoubleGaussianFunction, 2, TYPE, CTYPE);
    %function(GaussianFunction, 1, TYPE, CTYPE);
    %function(GaussianFunction, 2, TYPE, CTYPE);
    %function(IntegerDeltaFunction, 2, TYPE, CTYPE);
    %function(LanczosFunction, 1, TYPE, CTYPE);
    %function(LanczosFunction, 2, TYPE, CTYPE);
    %function(NullFunction, 1, TYPE, CTYPE);
    %function(NullFunction, 2, TYPE, CTYPE);
    %function(PolynomialFunction, 1, TYPE, CTYPE);
    %function(PolynomialFunction, 2, TYPE, CTYPE);
%enddef

/************************************************************************************************************/

%definePointers(D, double);
%definePointers(F, float);

%include "lsst/afw/math/Function.h"
%include "lsst/afw/math/FunctionLibrary.h"

%defineTemplates(D, double)
%defineTemplates(F, float)
