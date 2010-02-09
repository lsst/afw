// -*- lsst-c++ -*-
//

%{
#include "lsst/afw/image/Slice.h"
%}

%include "lsst/afw/image/Slice.h"

%define %declareSlice(TYPE,SUFFIX)
SWIG_SHARED_PTR_DERIVED(Slice##SUFFIX, lsst::afw::image::Image<TYPE>, lsst::afw::image::Slice<TYPE>);
SWIG_SHARED_PTR(Slice##SUFFIX, lsst::afw::image::Slice<TYPE>);
%template(Slice##SUFFIX) lsst::afw::image::Slice<TYPE>;
%enddef

%declareSlice(double, D)
%declareSlice(float, F)
 //%declareSlice(int, I)
