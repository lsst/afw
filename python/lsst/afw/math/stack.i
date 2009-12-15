// -*- lsst-c++ -*-

%{
#include "lsst/afw/math/Stack.h"
%}

%include "lsst/afw/math/Stack.h"

%define %declareStacks(PIXTYPE)
%template(statisticsStack) lsst::afw::math::statisticsStack<lsst::afw::image::Image<PIXTYPE> >;
%enddef

%declareStacks(float)
%declareStacks(double)
