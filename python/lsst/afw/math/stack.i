// -*- lsst-c++ -*-

%{
#include "lsst/afw/math/Stack.h"
%}

%include "lsst/afw/math/Stack.h"

%define %declareStacks(PIXTYPE)
%template(statisticsStack) lsst::afw::math::statisticsStack<PIXTYPE>;
%template(sliceOperate) lsst::afw::math::sliceOperate<PIXTYPE>;
%enddef

%declareStacks(float)
%declareStacks(double)
