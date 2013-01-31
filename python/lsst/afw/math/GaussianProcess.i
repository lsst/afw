%{
#include "lsst/afw/math/GaussianProcess.h"
%}

%define %declareGP(INTYPE,OUTTYPE,INSUFFIX,OUTSUFFIX)
%template(gaussianprocess##INSUFFIX##OUTSUFFIX) lsst::afw::math::gaussianprocess<INTYPE,OUTTYPE>;
%enddef

%define %declareKD(TYPE,SUFFIX)
%template(kd##SUFFIX) lsst::afw::math::kd<TYPE>
%enddef

%include "lsst/afw/math/GaussianProcess.h"

%declareGP(double, double, D, D);
%declareKD(double, D);
