%{
#include "lsst/afw/math/GaussianProcess.h"
%}

%define %declareGP(INTYPE,OUTTYPE,INSUFFIX,OUTSUFFIX)
%template(gaussianprocess##INSUFFIX##OUTSUFFIX) lsst::afw::math::gaussianprocess<INTYPE,OUTTYPE>;
%enddef

%include "lsst/afw/math/GaussianProcess.h"

%declareGP(double, double, D, D);
