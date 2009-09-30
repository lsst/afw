
%{
#include "lsst/afw/math/minimize.h"
%}

%import "Minuit/GenericFunction.h"
%include "Minuit/FCNBase.h"
%include "lsst/afw/math/Function.h"
%include "lsst/afw/math/FunctionLibrary.h"
%include "lsst/afw/math/minimize.h"

%template(pairDD) std::pair<double,double>;
%template(vectorPairDD) std::vector<std::pair<double,double> >;

//%template(minimize)             lsst::afw::math::minimize<float>;
%template(minimize)             lsst::afw::math::minimize<double>;
