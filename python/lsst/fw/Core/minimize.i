
%{
#include "lsst/fw/minimize.h"
%}

%include "Minuit/FCNBase.h"
%include "lsst/fw/Function.h"
%include "lsst/fw/FunctionLibrary.h"
%include "lsst/fw/minimize.h"

%template(pairDD) std::pair<double,double>;
%template(vectorPairDD) std::vector<std::pair<double,double> >;

//%template(minimize)             lsst::fw::function::minimize<float>;
%template(minimize)             lsst::fw::function::minimize<double>;
