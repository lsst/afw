%module(package="gptest") gptest

%{
#include "lsst/afw/math/GaussianProcess.h"
#include "lsst/afw/math/detail/GaussianProcessFunctions.h"
//#include "/Users/noldor/physics/lsststackW2013/garage/gptest/include/gptest/gptest.h"
//#include "/Users/noldor/physics/lsststackW2013/garage/gptest/include/gptest/detail/GaussianProcessFunctions.h"
%}

// Enable ndarray's NumPy typemaps; types are declared in %included files.
%{
#define PY_ARRAY_UNIQUE_SYMBOL LSST_AFW_MATH_NUMPY_ARRAY_API
#include "numpy/arrayobject.h"
#include "ndarray/swig.h"
#include "ndarray/swig/eigen.h"
%}
%init %{
    import_array();
%}
%include "ndarray.i"

%declareNumPyConverters(ndarray::Array<double,2,2>);
%declareNumPyConverters(ndarray::Array<double,1,1>);
%declareNumPyConverters(ndarray::Array<int,1,1>);
%declareNumPyConverters(ndarray::Array<int,2,2>);

%define %declareGP(TYPE,SUFFIX)
%template(GaussianProcess##SUFFIX) lsst::afw::math::GaussianProcess<TYPE>;
//%template(GaussianProcess##SUFFIX) gptest::GaussianProcess<TYPE>;
%enddef

%include "lsst/afw/math/GaussianProcess.h"
%include "lsst/afw/math/detail/GaussianProcessFunctions.h"
//%include "/Users/noldor/physics/lsststackW2013/garage/gptest/include/gptest/gptest.h"
//%include "/Users/noldor/physics/lsststackW2013/garage/gptest/include/gptest/detail/GaussianProcessFunctions.h"


%declareGP(double,D);
