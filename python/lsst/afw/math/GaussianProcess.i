
%{
#include "lsst/afw/math/GaussianProcess.h"
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


%define %declarePTR(TYPE)
%shared_ptr(lsst::afw::math::Covariogram<TYPE>);
%shared_ptr(lsst::afw::math::SquaredExpCovariogram<TYPE>);
%shared_ptr(lsst::afw::math::NeuralNetCovariogram<TYPE>);
%enddef

%declareNumPyConverters(ndarray::Array<double,2,2>);
%declareNumPyConverters(ndarray::Array<double,1,1>);
%declareNumPyConverters(ndarray::Array<int,1,1>);
%declareNumPyConverters(ndarray::Array<int,2,2>);

%define %declareGP(TYPE,SUFFIX)
%template(KdTree##SUFFIX) lsst::afw::math::KdTree<TYPE>;
%template(GaussianProcess##SUFFIX) lsst::afw::math::GaussianProcess<TYPE>;
%template(Covariogram##SUFFIX) lsst::afw::math::Covariogram<TYPE>;
%template(SquaredExpCovariogram##SUFFIX) lsst::afw::math::SquaredExpCovariogram<TYPE>;
%template(NeuralNetCovariogram##SUFFIX) lsst::afw::math::NeuralNetCovariogram<TYPE>;
%enddef

%declarePTR(double);

%include "lsst/afw/math/GaussianProcess.h"

%declareGP(double,D);

