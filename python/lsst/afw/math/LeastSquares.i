%{
#include "lsst/afw/math/LeastSquares.h"
%}

%declareNumPyConverters(ndarray::Array<double,1,1>);
%declareNumPyConverters(ndarray::Array<double,2,2>);
%declareNumPyConverters(ndarray::Array<double,1,0>);
%declareNumPyConverters(ndarray::Array<double,2,0>);

%declareNumPyConverters(ndarray::Array<double const,1,1>);
%declareNumPyConverters(ndarray::Array<double const,2,2>);
%declareNumPyConverters(ndarray::Array<double const,1,0>);
%declareNumPyConverters(ndarray::Array<double const,2,0>);

%include "lsst/afw/math/LeastSquares.h"

%template(fromDesignMatrix) lsst::afw::math::LeastSquares::fromDesignMatrix<double,double,0,0>;
%template(setDesignMatrix) lsst::afw::math::LeastSquares::setDesignMatrix<double,double,0,0>;
%template(fromNormalEquations) lsst::afw::math::LeastSquares::fromNormalEquations<double,double,0,0>;
%template(setNormalEquations) lsst::afw::math::LeastSquares::setNormalEquations<double,double,0,0>;
