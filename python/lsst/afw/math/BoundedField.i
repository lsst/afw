%{
#include "lsst/afw/math/BoundedField.h"
%}

%include "lsst/pex/config.h"
%import "lsst/afw/table/io/ioLib.i"

%declareNumPyConverters(ndarray::Array<double,1,1>);
%declareNumPyConverters(ndarray::Array<double const,1>);

%declareTablePersistable(BoundedField, lsst::afw::math::BoundedField)

%include "lsst/afw/math/BoundedField.h"

%define %instantiateBoundedField(T)
%template(fillImage) lsst::afw::math::BoundedField::fillImage<T>;
%template(fillImage) lsst::afw::math::BoundedField::addToImage<T>;
%template(fillImage) lsst::afw::math::BoundedField::multiplyImage<T>;
%template(fillImage) lsst::afw::math::BoundedField::divideImage<T>;
%enddef

%instantiateBoundedField(float)
%instantiateBoundedField(double)

%usePointerEquality(lsst::afw::math::BoundedField)
