%{
#include "lsst/afw/math/BoundedField.h"
#include "lsst/afw/math/ChebyshevBoundedField.h"
%}

%include "lsst/pex/config.h"
%include "lsst/afw/math/BoundedField.i"
%import "lsst/afw/table/io/ioLib.i"

%declareTablePersistable(ChebyshevBoundedField, lsst::afw::math::ChebyshevBoundedField)

%include "lsst/afw/math/ChebyshevBoundedField.h"

%template(fit) lsst::afw::math::ChebyshevBoundedField::fit<float>;
%template(fit) lsst::afw::math::ChebyshevBoundedField::fit<double>;

%castShared(lsst::afw::math::ChebyshevBoundedField, lsst::afw::math::BoundedField)

%pythoncode %{
import lsst.pex.config

@lsst.pex.config.wrap(ChebyshevBoundedFieldControl)
class ChebyshevBoundedFieldConfig(lsst.pex.config.Config):

    def computeSize(self):
        return self.makeControl().computeSize()

ChebyshevBoundedField.Control = ChebyshevBoundedFieldControl
ChebyshevBoundedField.ConfigClass = ChebyshevBoundedFieldConfig
%}
