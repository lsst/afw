// -*- lsst-c++ -*-

%{
#include "lsst/afw/detection/Source.h"
#include "lsst/afw/detection/DiaSource.h"
#include "lsst/afw/formatters/Utils.h"
#include <sstream>
%}

%rename(SourceVec)  lsst::afw::detection::SourceVector;

SWIG_SHARED_PTR(Source, lsst::afw::detection::Source);
SWIG_SHARED_PTR_DERIVED(SourceVec, lsst::daf::base::Persistable, lsst::afw::detection::SourceVector);

%include "lsst/afw/detection/Source.h"
%include "lsst/afw/detection/DiaSource.h"
%include "lsst/afw/formatters/Utils.h"

// Provide semi-useful printing of catalog records
%extend lsst::afw::detection::Source {
    std::string toString() {
        std::ostringstream os;
        os << "Source " << $self->getId();
        os.precision(9);
        os << " (" << $self->getRa() << ", " << $self->getDec() << ")";
        return os.str();
    }
};

%pythoncode %{
Source.__str__ = Source.toString
%}


%lsst_persistable(lsst::afw::detection::PersistableSourceVector);

