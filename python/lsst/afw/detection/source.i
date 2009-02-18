// -*- lsst-c++ -*-

%{
#include "lsst/afw/formatters/Utils.h"
#include "lsst/afw/detection/BaseSourceAttributes.h"
#include "lsst/afw/detection/Source.h"
#include "lsst/afw/detection/DiaSource.h"
#include <sstream>
%}

//Explicit STL container instantiation
//%template(SourceSet) std::vector<lsst::afw::detection::Source::Ptr>;
//%template(DiaSourceSet)   std::vector<lsst::afw::detection::DiaSource::Ptr>;

//shared_ptr declarations
SWIG_SHARED_PTR(SourceBase, lsst::afw::detection::BaseSourceAttributes<lsst::afw::detection::NUM_SOURCE_NULLABLE_FIELDS>);
SWIG_SHARED_PTR(DiaSourceBase, lsst::afw::detection::BaseSourceAttributes<lsst::afw::detection::NUM_DIASOURCE_NULLABLE_FIELDS>); 
 
SWIG_SHARED_PTR_DERIVED(SourceP, 
    lsst::afw::detection::BaseSourceAttributes<lsst::afw::detection::NUM_SOURCE_NULLABLE_FIELDS>, 
    lsst::afw::detection::Source);
SWIG_SHARED_PTR_DERIVED(DiaSourceP,
    lsst::afw::detection::BaseSourceAttributes<lsst::afw::detection::NUM_DIASOURCE_NULLABLE_FIELDS>,
    lsst::afw::detection::DiaSource);  

SWIG_SHARED_PTR_DERIVED(PersistableSourceVector,
    lsst::daf::base::Persistable,
    lsst::afw::detection::PersistableSourceVector);
SWIG_SHARED_PTR_DERIVED(PersistableDiaSourceVector,
    lsst::daf::base::Persistable,
    lsst::afw::detection::PersistableDiaSourceVector);

%include "lsst/afw/formatters/Utils.h"
%include "lsst/afw/detection/BaseSourceAttributes.h"    

//Explicit instantiation of BaseSourceAttributes
%template(SourceBase)
    lsst::afw::detection::BaseSourceAttributes<lsst::afw::detection::NUM_SOURCE_NULLABLE_FIELDS>;
%template(DiaSourceBase)
    lsst::afw::detection::BaseSourceAttributes<lsst::afw::detection::NUM_DIASOURCE_NULLABLE_FIELDS>;

%include "lsst/afw/detection/Source.h"
%include "lsst/afw/detection/DiaSource.h"

//Explicit STL container instantiation
%template(SourceSet) std::vector<lsst::afw::detection::Source::Ptr>;
%template(DiaSourceSet)   std::vector<lsst::afw::detection::DiaSource::Ptr>;

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

%extend lsst::afw::detection::DiaSource {
    std::string toString() {
        std::ostringstream os;
        os << "DiaSource " << $self->getId();
        os.precision(9);
        os << " (" << $self->getRa() << ", " << $self->getDec() << ")";
        return os.str();
    }
};

%pythoncode %{
Source.__str__ = Source.toString
%}

%pythoncode %{
DiaSource.__str__ = DiaSource.toString
%}



%lsst_persistable(lsst::afw::detection::PersistableSourceVector);
%lsst_persistable(lsst::afw::detection::PersistableDiaSourceVector);
