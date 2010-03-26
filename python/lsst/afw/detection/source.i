// -*- lsst-c++ -*-

%{
#include <sstream>
#include "lsst/afw/formatters/Utils.h"
#include "lsst/afw/detection/BaseSourceAttributes.h"
#include "lsst/afw/detection/Source.h"
#include "lsst/afw/detection/DiaSource.h"
#include "lsst/afw/detection/Astrometry.h"
#include "lsst/afw/detection/Photometry.h"
%}

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

/************************************************************************************************************/
/*
 * Must go before the includes
 */
%define %MeasurementBefore(WHAT)
SWIG_SHARED_PTR(Measurement##WHAT, lsst::afw::detection::Measurement<lsst::afw::detection::WHAT>);
SWIG_SHARED_PTR_DERIVED(WHAT,
                        lsst::afw::detection::Measurement<lsst::afw::detection::WHAT>,
                        lsst::afw::detection::WHAT);

%template(WHAT##Set) std::vector<boost::shared_ptr<lsst::afw::detection::WHAT> >;
%enddef
/*
 * Must go after the includes
 */
%define %MeasurementAfter(WHAT)
%template(Measurement##WHAT)
    lsst::afw::detection::Measurement<lsst::afw::detection::WHAT>;
%definePythonIterator(lsst::afw::detection::Measurement<lsst::afw::detection::WHAT>);
%enddef

/************************************************************************************************************/

SWIG_SHARED_PTR(Schema, lsst::afw::detection::Schema);
SWIG_SHARED_PTR_DERIVED(SchemaEntry, 
                        lsst::afw::detection::Schema,
                        lsst::afw::detection::SchemaEntry);

%template(SchemaVector) std::vector<boost::shared_ptr<lsst::afw::detection::Schema> >;
%definePythonIterator(lsst::afw::detection::Schema);

%MeasurementBefore(Astrometry);
%MeasurementBefore(Photometry);

%include "lsst/afw/detection/Schema.h"
%include "lsst/afw/detection/Measurement.h"

%MeasurementAfter(Astrometry);
%MeasurementAfter(Photometry);

%include "lsst/afw/detection/Astrometry.h"
%include "lsst/afw/detection/Photometry.h"

/************************************************************************************************************/

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

