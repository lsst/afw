// -*- lsst-c++ -*-

/* 
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
 * 
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the LSST License Statement and 
 * the GNU General Public License along with this program.  If not, 
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */
 

%{
#include <sstream>
#include "lsst/afw/formatters/Utils.h"
#include "lsst/afw/detection/BaseSourceAttributes.h"
#include "lsst/afw/detection/Source.h"
#include "lsst/afw/detection/DiaSource.h"
#include "lsst/afw/detection/Astrometry.h"
#include "lsst/afw/detection/Photometry.h"
#include "lsst/afw/detection/Shape.h"

#include "lsst/afw/formatters/SourceFormatter.h"
#include "lsst/afw/formatters/DiaSourceFormatter.h"
#include "boost/pointer_cast.hpp"
#include "boost/shared_ptr.hpp"
%}

%include "../boost_picklable.i"

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
%MeasurementBefore(Shape);

%include "lsst/afw/detection/Schema.h"
%include "lsst/afw/detection/Measurement.h"

%MeasurementAfter(Astrometry);
%MeasurementAfter(Photometry);
%MeasurementAfter(Shape);

%include "lsst/afw/detection/Astrometry.h"
%include "lsst/afw/detection/Photometry.h"
%include "lsst/afw/detection/Shape.h"

/************************************************************************************************************/

//Explicit instantiation of BaseSourceAttributes
%template(SourceBase)
    lsst::afw::detection::BaseSourceAttributes<lsst::afw::detection::NUM_SOURCE_NULLABLE_FIELDS>;
%template(DiaSourceBase)
    lsst::afw::detection::BaseSourceAttributes<lsst::afw::detection::NUM_DIASOURCE_NULLABLE_FIELDS>;

%extend lsst::afw::detection::Source {
    std::string toString() {
        std::ostringstream os;
        os << "Source " << $self->getId();
        os.precision(9);
        os << " (" << $self->getRa() << ", " << $self->getDec() << ")";
        return os.str();
    }

    %feature("shadow") _getFootprint %{
        def getFootprint(self):
            return $action(self)
    %}
    %feature("shadow") _setFootprint %{
        def setFootprint(self, fp):
            $action(self, fp)
    %}

    void _setFootprint(boost::shared_ptr<lsst::afw::detection::Footprint> const & fp) {
        boost::shared_ptr<lsst::afw::detection::Footprint const> constFp(fp);
        $self->setFootprint(constFp); 
    }
    boost::shared_ptr<lsst::afw::detection::Footprint> _getFootprint() {
        return boost::const_pointer_cast<lsst::afw::detection::Footprint>($self->getFootprint());
    }
};


%include "lsst/afw/detection/Source.h"
%include "lsst/afw/detection/DiaSource.h"

%boost_picklable(lsst::afw::detection::Source)
%boost_picklable(lsst::afw::detection::DiaSource)

//Explicit STL container instantiation
%template(SourceSet) std::vector<lsst::afw::detection::Source::Ptr>;
%template(DiaSourceSet)   std::vector<lsst::afw::detection::DiaSource::Ptr>;

// Provide semi-useful printing of catalog records


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
%boost_picklable(lsst::afw::detection::PersistableSourceVector);
%boost_picklable(lsst::afw::detection::PersistableDiaSourceVector);

%template(PersistableSourceVectorVector) std::vector<lsst::afw::detection::PersistableSourceVector::Ptr>;
