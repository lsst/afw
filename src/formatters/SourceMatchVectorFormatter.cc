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
 
//
//##====----------------                                ----------------====##/
//
//! \file
//! \brief  Implementation of persistence for PersistableSourceMatchVector instances
//
//##====----------------                                ----------------====##/

#include <memory>

#include "boost/format.hpp"

#include "lsst/daf/base.h"
#include "lsst/daf/persistence.h"
#include "lsst/pex/exceptions.h"
#include "lsst/pex/policy/Policy.h"
#include "lsst/afw/formatters/Utils.h"
#include "lsst/afw/detection/SourceMatch.h"
#include "lsst/afw/formatters/SourceMatchVectorFormatter.h"

namespace ex = lsst::pex::exceptions;
namespace det = lsst::afw::detection;

using lsst::daf::base::Persistable;
using lsst::daf::persistence::FitsStorage;
using lsst::daf::persistence::Storage;
using lsst::pex::policy::Policy;
using lsst::afw::detection::SourceMatch;
using lsst::afw::detection::PersistableSourceMatchVector;
using lsst::afw::image::Filter;

namespace form = lsst::afw::formatters;

// -- SourceMatchVectorFormatter ----------------

form::SourceMatchVectorFormatter::SourceMatchVectorFormatter(Policy::Ptr const & policy) :
    lsst::daf::persistence::Formatter(typeid(this)),
    _policy(policy)
{}


form::SourceMatchVectorFormatter::~SourceMatchVectorFormatter() {}


lsst::daf::persistence::Formatter::Ptr form::SourceMatchVectorFormatter::createInstance(Policy::Ptr policy) {
    return lsst::daf::persistence::Formatter::Ptr(new SourceMatchVectorFormatter(policy));
}


lsst::daf::persistence::FormatterRegistration form::SourceMatchVectorFormatter::registration(
    "PersistableSourceMatchVector",
    typeid(PersistableSourceMatchVector),
    createInstance
);

/** 
 * Persist a PersistableSourceMatchVector to FitsStorage
 */
void form::SourceMatchVectorFormatter::write(
    Persistable const * persistable,
    Storage::Ptr storage,
    lsst::daf::base::PropertySet::Ptr additionalData
) {
    if (!persistable)
        throw LSST_EXCEPT(ex::InvalidParameterException, "No Persistable provided");
    if (!storage)
        throw LSST_EXCEPT(ex::InvalidParameterException, "No Storage provided");

    PersistableSourceMatchVector const * p = dynamic_cast<PersistableSourceMatchVector const *>(persistable);
    if (!p)
        throw LSST_EXCEPT(ex::RuntimeErrorException,
                          "Persistable was not of concrete type SourceMatchVector");

    if (typeid(*storage) == typeid(FitsStorage)) {
        FitsStorage* bs = dynamic_cast<FitsStorage *>(storage.get());

        printf("SourceMatchVectorFormatter: persisting to path \"%s\"\n",
               bs->getPath().c_str());
        
    } else {
        throw LSST_EXCEPT(ex::InvalidParameterException, 
                          "Storage type is not supported"); 
    }
}


/** 
 * Retrieve a PersistableSourceMatchVector from FitsStorage.
 */
Persistable* form::SourceMatchVectorFormatter::read(
    Storage::Ptr storage,
    lsst::daf::base::PropertySet::Ptr additionalData
) {   
    std::auto_ptr<PersistableSourceMatchVector> p(new PersistableSourceMatchVector);
    
    if (typeid(*storage) == typeid(FitsStorage)) {
    } else {
        throw LSST_EXCEPT(ex::InvalidParameterException, 
                          "Storage type is not supported");
    }
    return p.release();
}

void form::SourceMatchVectorFormatter::update(Persistable*, 
                                              Storage::Ptr, lsst::daf::base::PropertySet::Ptr
                                              ) {
    throw LSST_EXCEPT(ex::RuntimeErrorException, 
                      "SourceMatchVectorFormatter: updates not supported");
}

