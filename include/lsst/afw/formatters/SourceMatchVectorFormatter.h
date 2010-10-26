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
//! \brief  Formatter subclasses for SourceMatchVector
//
//##====----------------                                ----------------====##/

#ifndef LSST_AFW_FORMATTERS_SOURCE_MATCH_VECTOR_FORMATTER_H
#define LSST_AFW_FORMATTERS_SOURCE_MATCH_VECTOR_FORMATTER_H

#include <string>
#include <vector>

#include "lsst/daf/base.h"
#include "lsst/daf/persistence.h"
#include "lsst/afw/detection/SourceMatch.h"

namespace lsst {
namespace afw {
namespace formatters {

class SourceMatchVectorFormatter : public lsst::daf::persistence::Formatter {
public:

    virtual ~SourceMatchVectorFormatter();

    virtual void write(
        lsst::daf::base::Persistable const *,
        lsst::daf::persistence::Storage::Ptr,
        lsst::daf::base::PropertySet::Ptr
    );
    virtual lsst::daf::base::Persistable* read(
        lsst::daf::persistence::Storage::Ptr,
        lsst::daf::base::PropertySet::Ptr
    );
    /*
     virtual void update(
     lsst::daf::base::Persistable*,
     lsst::daf::persistence::Storage::Ptr,
     lsst::daf::base::PropertySet::Ptr
     );
     */

    /*
    template <class Archive>
    static void delegateSerialize(
        Archive &,
        unsigned int const,
        lsst::daf::base::Persistable *
    );
     */

    lsst::pex::policy::Policy::Ptr _policy;

    explicit SourceMatchVectorFormatter(lsst::pex::policy::Policy::Ptr const & policy);

    static lsst::daf::persistence::Formatter::Ptr createInstance(
        lsst::pex::policy::Policy::Ptr
    );
    static lsst::daf::persistence::FormatterRegistration registration;

};


}}} // namespace lsst::afw::formatters

#endif

