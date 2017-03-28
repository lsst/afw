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

#ifndef LSST_AFW_FORMATTERS_PROPERTYLISTFORMATTER_H
#define LSST_AFW_FORMATTERS_PROPERTYLISTFORMATTER_H

/**
 * @brief Interface for PropertyListFormatter class
 *
 * @class lsst::afw::formatters::PropertyListFormatter
 * @brief Class implementing persistence and retrieval for PropertyLists.
 *
 * @ingroup afw
 */

#include "lsst/daf/base.h"
#include "lsst/daf/persistence.h"
#include "lsst/pex/policy/Policy.h"

namespace lsst {
namespace afw {
namespace formatters {

class PropertyListFormatter : public daf::persistence::Formatter {
public:
    virtual ~PropertyListFormatter() {}

    virtual void write(
        daf::base::Persistable const* persistable,
        daf::persistence::Storage::Ptr storage,
        daf::base::PropertySet::Ptr additionalData
    );

    virtual daf::base::Persistable* read(
        daf::persistence::Storage::Ptr storage,
        daf::base::PropertySet::Ptr additionalData
    );

    virtual void update(
        daf::base::Persistable* persistable,
        daf::persistence::Storage::Ptr storage,
        daf::base::PropertySet::Ptr additionalData
    );

    static daf::persistence::Formatter::Ptr createInstance(
        pex::policy::Policy::Ptr policy
    );

    template <class Archive>
    static void delegateSerialize(
        Archive& ar,
        int const version,
        daf::base::Persistable* persistable
    );

private:
    explicit PropertyListFormatter(PTR(pex::policy::Policy) policy);

    static daf::persistence::FormatterRegistration registration;
};

}}} // namespace lsst::afw::formatters

#endif
