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

#include "lsst/daf/base.h"
#include "lsst/daf/persistence.h"
#include "lsst/pex/policy/Policy.h"

namespace lsst {
namespace afw {
namespace formatters {

/**
 * Class implementing persistence and retrieval for PropertyLists.
 */
class PropertyListFormatter : public daf::persistence::Formatter {
public:
    virtual ~PropertyListFormatter() = default;

    PropertyListFormatter(PropertyListFormatter const&);
    PropertyListFormatter(PropertyListFormatter&&);
    PropertyListFormatter& operator=(PropertyListFormatter const&);
    PropertyListFormatter& operator=(PropertyListFormatter&&);

    virtual void write(daf::base::Persistable const* persistable,
                       std::shared_ptr<daf::persistence::FormatterStorage> storage,
                       std::shared_ptr<daf::base::PropertySet> additionalData);

    virtual daf::base::Persistable* read(std::shared_ptr<daf::persistence::FormatterStorage> storage,
                                         std::shared_ptr<daf::base::PropertySet> additionalData);

    virtual void update(daf::base::Persistable* persistable,
                        std::shared_ptr<daf::persistence::FormatterStorage> storage,
                        std::shared_ptr<daf::base::PropertySet> additionalData);

    static std::shared_ptr<daf::persistence::Formatter> createInstance(
            std::shared_ptr<pex::policy::Policy> policy);

    template <class Archive>
    static void delegateSerialize(Archive& ar, int const version, daf::base::Persistable* persistable);

private:
    explicit PropertyListFormatter(std::shared_ptr<pex::policy::Policy> policy);

    static daf::persistence::FormatterRegistration registration;
};
}
}
}  // namespace lsst::afw::formatters

#endif
