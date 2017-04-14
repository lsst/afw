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

#ifndef LSST_AFW_FORMATTERS_EXPOSUREFORMATTER_H
#define LSST_AFW_FORMATTERS_EXPOSUREFORMATTER_H

/*
 * Interface for ExposureFormatter class
 */


#include "lsst/daf/base.h"
#include "lsst/daf/persistence.h"
#include "lsst/pex/policy/Policy.h"

namespace lsst {
namespace afw {
namespace formatters {

/**
 * Class implementing persistence and retrieval for Exposures.
 */
template<typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
class ExposureFormatter : public lsst::daf::persistence::Formatter {
public:
    virtual ~ExposureFormatter(void);

    virtual void write(
        lsst::daf::base::Persistable const* persistable,
        std::shared_ptr<lsst::daf::persistence::Storage> storage,
        std::shared_ptr<lsst::daf::base::PropertySet> additionalData
    );
    virtual lsst::daf::base::Persistable* read(
        std::shared_ptr<lsst::daf::persistence::Storage> storage,
        std::shared_ptr<lsst::daf::base::PropertySet> additionalData
    );
    virtual void update(
        lsst::daf::base::Persistable* persistable,
        std::shared_ptr<lsst::daf::persistence::Storage> storage,
        std::shared_ptr<lsst::daf::base::PropertySet> additionalData
    );

    static std::shared_ptr<lsst::daf::persistence::Formatter> createInstance(
        std::shared_ptr<lsst::pex::policy::Policy> policy
    );

    template <class Archive>
    static void delegateSerialize(
        Archive& ar, unsigned int const version,
        lsst::daf::base::Persistable* persistable
    );
private:
    explicit ExposureFormatter(std::shared_ptr<lsst::pex::policy::Policy> policy);

    std::shared_ptr<lsst::pex::policy::Policy> _policy;

    static lsst::daf::persistence::FormatterRegistration registration;
};

}}} // namespace lsst::afw::formatters

#endif
