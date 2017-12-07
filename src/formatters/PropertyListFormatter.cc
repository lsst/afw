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

/*
 * Implementation of PropertyListFormatter class
 */
#include <cstdint>
#include <memory>
#include <string>

#include "lsst/daf/base.h"
#include "lsst/daf/base/PropertyList.h"
#include "lsst/daf/persistence.h"
#include "lsst/log/Log.h"
#include "lsst/afw/formatters/PropertyListFormatter.h"
#include "lsst/afw/fits.h"

namespace {
LOG_LOGGER _log = LOG_GET("afw.PropertyListFormatter");
}

using lsst::daf::persistence::FormatterStorage;

namespace lsst {
namespace afw {
namespace formatters {

lsst::daf::persistence::FormatterRegistration PropertyListFormatter::registration(
        "PropertyList", typeid(lsst::daf::base::PropertyList), createInstance);

PropertyListFormatter::PropertyListFormatter(std::shared_ptr<lsst::pex::policy::Policy>)
        : lsst::daf::persistence::Formatter(typeid(this)) {}

void PropertyListFormatter::write(lsst::daf::base::Persistable const* persistable,
                                  std::shared_ptr<lsst::daf::persistence::FormatterStorage> storage,
                                  std::shared_ptr<lsst::daf::base::PropertySet>) {
    LOGL_DEBUG(_log, "PropertyListFormatter write start");
    auto ip = dynamic_cast<lsst::daf::base::PropertyList const*>(persistable);
    if (ip == nullptr) {
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "Persisting non-PropertyList");
    }
    // TODO: Replace this with something better in DM-10776
    auto & actualStorage = *storage;
    if (typeid(actualStorage) == typeid(lsst::daf::persistence::FitsStorage)) {
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                          "FitsStorage for PropertyList read-only (writing is not supported)");
    }
    throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "Unrecognized FormatterStorage for PropertyList");
}

namespace {
std::unique_ptr<daf::base::PropertyList> readMetadataAsUniquePtr(std::string const& fileName, int hdu,
                                                                 bool strip) {
    auto metadata = std::make_unique<daf::base::PropertyList>();

    auto inheritedMetadata = fits::readMetadata(fileName, hdu, strip);
    metadata->combine(inheritedMetadata);

    return metadata;
}
}

lsst::daf::base::Persistable* PropertyListFormatter::read(
        std::shared_ptr<lsst::daf::persistence::FormatterStorage> storage,
        std::shared_ptr<lsst::daf::base::PropertySet>) {
    LOGL_DEBUG(_log, "PropertyListFormatter read start");
    // TODO: Replace this with something better in DM-10776
    auto fits = std::dynamic_pointer_cast<lsst::daf::persistence::FitsStorage>(storage);
    if (fits) {
        LOGL_DEBUG(_log, "PropertyListFormatter read FitsStorage");

        auto ip = readMetadataAsUniquePtr(fits->getPath(), fits->getHdu(), false);

        LOGL_DEBUG(_log, "PropertyListFormatter read end");
        return ip.release();
    }
    throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "Unrecognized FormatterStorage for PropertyList");
}

void PropertyListFormatter::update(lsst::daf::base::Persistable*,
                                   std::shared_ptr<lsst::daf::persistence::FormatterStorage>,
                                   std::shared_ptr<lsst::daf::base::PropertySet>) {
    throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "Unexpected call to update for PropertyList");
}

template <class Archive>
void PropertyListFormatter::delegateSerialize(Archive&, int const,
                                              lsst::daf::base::Persistable* persistable) {
    LOGL_DEBUG(_log, "PropertyListFormatter delegateSerialize start");
    auto ip = dynamic_cast<lsst::daf::base::PropertyList*>(persistable);
    if (ip == nullptr) {
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "Serializing non-PropertyList");
    }
    throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "PropertyList serialization not yet implemented");
}

std::shared_ptr<lsst::daf::persistence::Formatter> PropertyListFormatter::createInstance(
        std::shared_ptr<lsst::pex::policy::Policy> policy) {
    return std::shared_ptr<lsst::daf::persistence::Formatter>(new PropertyListFormatter(policy));
}
}
}
}  // namespace lsst::afw::formatters
