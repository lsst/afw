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


/** @file
 * @brief Implementation of PropertyListFormatter class
 *
 * @ingroup afw
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

namespace lsst {
namespace afw {
namespace formatters {

lsst::daf::persistence::FormatterRegistration PropertyListFormatter::registration(
    "PropertyList", typeid(lsst::daf::base::PropertyList), createInstance);

PropertyListFormatter::PropertyListFormatter(
    lsst::pex::policy::Policy::Ptr
)
    : lsst::daf::persistence::Formatter(typeid(this))
{
}

void PropertyListFormatter::write(
    lsst::daf::base::Persistable const* persistable,
    lsst::daf::persistence::Storage::Ptr storage,
    lsst::daf::base::PropertySet::Ptr
)
{
    LOGL_DEBUG(_log, "PropertyListFormatter write start");
    auto ip = dynamic_cast<lsst::daf::base::PropertyList const*>(persistable);
    if (ip == 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "Persisting non-PropertyList");
    }
    if (typeid(*storage) == typeid(lsst::daf::persistence::FitsStorage)) {
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                      "FitsStorage for PropertyList read-only (writing is not supported)");
    }
    throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                      "Unrecognized Storage for PropertyList");
}

namespace {
    // I'd use a unique_ptr, except that PropertyList won't allow me to merge non-shared_ptrs
    std::unique_ptr<daf::base::PropertyList>
    readMetadataAsUniquePtr(std::string const & fileName, int hdu, bool strip)
    {
        auto metadata = std::unique_ptr<daf::base::PropertyList>(new lsst::daf::base::PropertyList);
        //
        // We need a shared_ptr to be able to call PropertyList.combine()
        //
        auto inheritedMetadata = std::shared_ptr<daf::base::PropertyList>(new daf::base::PropertyList);

        fits::Fits fitsfile(fileName, "r", fits::Fits::AUTO_CLOSE | fits::Fits::AUTO_CHECK);
        fitsfile.setHdu(hdu);
        fitsfile.readMetadata(*inheritedMetadata, strip);
        metadata->combine(inheritedMetadata);

        return metadata;
    }
}

lsst::daf::base::Persistable* PropertyListFormatter::read(
    lsst::daf::persistence::Storage::Ptr storage,
    lsst::daf::base::PropertySet::Ptr
)
{
    LOGL_DEBUG(_log, "PropertyListFormatter read start");
    if(typeid(*storage) == typeid(lsst::daf::persistence::FitsStorage)) {

        LOGL_DEBUG(_log, "PropertyListFormatter read FitsStorage");

        auto fits = dynamic_cast<lsst::daf::persistence::FitsStorage*>(storage.get());
        auto ip = readMetadataAsUniquePtr(fits->getPath(), fits->getHdu(), false);

        LOGL_DEBUG(_log, "PropertyListFormatter read end");
        return ip.release();
    }
    throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                      "Unrecognized Storage for PropertyList");
}

void PropertyListFormatter::update(
    lsst::daf::base::Persistable*,
    lsst::daf::persistence::Storage::Ptr,
    lsst::daf::base::PropertySet::Ptr
)
{
    throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                      "Unexpected call to update for PropertyList");
}

template <class Archive>
void PropertyListFormatter::delegateSerialize(
    Archive&,
    int const,
    lsst::daf::base::Persistable* persistable
)
{
    LOGL_DEBUG(_log, "PropertyListFormatter delegateSerialize start");
    auto ip = dynamic_cast<lsst::daf::base::PropertyList*>(persistable);
    if (ip == 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "Serializing non-PropertyList");
    }
    throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                      "PropertyList serialization not yet implemented");
}

lsst::daf::persistence::Formatter::Ptr PropertyListFormatter::createInstance(
    lsst::pex::policy::Policy::Ptr policy
)
{
    return lsst::daf::persistence::Formatter::Ptr(new PropertyListFormatter(policy));
}

}}} // namespace lsst::afw::formatters
