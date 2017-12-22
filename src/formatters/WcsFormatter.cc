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
 * Implementation of WcsFormatter class
 */

#ifndef __GNUC__
#define __attribute__(x) /*NOTHING*/
#endif
static char const* SVNid __attribute__((unused)) = "$Id$";

#include <string>

// not used? #include <stdlib.h>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

//#include "boost/serialization/shared_ptr.hpp"
#include "wcslib/wcs.h"

#include "lsst/daf/base.h"
#include "lsst/daf/persistence.h"
#include "lsst/daf/persistence/PropertySetFormatter.h"
#include "lsst/log/Log.h"
#include "lsst/pex/exceptions.h"
#include "lsst/afw/formatters/ImageFormatter.h"
#include "lsst/afw/formatters/MaskedImageFormatter.h"
#include "lsst/afw/formatters/WcsFormatter.h"
#include "lsst/afw/image/Wcs.h"
#include "lsst/afw/fits.h"

namespace {
LOG_LOGGER _log = LOG_GET("afw.WcsFormatter");
}

namespace lsst {
namespace afw {
namespace formatters {

namespace dafBase = lsst::daf::base;
namespace dafPersist = lsst::daf::persistence;
namespace pexPolicy = lsst::pex::policy;
namespace pexExcept = lsst::pex::exceptions;

dafPersist::FormatterRegistration WcsFormatter::registration("Wcs", typeid(image::Wcs), createInstance);

WcsFormatter::WcsFormatter(std::shared_ptr<pexPolicy::Policy>) : dafPersist::Formatter(typeid(this)) {}

WcsFormatter::~WcsFormatter(void) {}

void WcsFormatter::write(dafBase::Persistable const* persistable,
                         std::shared_ptr<dafPersist::FormatterStorage> storage,
                         std::shared_ptr<dafBase::PropertySet>) {
    LOGL_DEBUG(_log, "WcsFormatter write start");
    image::Wcs const* ip = dynamic_cast<image::Wcs const*>(persistable);
    if (ip == 0) {
        throw LSST_EXCEPT(pexExcept::RuntimeError, "Persisting non-Wcs");
    }
    // TODO: Replace this with something better in DM-10776
    auto boost = std::dynamic_pointer_cast<dafPersist::BoostStorage>(storage);
    if (boost) {
        LOGL_DEBUG(_log, "WcsFormatter write BoostStorage");
        boost->getOArchive() & *ip;
        LOGL_DEBUG(_log, "WcsFormatter write end");
        return;
    }
    throw LSST_EXCEPT(pexExcept::RuntimeError, "Unrecognized FormatterStorage for Wcs");
}

dafBase::Persistable* WcsFormatter::read(std::shared_ptr<dafPersist::FormatterStorage> storage,
                                         std::shared_ptr<dafBase::PropertySet> additionalData) {
    LOGL_DEBUG(_log, "WcsFormatter read start");
    // TODO: Replace this with something better in DM-10776
    auto boost = std::dynamic_pointer_cast<dafPersist::BoostStorage>(storage);
    if (boost) {
        image::Wcs* ip = new image::Wcs;
        LOGL_DEBUG(_log, "WcsFormatter read BoostStorage");
        boost->getIArchive() & *ip;
        LOGL_DEBUG(_log, "WcsFormatter read end");
        return ip;
    }
    auto fits = std::dynamic_pointer_cast<dafPersist::FitsStorage>(storage);
    if (fits) {
        LOGL_DEBUG(_log, "WcsFormatter read FitsStorage");
        int hdu = additionalData->get<int>("hdu", fits::DEFAULT_HDU);
        std::shared_ptr<dafBase::PropertySet> md = afw::fits::readMetadata(fits->getPath(), hdu);
        image::Wcs* ip = new image::Wcs(md);
        LOGL_DEBUG(_log, "WcsFormatter read end");
        return ip;
    }
    throw LSST_EXCEPT(pexExcept::RuntimeError, "Unrecognized FormatterStorage for Wcs");
}

void WcsFormatter::update(dafBase::Persistable*, std::shared_ptr<dafPersist::FormatterStorage>,
                          std::shared_ptr<dafBase::PropertySet>) {
    throw LSST_EXCEPT(pexExcept::RuntimeError, "Unexpected call to update for Wcs");
}

std::shared_ptr<dafBase::PropertyList> WcsFormatter::generatePropertySet(image::Wcs const& wcs) {
    // Only generates properties for the first wcsInfo.
    std::shared_ptr<dafBase::PropertyList> wcsProps(new dafBase::PropertyList());

    assert(wcs._wcsInfo);  // default ctor is private, so an uninitialized Wcs should not exist in the wild

    wcsProps->add("NAXIS", wcs._wcsInfo[0].naxis, "number of data axes");
    // EQUINOX is "not relevant" (FITS definition, version 3.0, page 30) when
    // dealing with ICRS, and may confuse readers. Don't write it.
    if (strncmp(wcs._wcsInfo[0].radesys, "ICRS", 4) != 0) {
        wcsProps->add("EQUINOX", wcs._wcsInfo[0].equinox, "Equinox of coordinates");
    }
    wcsProps->add("RADESYS", std::string(wcs._wcsInfo[0].radesys), "Coordinate system for equinox");
    wcsProps->add("CRPIX1", wcs._wcsInfo[0].crpix[0], "WCS Coordinate reference pixel");
    wcsProps->add("CRPIX2", wcs._wcsInfo[0].crpix[1], "WCS Coordinate reference pixel");
    wcsProps->add("CD1_1", wcs._wcsInfo[0].cd[0], "WCS Coordinate scale matrix");
    wcsProps->add("CD1_2", wcs._wcsInfo[0].cd[1], "WCS Coordinate scale matrix");
    wcsProps->add("CD2_1", wcs._wcsInfo[0].cd[2], "WCS Coordinate scale matrix");
    wcsProps->add("CD2_2", wcs._wcsInfo[0].cd[3], "WCS Coordinate scale matrix");
    wcsProps->add("CRVAL1", wcs._wcsInfo[0].crval[0], "WCS Ref value (RA in decimal degrees)");
    wcsProps->add("CRVAL2", wcs._wcsInfo[0].crval[1], "WCS Ref value (DEC in decimal degrees)");
    wcsProps->add("CUNIT1", std::string(wcs._wcsInfo[0].cunit[0]));
    wcsProps->add("CUNIT2", std::string(wcs._wcsInfo[0].cunit[1]));
    //
    // Some projections need PVi_j keywords.  Add them.
    //
    for (int i = 0; i != wcs._wcsInfo[0].npv; ++i) {
        auto const pv = wcs._wcsInfo[0].pv[i];
        int const ii = pv.i > 0 ? pv.i : (wcs._wcsInfo[0].lat + 1);  // 0 => latitude axis (see wcslib/wsc.h)
        char key[20];
        sprintf(key, "PV%d_%d", ii, pv.m);
        wcsProps->add(key, pv.value);
    }

    std::string ctype1(wcs._wcsInfo[0].ctype[0]);
    std::string ctype2(wcs._wcsInfo[0].ctype[1]);
    wcsProps->add("CTYPE1", ctype1, "WCS Coordinate type");
    wcsProps->add("CTYPE2", ctype2, "WCS Coordinate type");

    return wcsProps;
}

template <class Archive>
void WcsFormatter::delegateSerialize(Archive& ar, int const, dafBase::Persistable* persistable) {
    LOGL_DEBUG(_log, "WcsFormatter delegateSerialize start");
    image::Wcs* ip = dynamic_cast<image::Wcs*>(persistable);
    if (ip == 0) {
        throw LSST_EXCEPT(pexExcept::RuntimeError, "Serializing non-Wcs");
    }

    // Serialize most fields normally
    ar & ip->_nWcsInfo & ip->_relax;
    ar & ip->_wcsfixCtrl & ip->_wcshdrCtrl & ip->_nReject;
    ar & ip->_coordSystem;

    // If we are loading, create the array of Wcs parameter structs
    if (Archive::is_loading::value) {
        ip->_wcsInfo = reinterpret_cast<wcsprm*>(malloc(ip->_nWcsInfo * sizeof(wcsprm)));
    }

    for (int i = 0; i < ip->_nWcsInfo; ++i) {
        // If we are loading, initialize the struct first
        if (Archive::is_loading::value) {
            ip->_wcsInfo[i].flag = -1;
            wcsini(1, 2, &(ip->_wcsInfo[i]));
        }

        // Serialize only critical Wcs parameters
        // wcslib provides support for arrays of wcs', but we only
        // implement support for one.
        ar & ip->_wcsInfo[i].naxis;
        ar & ip->_wcsInfo[i].equinox;
        ar & ip->_wcsInfo[i].radesys;
        ar & ip->_wcsInfo[i].crpix[0];
        ar & ip->_wcsInfo[i].crpix[1];
        ar & ip->_wcsInfo[i].cd[0];
        ar & ip->_wcsInfo[i].cd[1];
        ar & ip->_wcsInfo[i].cd[2];
        ar & ip->_wcsInfo[i].cd[3];
        ar & ip->_wcsInfo[i].crval[0];
        ar & ip->_wcsInfo[i].crval[1];
        ar & ip->_wcsInfo[i].cunit[0];
        ar & ip->_wcsInfo[i].cunit[1];
        ar & ip->_wcsInfo[i].ctype[0];
        ar & ip->_wcsInfo[i].ctype[1];
        ar & ip->_wcsInfo[i].altlin;

        // If we are loading, compute intermediate values given those above
        if (Archive::is_loading::value) {
            ip->_wcsInfo[i].flag = 0;
            wcsset(&(ip->_wcsInfo[i]));
        }
    }
    LOGL_DEBUG(_log, "WcsFormatter delegateSerialize end");
}

// Explicit template specializations confuse Doxygen, tell it to ignore them
/// @cond
template void WcsFormatter::delegateSerialize(boost::archive::text_oarchive&, int, dafBase::Persistable*);
template void WcsFormatter::delegateSerialize(boost::archive::text_iarchive&, int, dafBase::Persistable*);
template void WcsFormatter::delegateSerialize(boost::archive::binary_oarchive&, int, dafBase::Persistable*);
template void WcsFormatter::delegateSerialize(boost::archive::binary_iarchive&, int, dafBase::Persistable*);
/// @endcond

std::shared_ptr<dafPersist::Formatter> WcsFormatter::createInstance(
        std::shared_ptr<pexPolicy::Policy> policy) {
    return std::shared_ptr<dafPersist::Formatter>(new WcsFormatter(policy));
}
}
}
}  // end lsst::afw::formatters
