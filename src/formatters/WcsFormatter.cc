// -*- lsst-c++ -*-

/** \file
 * \brief Implementation of WcsFormatter class
 *
 * \author $Author: ktlim $
 * \version $Revision: 2151 $
 * \date $Date$
 *
 * Contact: Kian-Tat Lim (ktl@slac.stanford.edu)
 *
 * \ingroup afw
 */

#ifndef __GNUC__
#  define __attribute__(x) /*NOTHING*/
#endif
static char const* SVNid __attribute__((unused)) = "$Id$";

#include <lsst/afw/formatters/WcsFormatter.h>

#include <lsst/pex/persistence/FormatterImpl.h>
#include <lsst/daf/data/DataPropertyFormatter.h>
#include <lsst/afw/formatters/ImageFormatter.h>
#include <lsst/afw/formatters/MaskedImageFormatter.h>
#include <lsst/afw/formatters/WcsFormatter.h>
#include <boost/serialization/shared_ptr.hpp>

#include <stdlib.h>

#include <lsst/afw/image.h>
#include "wcslib/wcs.h"

#include <lsst/daf/data/SupportFactory.h>
#include <lsst/pex/exceptions.h>
#include <lsst/pex/persistence/LogicalLocation.h>
#include <lsst/pex/persistence/BoostStorage.h>
#include <lsst/pex/persistence/FitsStorage.h>
#include <lsst/pex/utils/Trace.h>

#define EXEC_TRACE  20
static void execTrace(std::string s, int level = EXEC_TRACE) {
    lsst::pex::utils::Trace("afw.WcsFormatter", level, s);
}

using namespace lsst::pex::persistence;

namespace lsst {
namespace afw {
namespace formatters {

FormatterRegistration WcsFormatter::registration(
    "WCS", typeid(WCS), createInstance);

WcsFormatter::WcsFormatter(
    lsst::pex::policy::Policy::Ptr policy) :
    Formatter(typeid(*this)) {
}

WcsFormatter::~WcsFormatter(void) {
}

void WcsFormatter::write(
    Persistable const* persistable,
    Storage::Ptr storage,
    lsst::daf::data::DataProperty::PtrType additionalData) {
    execTrace("WcsFormatter write start");
    WCS const* ip =
        dynamic_cast<WCS const*>(persistable);
    if (ip == 0) {
        throw lsst::pex::exceptions::Runtime("Persisting non-WCS");
    }
    if (typeid(*storage) == typeid(BoostStorage)) {
        execTrace("WcsFormatter write BoostStorage");
        BoostStorage* boost = dynamic_cast<BoostStorage*>(storage.get());
        boost->getOArchive() & *ip;
        execTrace("WcsFormatter write end");
        return;
    }
    throw lsst::pex::exceptions::Runtime("Unrecognized Storage for WCS");
}

Persistable* WcsFormatter::read(
    Storage::Ptr storage,
    lsst::daf::data::DataProperty::PtrType additionalData) {
    execTrace("WcsFormatter read start");
    WCS* ip = new WCS;
    if (typeid(*storage) == typeid(BoostStorage)) {
        execTrace("WcsFormatter read BoostStorage");
        BoostStorage* boost = dynamic_cast<BoostStorage*>(storage.get());
        boost->getIArchive() & *ip;
        execTrace("WcsFormatter read end");
        return ip;
    }
    throw lsst::pex::exceptions::Runtime("Unrecognized Storage for WCS");
}

void WcsFormatter::update(
    Persistable* persistable,
    Storage::Ptr storage,
    lsst::daf::data::DataProperty::PtrType additionalData) {
    throw lsst::pex::exceptions::Runtime("Unexpected call to update for WCS");
}

lsst::daf::data::DataProperty::PtrType
WcsFormatter::generateDataProperty(WCS const& wcs) {
    // Only generates DP for the first wcsInfo.
    using lsst::daf::data::DataProperty;
    DataProperty::PtrType wcsDP =
        lsst::daf::data::SupportFactory::createPropertyNode("WCS");
    wcsDP->addProperty(DataProperty("NAXIS", wcs._wcsInfo[0].naxis));
    wcsDP->addProperty(DataProperty("EQUINOX", wcs._wcsInfo[0].equinox));
    wcsDP->addProperty(DataProperty("RADECSYS", std::string(wcs._wcsInfo[0].radesys)));
    wcsDP->addProperty(DataProperty("CRPIX1", wcs._wcsInfo[0].crpix[0]));
    wcsDP->addProperty(DataProperty("CRPIX2", wcs._wcsInfo[0].crpix[1]));
    wcsDP->addProperty(DataProperty("CD1_1", wcs._wcsInfo[0].cd[0]));
    wcsDP->addProperty(DataProperty("CD1_2", wcs._wcsInfo[0].cd[1]));
    wcsDP->addProperty(DataProperty("CD2_1", wcs._wcsInfo[0].cd[2]));
    wcsDP->addProperty(DataProperty("CD2_2", wcs._wcsInfo[0].cd[3]));
    wcsDP->addProperty(DataProperty("CRVAL1", wcs._wcsInfo[0].crval[0]));
    wcsDP->addProperty(DataProperty("CRVAL2", wcs._wcsInfo[0].crval[1]));
    wcsDP->addProperty(DataProperty("CUNIT1", std::string(wcs._wcsInfo[0].cunit[0])));
    wcsDP->addProperty(DataProperty("CUNIT2", std::string(wcs._wcsInfo[0].cunit[1])));
    wcsDP->addProperty(DataProperty("CTYPE1", std::string(wcs._wcsInfo[0].ctype[0])));
    wcsDP->addProperty(DataProperty("CTYPE2", std::string(wcs._wcsInfo[0].ctype[1])));
    return wcsDP;
}

template <class Archive>
void WcsFormatter::delegateSerialize(
    Archive& ar, int const version, Persistable* persistable) {
    execTrace("WcsFormatter delegateSerialize start");
    WCS* ip = dynamic_cast<WCS*>(persistable);
    if (ip == 0) {
        throw lsst::pex::exceptions::Runtime("Serializing non-WCS");
    }

    // Serialize most fields normally
    ar & ip->_fitsMetaData & ip->_nWcsInfo & ip->_relax;
    ar & ip->_wcsfixCtrl & ip->_wcshdrCtrl & ip->_nReject;

    // If we are loading, create the array of WCS parameter structs
    if (Archive::is_loading::value) {
        ip->_wcsInfo =
            reinterpret_cast<wcsprm*>(malloc(ip->_nWcsInfo * sizeof(wcsprm)));
    }

    // Serialize each WCS parameter struct
    for (int i = 0; i < ip->_nWcsInfo; ++i) {

        // If we are loading, initialize the struct first
        if (Archive::is_loading::value) {
            ip->_wcsInfo[i].flag = -1;
            wcsini(1, 2, &(ip->_wcsInfo[i]));
        }

        // Serialize only critical WCS parameters
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

        // If we are loading, compute intermediate values given those above
        if (Archive::is_loading::value) {
            ip->_wcsInfo[i].flag = 0;
            wcsset(&(ip->_wcsInfo[i]));
        }
    }
    execTrace("WcsFormatter delegateSerialize end");
}

Formatter::Ptr WcsFormatter::createInstance(
    lsst::pex::policy::Policy::Ptr policy) {
    return Formatter::Ptr(new WcsFormatter(policy));
}

}}} // namespace lsst::afw::formatters
