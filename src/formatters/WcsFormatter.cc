// -*- lsst-c++ -*-

/** @file
 * @brief Implementation of WcsFormatter class
 *
 * @author $Author: ktlim $
 * @version $Revision: 2151 $
 * @date $Date$
 *
 * Contact: Kian-Tat Lim (ktl@slac.stanford.edu)
 *
 * @ingroup afw
 */

#ifndef __GNUC__
#  define __attribute__(x) /*NOTHING*/
#endif
static char const* SVNid __attribute__((unused)) = "$Id$";

// not used? #include <stdlib.h>

#include "boost/serialization/shared_ptr.hpp"
#include "wcslib/wcs.h"

#include "lsst/daf/base.h"
#include "lsst/daf/persistence.h"
#include "lsst/daf/persistence/PropertySetFormatter.h"
#include "lsst/pex/exceptions.h"
#include "lsst/pex/logging/Trace.h"
#include "lsst/afw/formatters/WcsFormatter.h"
#include "lsst/afw/formatters/ImageFormatter.h"
#include "lsst/afw/formatters/MaskedImageFormatter.h"
#include "lsst/afw/formatters/WcsFormatter.h"
#include "lsst/afw/image/Wcs.h"

#define EXEC_TRACE  20
static void execTrace(std::string s, int level = EXEC_TRACE) {
    lsst::pex::logging::Trace("afw.WcsFormatter", level, s);
}

using lsst::daf::base::Persistable;
using lsst::daf::persistence::BoostStorage;
using lsst::daf::persistence::Storage;
using lsst::afw::image::Wcs;

namespace lsst {
namespace afw {
namespace formatters {

lsst::daf::persistence::FormatterRegistration WcsFormatter::registration(
    "Wcs", typeid(Wcs), createInstance);

WcsFormatter::WcsFormatter(
    lsst::pex::policy::Policy::Ptr policy) :
    lsst::daf::persistence::Formatter(typeid(*this)) {
}

WcsFormatter::~WcsFormatter(void) {
}

void WcsFormatter::write(
    Persistable const* persistable,
    Storage::Ptr storage,
    lsst::daf::base::PropertySet::Ptr additionalData) {
    execTrace("WcsFormatter write start");
    Wcs const* ip =
        dynamic_cast<Wcs const*>(persistable);
    if (ip == 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException, "Persisting non-Wcs");
    }
    if (typeid(*storage) == typeid(BoostStorage)) {
        execTrace("WcsFormatter write BoostStorage");
        BoostStorage* boost = dynamic_cast<BoostStorage*>(storage.get());
        boost->getOArchive() & *ip;
        execTrace("WcsFormatter write end");
        return;
    }
    throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException, "Unrecognized Storage for Wcs");
}

Persistable* WcsFormatter::read(
    Storage::Ptr storage,
    lsst::daf::base::PropertySet::Ptr additionalData) {
    execTrace("WcsFormatter read start");
    Wcs* ip = new Wcs;
    if (typeid(*storage) == typeid(BoostStorage)) {
        execTrace("WcsFormatter read BoostStorage");
        BoostStorage* boost = dynamic_cast<BoostStorage*>(storage.get());
        boost->getIArchive() & *ip;
        execTrace("WcsFormatter read end");
        return ip;
    }
    throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException, "Unrecognized Storage for Wcs");
}

void WcsFormatter::update(
    Persistable* persistable,
    Storage::Ptr storage,
    lsst::daf::base::PropertySet::Ptr additionalData) {
    throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException, "Unexpected call to update for Wcs");
}

static void copyMetadata(std::string const& prefix,
                         lsst::daf::base::PropertySet::Ptr src,
                         lsst::daf::base::PropertySet::Ptr dest) {
    std::string header = prefix + "_ORDER";
    if (!src->exists(header)) {
        return;
    }
    int order = src->get<int>(header);
    boost::format param("%1%_%2%_%3%");
    for (int i = 0; i <= order; ++i) {
        for (int j = 0; j <= order - i; ++j) {
            header = (param % prefix % i % j).str();
            if (src->exists(header)) {
                dest->set(header, src->get<double>(header));
            }
        }
    }
    header = prefix + "_DMAX";
    if (src->exists(header)) {
        dest->set(header, src->get<double>(header));
    }
}

lsst::daf::base::PropertySet::Ptr
WcsFormatter::generatePropertySet(Wcs const& wcs) {
    // Only generates properties for the first wcsInfo.
    lsst::daf::base::PropertySet::Ptr wcsProps(new lsst::daf::base::PropertySet());
    if (!wcs) {                         // nothing to add
        return wcsProps;
    }

    wcsProps->add("NAXIS", wcs._wcsInfo[0].naxis);
    wcsProps->add("EQUINOX", wcs._wcsInfo[0].equinox);
    wcsProps->add("RADECSYS", std::string(wcs._wcsInfo[0].radesys));
    wcsProps->add("CRPIX1", wcs._wcsInfo[0].crpix[0]);
    wcsProps->add("CRPIX2", wcs._wcsInfo[0].crpix[1]);
    wcsProps->add("CD1_1", wcs._wcsInfo[0].cd[0]);
    wcsProps->add("CD1_2", wcs._wcsInfo[0].cd[1]);
    wcsProps->add("CD2_1", wcs._wcsInfo[0].cd[2]);
    wcsProps->add("CD2_2", wcs._wcsInfo[0].cd[3]);
    wcsProps->add("CRVAL1", wcs._wcsInfo[0].crval[0]);
    wcsProps->add("CRVAL2", wcs._wcsInfo[0].crval[1]);
    wcsProps->add("CUNIT1", std::string(wcs._wcsInfo[0].cunit[0]));
    wcsProps->add("CUNIT2", std::string(wcs._wcsInfo[0].cunit[1]));
    wcsProps->add("CTYPE1", std::string(wcs._wcsInfo[0].ctype[0]));
    wcsProps->add("CTYPE2", std::string(wcs._wcsInfo[0].ctype[1]));

    copyMetadata("A", wcs._fitsMetadata, wcsProps);
    copyMetadata("B", wcs._fitsMetadata, wcsProps);
    copyMetadata("AP", wcs._fitsMetadata, wcsProps);
    copyMetadata("BP", wcs._fitsMetadata, wcsProps);

    return wcsProps;
}

template <class Archive>
void WcsFormatter::delegateSerialize(
    Archive& ar, int const version, Persistable* persistable) {
    execTrace("WcsFormatter delegateSerialize start");
    Wcs* ip = dynamic_cast<Wcs*>(persistable);
    if (ip == 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException, "Serializing non-Wcs");
    }

    // Serialize most fields normally
    ar & ip->_fitsMetadata & ip->_nWcsInfo & ip->_relax;
    ar & ip->_wcsfixCtrl & ip->_wcshdrCtrl & ip->_nReject;

    // If we are loading, create the array of Wcs parameter structs
    if (Archive::is_loading::value) {
        ip->_wcsInfo =
            reinterpret_cast<wcsprm*>(malloc(ip->_nWcsInfo * sizeof(wcsprm)));
    }

    // Serialize each Wcs parameter struct
    for (int i = 0; i < ip->_nWcsInfo; ++i) {

        // If we are loading, initialize the struct first
        if (Archive::is_loading::value) {
            ip->_wcsInfo[i].flag = -1;
            wcsini(1, 2, &(ip->_wcsInfo[i]));
        }

        // Serialize only critical Wcs parameters
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

lsst::daf::persistence::Formatter::Ptr WcsFormatter::createInstance(
    lsst::pex::policy::Policy::Ptr policy) {
    return lsst::daf::persistence::Formatter::Ptr(new WcsFormatter(policy));
}

}}} // namespace lsst::afw::formatters
