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
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

//#include "boost/serialization/shared_ptr.hpp"
#include "wcslib/wcs.h"

#include "lsst/daf/base.h"
#include "lsst/daf/persistence.h"
#include "lsst/daf/persistence/PropertySetFormatter.h"
#include "lsst/pex/exceptions.h"
#include "lsst/pex/logging/Trace.h"
#include "lsst/afw/formatters/ImageFormatter.h"
#include "lsst/afw/formatters/MaskedImageFormatter.h"
#include "lsst/afw/formatters/WcsFormatter.h"
#include "lsst/afw/image/Wcs.h"

#define EXEC_TRACE  20
static void execTrace(std::string s, int level = EXEC_TRACE) {
    lsst::pex::logging::Trace("afw.WcsFormatter", level, s);
}


namespace afwForm = lsst::afw::formatters;
namespace afwImg = lsst::afw::image;
namespace dafBase = lsst::daf::base;
namespace dafPersist = lsst::daf::persistence;
namespace pexPolicy = lsst::pex::policy;
namespace pexExcept = lsst::pex::exceptions;


dafPersist::FormatterRegistration afwForm::WcsFormatter::registration(
    "Wcs", typeid(afwImg::Wcs), createInstance);

afwForm::WcsFormatter::WcsFormatter(
    pexPolicy::Policy::Ptr) :
    dafPersist::Formatter(typeid(this)) {
}

afwForm::WcsFormatter::~WcsFormatter(void) {
}

void afwForm::WcsFormatter::write(
    dafBase::Persistable const* persistable,
    dafPersist::Storage::Ptr storage,
    dafBase::PropertySet::Ptr) {
    execTrace("WcsFormatter write start");
    afwImg::Wcs const* ip =
        dynamic_cast<afwImg::Wcs const*>(persistable);
    if (ip == 0) {
        throw LSST_EXCEPT(pexExcept::RuntimeError, "Persisting non-Wcs");
    }
    if (typeid(*storage) == typeid(dafPersist::BoostStorage)) {
        execTrace("WcsFormatter write BoostStorage");
        dafPersist::BoostStorage* boost = dynamic_cast<dafPersist::BoostStorage*>(storage.get());
        boost->getOArchive() & *ip;
        execTrace("WcsFormatter write end");
        return;
    }
    throw LSST_EXCEPT(pexExcept::RuntimeError, "Unrecognized Storage for Wcs");
}

dafBase::Persistable* afwForm::WcsFormatter::read(
    dafPersist::Storage::Ptr storage,
    dafBase::PropertySet::Ptr additionalData) {
    execTrace("WcsFormatter read start");
    if (typeid(*storage) == typeid(dafPersist::BoostStorage)) {
        afwImg::Wcs* ip = new afwImg::Wcs;
        execTrace("WcsFormatter read BoostStorage");
        dafPersist::BoostStorage* boost = dynamic_cast<dafPersist::BoostStorage*>(storage.get());
        boost->getIArchive() & *ip;
        execTrace("WcsFormatter read end");
        return ip;
    }
    else if (typeid(*storage) == typeid(dafPersist::FitsStorage)) {
        execTrace("WcsFormatter read FitsStorage");
        dafPersist::FitsStorage* fits = dynamic_cast<dafPersist::FitsStorage*>(storage.get());
        int hdu = additionalData->get<int>("hdu", 0);
        dafBase::PropertySet::Ptr md =
            afwImg::readMetadata(fits->getPath(), hdu);
        afwImg::Wcs* ip = new afwImg::Wcs(md);
        execTrace("WcsFormatter read end");
        return ip;
    }
    throw LSST_EXCEPT(pexExcept::RuntimeError, "Unrecognized Storage for Wcs");
}

void afwForm::WcsFormatter::update(
    dafBase::Persistable*,
    dafPersist::Storage::Ptr,
    dafBase::PropertySet::Ptr) {
    throw LSST_EXCEPT(pexExcept::RuntimeError, "Unexpected call to update for Wcs");
}

dafBase::PropertyList::Ptr
afwForm::WcsFormatter::generatePropertySet(afwImg::Wcs const& wcs) {
    // Only generates properties for the first wcsInfo.
    dafBase::PropertyList::Ptr wcsProps(new dafBase::PropertyList());

    assert(wcs._wcsInfo); // default ctor is private, so an uninitialized Wcs should not exist in the wild

    wcsProps->add("NAXIS", wcs._wcsInfo[0].naxis, "number of data axes");
    wcsProps->add("EQUINOX", wcs._wcsInfo[0].equinox, "Equinox of coordinates");
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
        int const ii = pv.i > 0 ? pv.i : (wcs._wcsInfo[0].lat + 1); // 0 => latitude axis (see wcslib/wsc.h)
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
void afwForm::WcsFormatter::delegateSerialize(
    Archive& ar, int const, dafBase::Persistable* persistable) {
    execTrace("WcsFormatter delegateSerialize start");
    afwImg::Wcs* ip = dynamic_cast<afwImg::Wcs*>(persistable);
    if (ip == 0) {
        throw LSST_EXCEPT(pexExcept::RuntimeError, "Serializing non-Wcs");
    }

    // Serialize most fields normally
    ar & ip->_nWcsInfo & ip->_relax;
    ar & ip->_wcsfixCtrl & ip->_wcshdrCtrl & ip->_nReject;
    ar & ip->_coordSystem;

    
    // If we are loading, create the array of Wcs parameter structs
    if (Archive::is_loading::value) {
        ip->_wcsInfo =
            reinterpret_cast<wcsprm*>(malloc(ip->_nWcsInfo * sizeof(wcsprm)));
    }


    for (int i = 0; i < ip->_nWcsInfo; ++i) {
        // If we are loading, initialize the struct first
        if (Archive::is_loading::value) {
            ip->_wcsInfo[i].flag = -1;
            wcsini(1, 2, &(ip->_wcsInfo[i]));
        }

        // Serialize only critical Wcs parameters
        //wcslib provides support for arrays of wcs', but we only
        //implement support for one.
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
    execTrace("WcsFormatter delegateSerialize end");
}

template void afwForm::WcsFormatter::delegateSerialize(
    boost::archive::text_oarchive & , int, dafBase::Persistable*);
template void afwForm::WcsFormatter::delegateSerialize(
    boost::archive::text_iarchive & , int, dafBase::Persistable*);
template void afwForm::WcsFormatter::delegateSerialize(
    boost::archive::binary_oarchive & , int, dafBase::Persistable*);
template void afwForm::WcsFormatter::delegateSerialize(
    boost::archive::binary_iarchive & , int, dafBase::Persistable*);

dafPersist::Formatter::Ptr afwForm::WcsFormatter::createInstance(
    pexPolicy::Policy::Ptr policy) {
    return dafPersist::Formatter::Ptr(new afwForm::WcsFormatter(policy));
}

