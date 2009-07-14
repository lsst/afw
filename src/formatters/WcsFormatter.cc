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

//#include "boost/serialization/shared_ptr.hpp"
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


namespace afwForm = lsst::afw::formatters;
namespace afwImg = lsst::afw::image;
namespace dafBase = lsst::daf::base;
namespace dafPersist = lsst::daf::persistence;
namespace pexPolicy = lsst::pex::policy;
namespace pexExcept = lsst::pex::exceptions;


dafPersist::FormatterRegistration afwForm::WcsFormatter::registration(
    "Wcs", typeid(afwImg::Wcs), createInstance);

afwForm::WcsFormatter::WcsFormatter(
    pexPolicy::Policy::Ptr policy) :
    dafPersist::Formatter(typeid(*this)) {
}

afwForm::WcsFormatter::~WcsFormatter(void) {
}

void afwForm::WcsFormatter::write(
    dafBase::Persistable const* persistable,
    dafPersist::Storage::Ptr storage,
    dafBase::PropertySet::Ptr additionalData) {
    execTrace("WcsFormatter write start");
    afwImg::Wcs const* ip =
        dynamic_cast<afwImg::Wcs const*>(persistable);
    if (ip == 0) {
        throw LSST_EXCEPT(pexExcept::RuntimeErrorException, "Persisting non-Wcs");
    }
    if (typeid(*storage) == typeid(dafPersist::BoostStorage)) {
        execTrace("WcsFormatter write BoostStorage");
        dafPersist::BoostStorage* boost = dynamic_cast<dafPersist::BoostStorage*>(storage.get());
        boost->getOArchive() & *ip;
        execTrace("WcsFormatter write end");
        return;
    }
    throw LSST_EXCEPT(pexExcept::RuntimeErrorException, "Unrecognized Storage for Wcs");
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
    throw LSST_EXCEPT(pexExcept::RuntimeErrorException, "Unrecognized Storage for Wcs");
}

void afwForm::WcsFormatter::update(
    dafBase::Persistable* persistable,
    dafPersist::Storage::Ptr storage,
    dafBase::PropertySet::Ptr additionalData) {
    throw LSST_EXCEPT(pexExcept::RuntimeErrorException, "Unexpected call to update for Wcs");
}


/// Provide a function to serialise an Eigen::Matrix so we can persist the SIP matrices
template <class Archive>
void serializeEigenArray(Archive& ar, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& m) {
    int rows = m.rows();
    int cols = m.cols();
    ar & rows & cols;
    if (Archive::is_loading::value) {
        m = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>(rows, cols);
    }
    for (int j = 0; j < m.cols(); ++j) {
        for (int i = 0; i < m.rows(); ++i) {
            ar & m(i,j);
        }
    }
}


static void encodeSipHeader(lsst::daf::base::PropertySet::Ptr wcsProps,
                            std::string const& which,   ///< Either A,B, Ap or Bp
                            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> const& m) {
    int order = m.rows();
    if (m.cols() != order) {
        throw LSST_EXCEPT(pexExcept::DomainErrorException,
            "sip" + which + " matrix is not square");
    }
    if (order > 0) {
        order -= 1; // match SIP convention
        wcsProps->add(which + "_ORDER", static_cast<int>(order));
        for (int i = 0; i <= order; ++i) {
            for (int j = 0; j <= order; ++j) {
                double val = m(i, j);
                if (val != 0.0) {
                    wcsProps->add((boost::format("%1%_%2%_%3%")
                                   % which % i % j).str(), val);
                }
            }
        }
    }
}

dafBase::PropertySet::Ptr
afwForm::WcsFormatter::generatePropertySet(afwImg::Wcs const& wcs) {
    // Only generates properties for the first wcsInfo.
    dafBase::PropertySet::Ptr wcsProps(new dafBase::PropertySet());
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
    std::string ctype1(wcs._wcsInfo[0].ctype[0]);
    std::string ctype2(wcs._wcsInfo[0].ctype[1]);
    if (wcs._sipA.rows() > 0 || wcs._sipB.rows() > 0 ||
        wcs._sipAp.rows() > 0 || wcs._sipBp.rows() > 0) {
        if (ctype1.rfind("-SIP") == std::string::npos) {
            ctype1 += "-SIP";
        }
        if (ctype2.rfind("-SIP") == std::string::npos) {
            ctype2 += "-SIP";
        }
        encodeSipHeader(wcsProps, "A", wcs._sipA);
        encodeSipHeader(wcsProps, "B", wcs._sipB);
        encodeSipHeader(wcsProps, "AP", wcs._sipAp);
        encodeSipHeader(wcsProps, "BP", wcs._sipBp);
    }
    wcsProps->add("CTYPE1", ctype1);
    wcsProps->add("CTYPE2", ctype2);

    return wcsProps;
}

template <class Archive>
void afwForm::WcsFormatter::delegateSerialize(
    Archive& ar, int const version, dafBase::Persistable* persistable) {
    execTrace("WcsFormatter delegateSerialize start");
    afwImg::Wcs* ip = dynamic_cast<afwImg::Wcs*>(persistable);
    if (ip == 0) {
        throw LSST_EXCEPT(pexExcept::RuntimeErrorException, "Serializing non-Wcs");
    }

    // Serialize most fields normally
    ar & ip->_nWcsInfo & ip->_relax;
    ar & ip->_wcsfixCtrl & ip->_wcshdrCtrl & ip->_nReject;
    
    serializeEigenArray(ar, ip->_sipA);
    serializeEigenArray(ar, ip->_sipAp);
    serializeEigenArray(ar, ip->_sipB);
    serializeEigenArray(ar, ip->_sipBp);
    
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

dafPersist::Formatter::Ptr afwForm::WcsFormatter::createInstance(
    pexPolicy::Policy::Ptr policy) {
    afwForm::WcsFormatter wcsF =  afwForm::WcsFormatter(policy);
    return dafPersist::Formatter::Ptr(&wcsF);
    //return dafPersist::Formatter::Ptr(new afwForm::WcsFormatter(policy));
}
