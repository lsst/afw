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

#include <iostream>
#include <sstream>
#include <cmath>
#include <cstring>

#include "boost/format.hpp"

#include "wcslib/wcs.h"
#include "wcslib/wcsfix.h"
#include "wcslib/wcshdr.h"

#include "lsst/daf/base.h"
#include "lsst/daf/base/Citizen.h"
#include "lsst/afw/formatters/Utils.h"
#include "lsst/afw/formatters/TanWcsFormatter.h"
#include "lsst/pex/exceptions.h"
#include "lsst/log/Log.h"
#include "lsst/afw/geom/AffineTransform.h"
#include "lsst/afw/image/TanWcs.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/table/io/CatalogVector.h"

namespace lsst {
namespace afw {
namespace image {

const int lsstToFitsPixels = +1;
const int fitsToLsstPixels = -1;

TanWcs::TanWcs() : Wcs(), _hasDistortion(false), _sipA(), _sipB(), _sipAp(), _sipBp() {}

bool TanWcs::isPersistable() const {
    if (!_mayBePersistable()) {
        return false;
    }

    return true;
}

geom::Angle TanWcs::pixelScale() const {
    // HACK -- assume "CD" elements are set (and are in degrees)
    double* cd = _wcsInfo->m_cd;
    assert(cd);
    return sqrt(fabs(cd[0] * cd[3] - cd[1] * cd[2])) * geom::degrees;
}

TanWcs::TanWcs(std::shared_ptr<daf::base::PropertySet const> const& fitsMetadata)
        : Wcs(fitsMetadata), _hasDistortion(false), _sipA(), _sipB(), _sipAp(), _sipBp() {
    // Internal params for wcslib. These should be set via policy - but for the moment...
    _relax = 1;
    _wcsfixCtrl = 2;
    _wcshdrCtrl = 2;

    // Check that the header isn't empty
    if (fitsMetadata->nameCount() == 0) {
        std::string msg = "Fits metadata contains no cards";
        throw LSST_EXCEPT(pex::exceptions::InvalidParameterError, msg);
    }

    // Check for tangent plane projection
    std::string ctype1 = fitsMetadata->getAsString("CTYPE1");
    std::string ctype2 = fitsMetadata->getAsString("CTYPE2");

    if ((ctype1.substr(5, 3) != "TAN") || (ctype2.substr(5, 3) != "TAN")) {
        std::string msg = "One or more axes isn't in TAN projection (ctype1 = \"" + ctype1 +
                          "\", ctype2 = \"" + ctype2 + "\")";
        throw LSST_EXCEPT(pex::exceptions::InvalidParameterError, msg);
    }

    // Check for distorton terms. With two ctypes, there are 4 alternatives, only
    // two of which are valid.. Both have distortion terms or both don't.
    int nSip = (((ctype1.substr(8, 4) == "-SIP") ? 1 : 0) + ((ctype2.substr(8, 4) == "-SIP") ? 1 : 0));

    bool isTpv = false;
    {
        std::string key = "TPV_WCS";
        if (fitsMetadata->exists(key) && fitsMetadata->getAsBool(key)) {
            isTpv = true;
        }
    }

    if (isTpv) {
        LOGL_DEBUG("afw.wcs", "Ignoring TPV terms");
    }

    switch (nSip) {
        case 0:
            _hasDistortion = false;
            break;
        case 1: {  // Invalid case. Throw an exception
            std::string msg = "Distortion key found for only one CTYPE";
            throw LSST_EXCEPT(pex::exceptions::InvalidParameterError, msg);
        } break;  // Not necessary, but looks naked without it.
        case 2:
            _hasDistortion = true;

            // Hide the distortion from wcslib
            // Make a copy that we can hack up
            std::shared_ptr<daf::base::PropertySet> const& hackMetadata = fitsMetadata->deepCopy();
            hackMetadata->set<std::string>("CTYPE1", ctype1.substr(0, 8));
            hackMetadata->set<std::string>("CTYPE2", ctype2.substr(0, 8));

            // Save SIP information
            decodeSipHeader(*hackMetadata, "A", _sipA);
            decodeSipHeader(*hackMetadata, "B", _sipB);
            decodeSipHeader(*hackMetadata, "AP", _sipAp);
            decodeSipHeader(*hackMetadata, "BP", _sipBp);

            // Remove SIP headers so that wcslib cannot attempt to use them
            std::vector<std::string> sipPrefixes = {"A", "B", "AP", "BP"};

            std::string sipName;
            std::string orderName;
            for (auto const& prefix : sipPrefixes) {
                orderName = (boost::format("%1%_ORDER") % prefix).str();
                if (!hackMetadata->exists(orderName)) continue;
                int order = hackMetadata->getAsInt(orderName);

                for (int p = 0; p <= order; p++) {
                    for (int q = 0; p + q <= order; q++) {
                        sipName = (boost::format("%1%_%2%_%3%") % prefix % p % q).str();
                        if (hackMetadata->exists(sipName)) {
                            hackMetadata->remove(sipName);
                        }
                    }
                }
                sipName = (boost::format("%1%_DMAX") % prefix).str();
                if (hackMetadata->exists(sipName)) {
                    hackMetadata->remove(sipName);
                }
                sipName = (boost::format("%1%_ORDER") % prefix).str();
                if (hackMetadata->exists(sipName)) {
                    hackMetadata->remove(sipName);
                }
            }

            // this gets called in the Wcs (base class) constructor
            // We just changed fitsMetadata, so we have to re-init wcslib
            initWcsLibFromFits(hackMetadata);

            break;
    }

    // Check that the existence of forward sip matrices <=> existence of reverse matrices
    if (_hasDistortion) {
        if (_sipA.rows() <= 1 || _sipB.rows() <= 1) {
            std::string msg = "Existence of forward distorton matrices suggested, but not found";
            throw LSST_EXCEPT(pex::exceptions::InvalidParameterError, msg);
        }

        if (_sipAp.rows() <= 1 || _sipBp.rows() <= 1) {
            std::string msg = "Forward distorton matrices present, but no reverse matrices";
            throw LSST_EXCEPT(pex::exceptions::InvalidParameterError, msg);
        }
    }
}

void TanWcs::decodeSipHeader(daf::base::PropertySet const& fitsMetadata, std::string const& which,
                             Eigen::MatrixXd& m) {
    std::string header = which + "_ORDER";
    if (!fitsMetadata.exists(header)) return;
    int order = fitsMetadata.getAsInt(header);
    m.resize(order + 1, order + 1);
    boost::format format("%1%_%2%_%3%");
    for (int i = 0; i <= order; ++i) {
        for (int j = 0; j <= order; ++j) {
            header = (format % which % i % j).str();
            if (fitsMetadata.exists(header)) {
                m(i, j) = fitsMetadata.getAsDouble(header);
            } else {
                m(i, j) = 0.0;
            }
        }
    }
}

TanWcs::TanWcs(geom::Point2D const& crval, geom::Point2D const& crpix, Eigen::Matrix2d const& cd,
               double equinox, std::string const& raDecSys, std::string const& cunits1,
               std::string const& cunits2)
        : Wcs(crval, crpix, cd, "RA---TAN", "DEC--TAN", equinox, raDecSys, cunits1, cunits2),
          _hasDistortion(false),
          _sipA(),
          _sipB(),
          _sipAp(),
          _sipBp() {}

TanWcs::TanWcs(geom::Point2D const& crval, geom::Point2D const& crpix, Eigen::Matrix2d const& cd,
               Eigen::MatrixXd const& sipA, Eigen::MatrixXd const& sipB, Eigen::MatrixXd const& sipAp,
               Eigen::MatrixXd const& sipBp, double equinox, std::string const& raDecSys,
               std::string const& cunits1, std::string const& cunits2)
        : Wcs(crval, crpix, cd, "RA---TAN", "DEC--TAN", equinox, raDecSys, cunits1, cunits2),
          _hasDistortion(true),
          // Sip's set by a dedicated method that does error checking
          _sipA(),
          _sipB(),
          _sipAp(),
          _sipBp() {
    // Input checking is done constructor of base class, so don't need to do any here.

    // Set the distortion terms
    setDistortionMatrices(sipA, sipB, sipAp, sipBp);
}

TanWcs::TanWcs(TanWcs const& rhs)
        : Wcs(rhs),
          _hasDistortion(rhs._hasDistortion),
          _sipA(rhs._sipA),
          _sipB(rhs._sipB),
          _sipAp(rhs._sipAp),
          _sipBp(rhs._sipBp) {}

bool TanWcs::_isSubset(Wcs const& rhs) const {
    if (!Wcs::_isSubset(rhs)) {
        return false;
    }
    // We only care about the derived-class part if we have a distortion; this could mean
    // a TanWcs with no distortion may be equal to a plain Wcs, but that doesn't happen
    // in practice because have different wcslib data structures.
    if (this->hasDistortion()) {
        TanWcs const* other = dynamic_cast<TanWcs const*>(&rhs);
        return other && other->_hasDistortion && _sipA == other->_sipA && _sipB == other->_sipB &&
               _sipAp == other->_sipAp && _sipBp == other->_sipBp;
    }
    return true;
}

std::shared_ptr<Wcs> TanWcs::clone(void) const { return std::shared_ptr<Wcs>(new TanWcs(*this)); }

//
// Accessors
//
geom::Point2D TanWcs::skyToPixelImpl(geom::Angle sky1,  // RA
                                     geom::Angle sky2   // Dec
                                     ) const {
    if (_wcsInfo == NULL) {
        throw(LSST_EXCEPT(pex::exceptions::RuntimeError, "Wcs structure not initialised"));
    }

    double skyTmp[2];
    double imgcrd[2];
    double phi, theta;
    double pixTmp[2];

    // Estimate undistorted pixel coordinates
    int stat[1];
    int status = 0;

    skyTmp[_wcsInfo->lng] = sky1.asDegrees();
    skyTmp[_wcsInfo->lat] = sky2.asDegrees();

    status = wcss2p(_wcsInfo, 1, 2, skyTmp, &phi, &theta, imgcrd, pixTmp, stat);
    if (status > 0) {
        throw LSST_EXCEPT(pex::exceptions::RuntimeError,
                          (boost::format("Error: wcslib returned a status code of %d at sky %s, %s deg: %s") %
                           status % sky1.asDegrees() % sky2.asDegrees() % wcs_errmsg[status])
                                  .str());
    }

    // Correct for distortion. We follow the notation of Shupe et al. here, including
    // capitalisation
    if (_hasDistortion) {
        geom::Point2D pix = geom::Point2D(pixTmp[0], pixTmp[1]);
        geom::Point2D dpix = distortPixel(pix);
        pixTmp[0] = dpix[0];
        pixTmp[1] = dpix[1];
    }

    // wcslib assumes 1-indexed coords
    double offset = PixelZeroPos + fitsToLsstPixels;
    return geom::Point2D(pixTmp[0] + offset, pixTmp[1] + offset);
}

namespace {

/** @internal Generate a vector of polynomial elements, x^i
 *
 * This is useful for optimising polynomial evaluations */
std::vector<double> polynomialElements(std::size_t const order, double const value) {
    std::vector<double> poly(order + 1);
    poly[0] = 1.0;
    if (order == 0) {
        return poly;
    }
    poly[1] = value;
    for (std::size_t i = 2; i <= order; ++i) {
        poly[i] = poly[i - 1] * value;
    }
    return poly;
}

}  // anonymous namespace

geom::Point2D TanWcs::undistortPixel(geom::Point2D const& pix) const {
    if (!_hasDistortion) {
        return geom::Point2D(pix);
    }
    // If the following assertions aren't true then something has gone seriously wrong.
    assert(_sipB.rows() > 0);
    assert(_sipA.rows() == _sipA.cols());
    assert(_sipB.rows() == _sipB.cols());

    // Polynomial orders for U = f(u,v) and V = g(u,v)
    // Note these may be different in general, but not usually in practice.
    // We use int here rather than size_t because they need to be compared against
    // negative values in the loops below.
    int const fOrder = _sipA.rows();
    int const gOrder = _sipB.rows();

    double u = pix[0] - _wcsInfo->crpix[0];  // Relative pixel coords
    double v = pix[1] - _wcsInfo->crpix[1];

    std::vector<double> uPoly = polynomialElements(std::max(fOrder, gOrder), u);
    std::vector<double> vPoly = polynomialElements(std::max(fOrder, gOrder), v);

    double f = 0;
    for (int i = 0; i < fOrder; ++i) {
        for (int j = 0; j < fOrder - i; ++j) {
            f += _sipA(i, j) * uPoly[i] * vPoly[j];
        }
    }

    double g = 0;
    for (int i = 0; i < gOrder; ++i) {
        for (int j = 0; j < gOrder - i; ++j) {
            g += _sipB(i, j) * uPoly[i] * vPoly[j];
        }
    }

    return geom::Point2D(pix[0] + f, pix[1] + g);
}

geom::Point2D TanWcs::distortPixel(geom::Point2D const& pix) const {
    if (!_hasDistortion) {
        return geom::Point2D(pix);
    }
    // If the following assertions aren't true then something has gone seriously wrong.
    assert(_sipBp.rows() > 0);
    assert(_sipAp.rows() == _sipAp.cols());
    assert(_sipBp.rows() == _sipBp.cols());

    // Polynomial orders for u = F(U,V) and v = G(U,V)
    // Note these may be different in general, but not usually in practise.
    std::size_t const fOrder = _sipAp.rows();
    std::size_t const gOrder = _sipBp.rows();

    double U = pix[0] - _wcsInfo->crpix[0];  // Relative, undistorted pixel coords
    double V = pix[1] - _wcsInfo->crpix[1];

    std::vector<double> uPoly = polynomialElements(std::max(fOrder, gOrder), U);
    std::vector<double> vPoly = polynomialElements(std::max(fOrder, gOrder), V);

    double F = 0;
    for (std::size_t i = 0; i < fOrder; ++i) {
        for (std::size_t j = 0; j < fOrder; ++j) {
            F += _sipAp(i, j) * uPoly[i] * vPoly[j];
        }
    }

    double G = 0;
    for (std::size_t i = 0; i < gOrder; ++i) {
        for (std::size_t j = 0; j < gOrder; ++j) {
            G += _sipBp(i, j) * uPoly[i] * vPoly[j];
        }
    }
    return geom::Point2D(U + F + _wcsInfo->crpix[0], V + G + _wcsInfo->crpix[1]);
}

/*
 * Worker routine for pixelToSky
 */
void TanWcs::pixelToSkyImpl(double pixel1, double pixel2, geom::Angle sky[2]) const {
    if (_wcsInfo == NULL) {
        throw(LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "Wcs structure not initialised"));
    }

    // wcslib assumes 1-indexed coordinates
    double pixTmp[2] = {pixel1 - PixelZeroPos + lsstToFitsPixels, pixel2 - PixelZeroPos + lsstToFitsPixels};
    double imgcrd[2];
    double phi, theta;

    // Correct pixel positions for distortion if necessary
    if (_hasDistortion) {
        geom::Point2D pix = geom::Point2D(pixTmp[0], pixTmp[1]);
        geom::Point2D dpix = undistortPixel(pix);
        pixTmp[0] = dpix[0];
        pixTmp[1] = dpix[1];
    }

    int status = 0;
    double skyTmp[2];
    if (wcsp2s(_wcsInfo, 1, 2, pixTmp, imgcrd, &phi, &theta, skyTmp, &status) > 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                          (boost::format("Error: wcslib returned a status code of %d at pixel %s, %s: %s") %
                           status % pixel1 % pixel2 % wcs_errmsg[status])
                                  .str());
    }
    sky[0] = skyTmp[0] * geom::degrees;
    sky[1] = skyTmp[1] * geom::degrees;
}

void TanWcs::flipImage(int flipLR, int flipTB, lsst::afw::geom::Extent2I dimensions) const {
    if (hasDistortion()) {
        throw LSST_EXCEPT(pex::exceptions::LogicError, "flipImage is not implemented for TAN-SIP");
    }
    Wcs::flipImage(flipLR, flipTB, dimensions);
}

void TanWcs::rotateImageBy90(int nQuarter, lsst::afw::geom::Extent2I dimensions) const {
    if (hasDistortion()) {
        throw LSST_EXCEPT(pex::exceptions::LogicError, "rotateImageBy90 is not implemented for TAN-SIP");
    }
    Wcs::rotateImageBy90(nQuarter, dimensions);
}

std::shared_ptr<daf::base::PropertyList> TanWcs::getFitsMetadata() const {
    return formatters::TanWcsFormatter::generatePropertySet(*this);
}

//
// Mutators
//

void TanWcs::setDistortionMatrices(Eigen::MatrixXd const& sipA, Eigen::MatrixXd const& sipB,
                                   Eigen::MatrixXd const& sipAp, Eigen::MatrixXd const& sipBp) {
    if (sipA.rows() != sipA.cols()) {
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "Error: Matrix sipA must be square");
    }

    if (sipB.rows() != sipB.cols()) {
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "Error: Matrix sipB must be square");
    }

    if (sipAp.rows() != sipAp.cols()) {
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "Error: Matrix sipAp must be square");
    }

    if (sipBp.rows() != sipBp.cols()) {
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "Error: Matrix sipBp must be square");
    }

    // Set the SIP terms
    _hasDistortion = true;
    _sipA = sipA;
    _sipB = sipB;
    _sipAp = sipAp;
    _sipBp = sipBp;
}

// -------------- Table-based Persistence -------------------------------------------------------------------

/*
 *  We use the Wcs base class persistence to write one table, and then add another containing
 *  the SIP coefficients only if hasDistortion() is true.
 *
 *  The second table's schema depends on the SIP orders, so it will not necessarily be the same
 *  for all TanWcs objects.
 */

class TanWcsFactory : public table::io::PersistableFactory {
public:
    explicit TanWcsFactory(std::string const& name) : table::io::PersistableFactory(name) {}

    virtual std::shared_ptr<table::io::Persistable> read(InputArchive const& archive,
                                                         CatalogVector const& catalogs) const {
        LSST_ARCHIVE_ASSERT(catalogs.size() >= 1u);
        std::shared_ptr<table::BaseRecord const> sipRecord;
        if (catalogs.size() > 1u) {
            LSST_ARCHIVE_ASSERT(catalogs.size() == 2u);
            LSST_ARCHIVE_ASSERT(catalogs.front().size() == 1u);
            LSST_ARCHIVE_ASSERT(catalogs.back().size() == 1u);
            sipRecord = catalogs.back().begin();
        }
        std::shared_ptr<TanWcs> result(new TanWcs(catalogs.front().front(), sipRecord));
        return result;
    }
};

namespace {

std::string getTanWcsPersistenceName() { return "TanWcs"; }

TanWcsFactory registration(getTanWcsPersistenceName());

}  // anonymous

std::string TanWcs::getPersistenceName() const { return getTanWcsPersistenceName(); }

void TanWcs::write(OutputArchiveHandle& handle) const {
    Wcs::write(handle);
    if (hasDistortion()) {
        afw::table::Schema schema;
        afw::table::Key<afw::table::Array<double> > keyA(schema.addField<afw::table::Array<double> >(
                "A", "x forward transform coefficients (column-major)", _sipA.size()));
        afw::table::Key<afw::table::Array<double> > keyB(schema.addField<afw::table::Array<double> >(
                "B", "y forward transform coefficients (column-major)", _sipB.size()));
        afw::table::Key<afw::table::Array<double> > keyAp(schema.addField<afw::table::Array<double> >(
                "Ap", "x reverse transform coefficients (column-major)", _sipAp.size()));
        afw::table::Key<afw::table::Array<double> > keyBp(schema.addField<afw::table::Array<double> >(
                "Bp", "y reverse transform coefficients (column-major)", _sipBp.size()));
        afw::table::BaseCatalog catalog = handle.makeCatalog(schema);
        std::shared_ptr<afw::table::BaseRecord> record = catalog.addNew();
        Eigen::Map<Eigen::MatrixXd> mapA((*record)[keyA].getData(), _sipA.rows(), _sipA.cols());
        mapA = _sipA;
        Eigen::Map<Eigen::MatrixXd> mapB((*record)[keyB].getData(), _sipB.rows(), _sipB.cols());
        mapB = _sipB;
        Eigen::Map<Eigen::MatrixXd> mapAp((*record)[keyAp].getData(), _sipAp.rows(), _sipAp.cols());
        mapAp = _sipAp;
        Eigen::Map<Eigen::MatrixXd> mapBp((*record)[keyBp].getData(), _sipBp.rows(), _sipBp.cols());
        mapBp = _sipBp;
        handle.saveCatalog(catalog);
    }
}

TanWcs::TanWcs(afw::table::BaseRecord const& mainRecord,
               std::shared_ptr<afw::table::BaseRecord const> sipRecord)
        : Wcs(mainRecord), _hasDistortion(sipRecord) {
    if (_hasDistortion) {
        typedef afw::table::Array<double> Array;
        afw::table::Key<Array> kA;
        afw::table::Key<Array> kB;
        afw::table::Key<Array> kAp;
        afw::table::Key<Array> kBp;
        try {
            kA = sipRecord->getSchema()["A"];
            kB = sipRecord->getSchema()["B"];
            kAp = sipRecord->getSchema()["Ap"];
            kBp = sipRecord->getSchema()["Bp"];
        } catch (...) {
            throw LSST_EXCEPT(afw::table::io::MalformedArchiveError,
                              "Incorrect schema for TanWcs distortion terms");
        }
        // Adding 0.5 and truncating the result here guarantees we'll get the right answer
        // for small ints even when round-off error is involved.
        int nA = int(std::sqrt(kA.getSize() + 0.5));
        int nB = int(std::sqrt(kB.getSize() + 0.5));
        int nAp = int(std::sqrt(kAp.getSize() + 0.5));
        int nBp = int(std::sqrt(kBp.getSize() + 0.5));
        if (nA * nA != kA.getSize()) {
            throw LSST_EXCEPT(afw::table::io::MalformedArchiveError, "Forward X SIP matrix is not square.");
        }
        if (nB * nB != kB.getSize()) {
            throw LSST_EXCEPT(afw::table::io::MalformedArchiveError, "Forward Y SIP matrix is not square.");
        }
        if (nAp * nAp != kAp.getSize()) {
            throw LSST_EXCEPT(afw::table::io::MalformedArchiveError, "Reverse X SIP matrix is not square.");
        }
        if (nBp * nBp != kBp.getSize()) {
            throw LSST_EXCEPT(afw::table::io::MalformedArchiveError, "Reverse Y SIP matrix is not square.");
        }
        Eigen::Map<Eigen::MatrixXd const> mapA((*sipRecord)[kA].getData(), nA, nA);
        _sipA = mapA;
        Eigen::Map<Eigen::MatrixXd const> mapB((*sipRecord)[kB].getData(), nB, nB);
        _sipB = mapB;
        Eigen::Map<Eigen::MatrixXd const> mapAp((*sipRecord)[kAp].getData(), nAp, nAp);
        _sipAp = mapAp;
        Eigen::Map<Eigen::MatrixXd const> mapBp((*sipRecord)[kBp].getData(), nBp, nBp);
        _sipBp = mapBp;
    }
}
}
}
}  // namespace lsst::afw::image
