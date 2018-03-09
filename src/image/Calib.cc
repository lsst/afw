// -*- lsst-c++ -*-

/*
 * LSST Data Management System
 * Copyright 2008-2016 LSST Corporation.
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
 * Classes to support calibration (e.g. photometric zero points, exposure times)
 */
#include <cmath>
#include <cstdint>
#include <string>

#include "boost/format.hpp"
#include "boost/algorithm/string/trim.hpp"

#include "ndarray.h"
#include "lsst/pex/exceptions.h"
#include "lsst/daf/base/PropertySet.h"
#include "lsst/afw/image/Calib.h"
#include "lsst/afw/cameraGeom/Detector.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/table/io/CatalogVector.h"

namespace lsst {
namespace afw {
namespace image {

/// Compute AB magnitude from flux in Janskys
template <typename T>
ndarray::Array<T, 1> abMagFromFlux(ndarray::Array<T const, 1> const& flux) {
    ndarray::Array<T, 1> out = ndarray::allocate(flux.getShape());
    for (std::size_t ii = 0; ii < flux.getNumElements(); ++ii) {
        out[ii] = abMagFromFlux(flux[ii]);
    }
    return out;
}

/// Compute AB magnitude error from flux and flux error in Janskys
template <typename T>
ndarray::Array<T, 1> abMagErrFromFluxErr(ndarray::Array<T const, 1> const& fluxErr,
                                         ndarray::Array<T const, 1> const& flux) {
    if (flux.getNumElements() != fluxErr.getNumElements()) {
        throw LSST_EXCEPT(pex::exceptions::LengthError,
                          (boost::format("Length mismatch: %d vs %d") % flux.getNumElements() %
                           fluxErr.getNumElements()).str());
    }
    ndarray::Array<T, 1> out = ndarray::allocate(flux.getShape());
    for (std::size_t ii = 0; ii < flux.getNumElements(); ++ii) {
        out[ii] = abMagErrFromFluxErr(fluxErr[ii], flux[ii]);
    }
    return out;
}

/// Compute flux in Janskys from AB magnitude
template <typename T>
ndarray::Array<T, 1> fluxFromABMag(ndarray::Array<T const, 1> const& mag) {
    ndarray::Array<T, 1> out = ndarray::allocate(mag.getShape());
    for (std::size_t ii = 0; ii < mag.getNumElements(); ++ii) {
        out[ii] = fluxFromABMag(mag[ii]);
    }
    return out;
}

/// Compute flux error in Janskys from AB magnitude error and AB magnitude
template <typename T>
ndarray::Array<T, 1> fluxErrFromABMagErr(ndarray::Array<T const, 1> const& magErr,
                                         ndarray::Array<T const, 1> const& mag) {
    if (mag.getNumElements() != magErr.getNumElements()) {
        throw LSST_EXCEPT(pex::exceptions::LengthError,
                          (boost::format("Length mismatch: %d vs %d") % mag.getNumElements() %
                           magErr.getNumElements()).str());
    }
    ndarray::Array<T, 1> out = ndarray::allocate(mag.getShape());
    for (std::size_t ii = 0; ii < mag.getNumElements(); ++ii) {
        out[ii] = fluxErrFromABMagErr(magErr[ii], mag[ii]);
    }
    return out;
}


Calib::Calib() : _fluxMag0(0.0), _fluxMag0Sigma(0.0) {}
Calib::Calib(double fluxMag0) : _fluxMag0(fluxMag0), _fluxMag0Sigma(0.0) {}
Calib::Calib(std::vector<std::shared_ptr<Calib const>> const& calibs) : _fluxMag0(0.0), _fluxMag0Sigma(0.0) {
    if (calibs.empty()) {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError,
                          "You must provide at least one input Calib");
    }

    double const fluxMag00 = calibs[0]->_fluxMag0;
    double const fluxMag0Sigma0 = calibs[0]->_fluxMag0Sigma;

    for (std::vector<std::shared_ptr<Calib const>>::const_iterator ptr = calibs.begin(); ptr != calibs.end();
         ++ptr) {
        Calib const& calib = **ptr;

        if (::fabs(fluxMag00 - calib._fluxMag0) > std::numeric_limits<double>::epsilon() ||
            ::fabs(fluxMag0Sigma0 - calib._fluxMag0Sigma) > std::numeric_limits<double>::epsilon()) {
            throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError,
                              (boost::format("You may only combine calibs with the same fluxMag0: "
                                             "%g +- %g v %g +- %g") %
                               calib.getFluxMag0().first % calib.getFluxMag0().second %
                               calibs[0]->getFluxMag0().first % calibs[0]->getFluxMag0().second)
                                      .str());
        }
    }
}

Calib::Calib(std::shared_ptr<lsst::daf::base::PropertySet const> metadata) {
    double fluxMag0 = 0.0, fluxMag0Sigma = 0.0;

    auto key = "FLUXMAG0";
    if (metadata->exists(key)) {
        fluxMag0 = metadata->getAsDouble(key);

        key = "FLUXMAG0ERR";
        if (metadata->exists(key)) {
            fluxMag0Sigma = metadata->getAsDouble(key);
        }
    }

    _fluxMag0 = fluxMag0;
    _fluxMag0Sigma = fluxMag0Sigma;
}
bool Calib::_throwOnNegativeFlux = true;
void Calib::setThrowOnNegativeFlux(bool raiseException) { _throwOnNegativeFlux = raiseException; }

bool Calib::getThrowOnNegativeFlux() { return _throwOnNegativeFlux; }

Calib::Calib(Calib const&) = default;
Calib::Calib(Calib&&) = default;
Calib& Calib::operator=(Calib const&) = default;
Calib& Calib::operator=(Calib&&) = default;
Calib::~Calib() = default;

namespace detail {
int stripCalibKeywords(std::shared_ptr<lsst::daf::base::PropertySet> metadata) {
    int nstripped = 0;

    auto key = "FLUXMAG0";
    if (metadata->exists(key)) {
        metadata->remove(key);
        nstripped++;
    }

    key = "FLUXMAG0ERR";
    if (metadata->exists(key)) {
        metadata->remove(key);
        nstripped++;
    }

    return nstripped;
}
}  // namespace detail

bool Calib::operator==(Calib const& rhs) const {
    return _fluxMag0 == rhs._fluxMag0 && _fluxMag0Sigma == rhs._fluxMag0Sigma;
}

void Calib::setFluxMag0(double fluxMag0, double fluxMag0Sigma) {
    _fluxMag0 = fluxMag0;
    _fluxMag0Sigma = fluxMag0Sigma;
}
void Calib::setFluxMag0(std::pair<double, double> fluxMag0AndSigma) {
    _fluxMag0 = fluxMag0AndSigma.first;
    _fluxMag0Sigma = fluxMag0AndSigma.second;
}

std::pair<double, double> Calib::getFluxMag0() const { return std::make_pair(_fluxMag0, _fluxMag0Sigma); }

namespace {

inline void checkNegativeFlux0(double fluxMag0) {
    if (fluxMag0 <= 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::DomainError,
                          (boost::format("Flux of 0-mag object must be >= 0: saw %g") % fluxMag0).str());
    }
}
inline bool isNegativeFlux(double flux, bool doThrow) {
    if (flux <= 0) {
        if (doThrow) {
            throw LSST_EXCEPT(lsst::pex::exceptions::DomainError,
                              (boost::format("Flux must be >= 0: saw %g") % flux).str());
        }
        return true;
    }
    return false;
}
inline double convertToFlux(double fluxMag0, double mag) { return fluxMag0 * ::pow(10.0, -0.4 * mag); }
inline double convertToFluxErr(double fluxMag0InvSNR, double flux, double magErr) {
    // Want to:
    //     return flux * hypot(_fluxMag0Sigma/_fluxMag0, 0.4*std::log(10)*magSigma/mag);
    // But hypot is not standard C++ so use <http://en.wikipedia.org/wiki/Hypot#Implementation>
    double a = fluxMag0InvSNR;
    double b = 0.4 * std::log(10.0) * magErr;
    if (std::abs(a) < std::abs(b)) {
        std::swap(a, b);
    }
    return flux * std::abs(a) * std::sqrt(1 + std::pow(b / a, 2));
}
inline double convertToMag(double fluxMag0, double flux) { return -2.5 * ::log10(flux / fluxMag0); }

inline void convertToMagWithErr(double* mag, double* magErr, double fluxMag0, double fluxMag0Err,
                                double flux, double fluxErr) {
    *mag = -2.5*std::log10(flux/fluxMag0);
    double const x = fluxErr/flux;
    double const y = fluxMag0Err/fluxMag0;
    *magErr = (2.5/std::log(10.0))*std::sqrt(x*x + y*y);
}

}  // anonymous namespace

double Calib::getFlux(double const mag) const {
    checkNegativeFlux0(_fluxMag0);
    return convertToFlux(_fluxMag0, mag);
}
ndarray::Array<double, 1> Calib::getFlux(ndarray::Array<double const, 1> const& mag) const {
    checkNegativeFlux0(_fluxMag0);
    ndarray::Array<double, 1> flux = ndarray::allocate(mag.size());
    ndarray::Array<double const, 1>::Iterator inIter = mag.begin();
    ndarray::Array<double, 1>::Iterator outIter = flux.begin();
    for (; inIter != mag.end(); ++inIter, ++outIter) {
        *outIter = convertToFlux(_fluxMag0, *inIter);
    }
    return flux;
}

std::pair<double, double> Calib::getFlux(double const mag, double const magSigma) const {
    checkNegativeFlux0(_fluxMag0);
    double const flux = convertToFlux(_fluxMag0, mag);
    double const fluxErr = convertToFluxErr(_fluxMag0Sigma / _fluxMag0, flux, magSigma);
    return std::make_pair(flux, fluxErr);
}

std::pair<ndarray::Array<double, 1>, ndarray::Array<double, 1>> Calib::getFlux(
        ndarray::Array<double const, 1> const& mag, ndarray::Array<double const, 1> const& magErr) const {
    checkNegativeFlux0(_fluxMag0);
    if (mag.size() != magErr.size()) {
        throw LSST_EXCEPT(
                lsst::pex::exceptions::LengthError,
                (boost::format("Size of mag (%d) and magErr (%d) don't match") % mag.size() % magErr.size())
                        .str());
    }

    ndarray::Array<double, 1> flux = ndarray::allocate(mag.size());
    ndarray::Array<double, 1> fluxErr = ndarray::allocate(mag.size());
    ndarray::Array<double const, 1>::Iterator magIter = mag.begin();
    ndarray::Array<double const, 1>::Iterator magErrIter = magErr.begin();
    ndarray::Array<double, 1>::Iterator fluxIter = flux.begin();
    ndarray::Array<double, 1>::Iterator fluxErrIter = fluxErr.begin();

    double fluxMag0InvSNR = _fluxMag0Sigma / _fluxMag0;
    for (; magIter != mag.end(); ++magIter, ++magErrIter, ++fluxIter, ++fluxErrIter) {
        *fluxIter = convertToFlux(_fluxMag0, *magIter);
        *fluxErrIter = convertToFluxErr(fluxMag0InvSNR, *fluxIter, *magErrIter);
    }

    return std::make_pair(flux, fluxErr);
}

double Calib::getMagnitude(double const flux) const {
    checkNegativeFlux0(_fluxMag0);
    if (isNegativeFlux(flux, Calib::getThrowOnNegativeFlux())) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return convertToMag(_fluxMag0, flux);
}

std::pair<double, double> Calib::getMagnitude(double const flux, double const fluxErr) const {
    checkNegativeFlux0(_fluxMag0);
    if (isNegativeFlux(flux, Calib::getThrowOnNegativeFlux())) {
        double const NaN = std::numeric_limits<double>::quiet_NaN();
        return std::make_pair(NaN, NaN);
    }

    double mag, magErr;
    convertToMagWithErr(&mag, &magErr, _fluxMag0, _fluxMag0Sigma, flux, fluxErr);
    return std::make_pair(mag, magErr);
}

ndarray::Array<double, 1> Calib::getMagnitude(ndarray::Array<double const, 1> const& flux) const {
    checkNegativeFlux0(_fluxMag0);
    ndarray::Array<double, 1> mag = ndarray::allocate(flux.size());
    ndarray::Array<double const, 1>::Iterator fluxIter = flux.begin();
    ndarray::Array<double, 1>::Iterator magIter = mag.begin();
    int nonPositive = 0;
    for (; fluxIter != flux.end(); ++fluxIter, ++magIter) {
        if (isNegativeFlux(*fluxIter, false)) {
            ++nonPositive;
            *magIter = std::numeric_limits<double>::quiet_NaN();
            continue;
        }
        *magIter = convertToMag(_fluxMag0, *fluxIter);
    }
    if (nonPositive && Calib::getThrowOnNegativeFlux()) {
        throw LSST_EXCEPT(lsst::pex::exceptions::DomainError,
                          (boost::format("Flux must be >= 0: %d non-positive seen") % nonPositive).str());
    }
    return mag;
}

std::pair<ndarray::Array<double, 1>, ndarray::Array<double, 1>> Calib::getMagnitude(
        ndarray::Array<double const, 1> const& flux, ndarray::Array<double const, 1> const& fluxErr) const {
    checkNegativeFlux0(_fluxMag0);
    if (flux.size() != fluxErr.size()) {
        throw LSST_EXCEPT(lsst::pex::exceptions::LengthError,
                          (boost::format("Size of flux (%d) and fluxErr (%d) don't match") % flux.size() %
                           fluxErr.size())
                                  .str());
    }

    ndarray::Array<double, 1> mag = ndarray::allocate(flux.size());
    ndarray::Array<double, 1> magErr = ndarray::allocate(flux.size());
    ndarray::Array<double const, 1>::Iterator fluxIter = flux.begin();
    ndarray::Array<double const, 1>::Iterator fluxErrIter = fluxErr.begin();
    ndarray::Array<double, 1>::Iterator magIter = mag.begin();
    ndarray::Array<double, 1>::Iterator magErrIter = magErr.begin();
    int nonPositive = 0;
    for (; fluxIter != flux.end(); ++fluxIter, ++fluxErrIter, ++magIter, ++magErrIter) {
        if (isNegativeFlux(*fluxIter, false)) {
            ++nonPositive;
            double const NaN = std::numeric_limits<double>::quiet_NaN();
            *magIter = NaN;
            *magErrIter = NaN;
            continue;
        }
        double f, df;
        convertToMagWithErr(&f, &df, _fluxMag0, _fluxMag0Sigma, *fluxIter, *fluxErrIter);
        *magIter = f;
        *magErrIter = df;
    }
    if (nonPositive && Calib::getThrowOnNegativeFlux()) {
        throw LSST_EXCEPT(lsst::pex::exceptions::DomainError,
                          (boost::format("Flux must be >= 0: %d non-positive seen") % nonPositive).str());
    }
    return std::make_pair(mag, magErr);
}

namespace {

int const CALIB_TABLE_CURRENT_VERSION = 2;         // current version of ExposureTable
std::string const EXPTIME_FIELD_NAME = "exptime";  // name of exposure time field

class CalibKeys {
public:
    table::Schema schema;
    table::Key<std::int64_t> midTime;
    table::Key<double> expTime;
    table::Key<double> fluxMag0;
    table::Key<double> fluxMag0Sigma;

    // No copying
    CalibKeys(CalibKeys const &) = delete;
    CalibKeys& operator=(CalibKeys const &) = delete;

    // No moving
    CalibKeys(CalibKeys&&) = delete;
    CalibKeys& operator=(CalibKeys&&) = delete;

    explicit CalibKeys(int tableVersion = CALIB_TABLE_CURRENT_VERSION)
            : schema(), midTime(), expTime(), fluxMag0(), fluxMag0Sigma() {
        if (tableVersion == 1) {
            // obsolete fields
            midTime = schema.addField<std::int64_t>(
                    "midtime", "middle of the time of the exposure relative to Unix epoch", "ns");
            expTime = schema.addField<double>(EXPTIME_FIELD_NAME, "exposure time", "s");
        }
        fluxMag0 = schema.addField<double>("fluxmag0", "flux of a zero-magnitude object", "count");
        fluxMag0Sigma = schema.addField<double>("fluxmag0.err", "1-sigma error on fluxmag0", "count");
    }
};

class CalibFactory : public table::io::PersistableFactory {
public:
    std::shared_ptr<table::io::Persistable> read(InputArchive const& archive,
                                                         CatalogVector const& catalogs) const override {
        // table version is not persisted, so we don't have a clean way to determine the version;
        // the hack is version = 1 if exptime found, else current
        int tableVersion = 1;
        try {
            catalogs.front().getSchema().find<double>(EXPTIME_FIELD_NAME);
        } catch (pex::exceptions::NotFoundError) {
            tableVersion = CALIB_TABLE_CURRENT_VERSION;
        }

        CalibKeys const keys{tableVersion};
        LSST_ARCHIVE_ASSERT(catalogs.size() == 1u);
        LSST_ARCHIVE_ASSERT(catalogs.front().size() == 1u);
        LSST_ARCHIVE_ASSERT(catalogs.front().getSchema() == keys.schema);
        table::BaseRecord const& record = catalogs.front().front();
        std::shared_ptr<Calib> result(new Calib());
        result->setFluxMag0(record.get(keys.fluxMag0), record.get(keys.fluxMag0Sigma));
        return result;
    }

    explicit CalibFactory(std::string const& name) : table::io::PersistableFactory(name) {}
};

std::string getCalibPersistenceName() { return "Calib"; }

CalibFactory registration(getCalibPersistenceName());

}  // namespace

std::string Calib::getPersistenceName() const { return getCalibPersistenceName(); }

void Calib::write(OutputArchiveHandle& handle) const {
    CalibKeys const keys{};
    table::BaseCatalog cat = handle.makeCatalog(keys.schema);
    std::shared_ptr<table::BaseRecord> record = cat.addNew();
    std::pair<double, double> fluxMag0 = getFluxMag0();
    record->set(keys.fluxMag0, fluxMag0.first);
    record->set(keys.fluxMag0Sigma, fluxMag0.second);
    handle.saveCatalog(cat);
}

Calib& Calib::operator*=(double const scale) {
    _fluxMag0 *= scale;
    _fluxMag0Sigma *= scale;
    return *this;
}


// Explicit instantiation
#define INSTANTIATE(TYPE) \
template ndarray::Array<TYPE, 1> abMagFromFlux(ndarray::Array<TYPE const, 1> const& flux); \
template ndarray::Array<TYPE, 1> abMagErrFromFluxErr(ndarray::Array<TYPE const, 1> const& fluxErr, \
                                                     ndarray::Array<TYPE const, 1> const& flux); \
template ndarray::Array<TYPE, 1> fluxFromABMag(ndarray::Array<TYPE const, 1> const& mag); \
template ndarray::Array<TYPE, 1> fluxErrFromABMagErr(ndarray::Array<TYPE const, 1> const& magErr, \
                                                     ndarray::Array<TYPE const, 1> const& mag);

INSTANTIATE(float);
INSTANTIATE(double);

}  // namespace image
}  // namespace afw
}  // namespace lsst
