/*
 * LSST Data Management System
 * Copyright 2017 LSST Corporation.
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

#include <cmath>

#include "lsst/afw/geom/Point.h"
#include "lsst/afw/image/PhotoCalib.h"
#include "lsst/afw/math/BoundedField.h"
#include "lsst/afw/table/Source.h"
#include "lsst/afw/table/io/CatalogVector.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/daf/base/PropertySet.h"
#include "lsst/pex/exceptions.h"
#include "ndarray.h"

namespace lsst {
namespace afw {
namespace image {

// ------------------- helpers -------------------

namespace {

double pow2(double value) { return value * value; }

double toMaggies(double instFlux, double scale) { return instFlux / scale; }

double toMagnitude(double instFlux, double scale) { return -2.5 * log10(instFlux / scale); }

double toMaggiesErr(double instFlux, double instFluxErr, double scale, double scaleErr, double maggies) {
    return maggies * sqrt(pow2(instFluxErr / instFlux) + pow2(scaleErr / scale));
}

double toMagnitudeErr(double instFlux, double instFluxErr, double scale, double scaleErr) {
    return 2.5 / log(10.0) * sqrt(pow2(instFluxErr / instFlux) + pow2(scaleErr / scale));
}

}  // anonymous namespace

// ------------------- Conversions to Maggies -------------------

double PhotoCalib::instFluxToMaggies(double instFlux, afw::geom::Point<double, 2> const &point) const {
    if (_isConstant)
        return toMaggies(instFlux, _instFluxMag0);
    else
        return toMaggies(instFlux, _zeroPoint->evaluate(point));
}

double PhotoCalib::instFluxToMaggies(double instFlux) const { return toMaggies(instFlux, _instFluxMag0); }

Measurement PhotoCalib::instFluxToMaggies(double instFlux, double instFluxErr,
                                          afw::geom::Point<double, 2> const &point) const {
    double instFluxMag0, err, maggies;
    if (_isConstant)
        instFluxMag0 = _instFluxMag0;
    else
        instFluxMag0 = _zeroPoint->evaluate(point);
    maggies = toMaggies(instFlux, instFluxMag0);
    err = toMaggiesErr(instFlux, instFluxErr, instFluxMag0, _instFluxMag0Err, maggies);
    return Measurement(maggies, err);
}

Measurement PhotoCalib::instFluxToMaggies(double instFlux, double instFluxErr) const {
    double maggies = toMaggies(instFlux, _instFluxMag0);
    double err = toMaggiesErr(instFlux, instFluxErr, _instFluxMag0, _instFluxMag0Err, maggies);
    return Measurement(maggies, err);
}

Measurement PhotoCalib::instFluxToMaggies(afw::table::SourceRecord const &sourceRecord,
                                          std::string const &instFluxField) const {
    auto position = sourceRecord.getCentroid();
    auto instFluxKey = sourceRecord.getSchema().find<double>(instFluxField + "_flux").key;
    auto instFluxErrKey = sourceRecord.getSchema().find<double>(instFluxField + "_fluxSigma").key;
    return instFluxToMaggies(sourceRecord.get(instFluxKey), sourceRecord.get(instFluxErrKey), position);
}
ndarray::Array<double, 2, 2> PhotoCalib::instFluxToMaggies(afw::table::SourceCatalog const &sourceCatalog,
                                                           std::string const &instFluxField) const {
    ndarray::Array<double, 2, 2> result =
            ndarray::allocate(ndarray::makeVector(int(sourceCatalog.size()), 2));
    instFluxToMaggiesArray(sourceCatalog, instFluxField, result);
    return result;
}

void PhotoCalib::instFluxToMaggies(afw::table::SourceCatalog &sourceCatalog, std::string const &instFluxField,
                                   std::string const &outField) const {
    auto instFluxKey = sourceCatalog.getSchema().find<double>(instFluxField + "_flux").key;
    auto instFluxErrKey = sourceCatalog.getSchema().find<double>(instFluxField + "_fluxSigma").key;
    auto maggiesKey = sourceCatalog.getSchema().find<double>(outField + "_calFlux").key;
    auto maggiesErrKey = sourceCatalog.getSchema().find<double>(outField + "_calFluxErr").key;
    for (auto & record : sourceCatalog) {
        auto result = instFluxToMaggies(record.get(instFluxKey), record.get(instFluxErrKey),
                                        record.getCentroid());
        record.set(maggiesKey, result.value);
        record.set(maggiesErrKey, result.err);
    }
}

// ------------------- Conversions to Magnitudes -------------------

double PhotoCalib::instFluxToMagnitude(double instFlux, afw::geom::Point<double, 2> const &point) const {
    if (_isConstant)
        return toMagnitude(instFlux, _instFluxMag0);
    else
        return toMagnitude(instFlux, _zeroPoint->evaluate(point));
}

double PhotoCalib::instFluxToMagnitude(double instFlux) const { return toMagnitude(instFlux, _instFluxMag0); }

Measurement PhotoCalib::instFluxToMagnitude(double instFlux, double instFluxErr,
                                            afw::geom::Point<double, 2> const &point) const {
    double instFluxMag0, err, magnitude;
    if (_isConstant)
        instFluxMag0 = _instFluxMag0;
    else
        instFluxMag0 = _zeroPoint->evaluate(point);
    magnitude = toMagnitude(instFlux, instFluxMag0);
    err = toMagnitudeErr(instFlux, instFluxErr, instFluxMag0, _instFluxMag0Err);
    return Measurement(magnitude, err);
}

Measurement PhotoCalib::instFluxToMagnitude(double instFlux, double instFluxErr) const {
    double magnitude = toMagnitude(instFlux, _instFluxMag0);
    double err = toMagnitudeErr(instFlux, instFluxErr, _instFluxMag0, _instFluxMag0Err);
    return Measurement(magnitude, err);
}

Measurement PhotoCalib::instFluxToMagnitude(afw::table::SourceRecord const &sourceRecord,
                                            std::string const &instFluxField) const {
    auto position = sourceRecord.getCentroid();
    auto instFluxKey = sourceRecord.getSchema().find<double>(instFluxField + "_flux").key;
    auto instFluxErrKey = sourceRecord.getSchema().find<double>(instFluxField + "_fluxSigma").key;
    return instFluxToMagnitude(sourceRecord.get(instFluxKey), sourceRecord.get(instFluxErrKey), position);
}

ndarray::Array<double, 2, 2> PhotoCalib::instFluxToMagnitude(afw::table::SourceCatalog const &sourceCatalog,
                                                             std::string const &instFluxField) const {
    ndarray::Array<double, 2, 2> result =
            ndarray::allocate(ndarray::makeVector(int(sourceCatalog.size()), 2));
    instFluxToMagnitudeArray(sourceCatalog, instFluxField, result);
    return result;
}

void PhotoCalib::instFluxToMagnitude(afw::table::SourceCatalog &sourceCatalog,
                                     std::string const &instFluxField, std::string const &outField) const {
    auto instFluxKey = sourceCatalog.getSchema().find<double>(instFluxField + "_flux").key;
    auto instFluxErrKey = sourceCatalog.getSchema().find<double>(instFluxField + "_fluxSigma").key;
    auto magKey = sourceCatalog.getSchema().find<double>(outField + "_mag").key;
    auto magErrKey = sourceCatalog.getSchema().find<double>(outField + "_magErr").key;
    for (auto & record : sourceCatalog) {
        auto result = instFluxToMagnitude(record.get(instFluxKey), record.get(instFluxErrKey),
                                          record.getCentroid());
        record.set(magKey, result.value);
        record.set(magErrKey, result.err);
    }
}

// ------------------- other utility methods -------------------

double PhotoCalib::magnitudeToInstFlux(double magnitude) const {
    return pow(10, magnitude / -2.5) * _instFluxMag0;
}

std::shared_ptr<math::BoundedField> PhotoCalib::computeScaledZeroPoint() const {
    return *(_zeroPoint) / _instFluxMag0;
}

std::shared_ptr<math::BoundedField> PhotoCalib::computeScalingTo(std::shared_ptr<PhotoCalib> other) const {
    throw LSST_EXCEPT(pex::exceptions::LogicError, "Not Implemented: See DM-10154.");
}

bool PhotoCalib::operator==(PhotoCalib const &rhs) const {
    return (_instFluxMag0 == rhs._instFluxMag0 && _instFluxMag0Err == rhs._instFluxMag0Err &&
            (*_zeroPoint) == *(rhs._zeroPoint));
}

double PhotoCalib::computeInstFluxMag0(std::shared_ptr<afw::math::BoundedField> zeroPoint) const {
    return zeroPoint->mean();
}

std::ostream &operator<<(std::ostream &os, PhotoCalib const &photoCalib) {
    if (photoCalib._isConstant)
        os << "spatially constant with ";
    else
        os << *(photoCalib._zeroPoint) << " with ";
    return os << "mean: " << photoCalib._instFluxMag0 << " err: " << photoCalib._instFluxMag0Err;
}

// ------------------- persistence -------------------

namespace {

class PhotoCalibSchema {
public:
    table::Schema schema;
    table::Key<double> instFluxMag0;
    table::Key<double> instFluxMag0Err;
    table::Key<table::Flag> isConstant;
    table::Key<int> field;

    // No copying
    PhotoCalibSchema(PhotoCalibSchema const &) = delete;
    PhotoCalibSchema &operator=(PhotoCalibSchema const &) = delete;
    // No moving
    PhotoCalibSchema(PhotoCalibSchema &&) = delete;
    PhotoCalibSchema &operator=(PhotoCalibSchema &&) = delete;

    static PhotoCalibSchema const &get() {
        static PhotoCalibSchema const instance;
        return instance;
    }

private:
    PhotoCalibSchema()
            : schema(),
              instFluxMag0(schema.addField<double>("instFluxMag0", "instFlux of a zero-magnitude object",
                                                   "count")),
              instFluxMag0Err(
                      schema.addField<double>("instFluxMag0Err", "1-err error on instFluxmag0", "count")),
              isConstant(schema.addField<table::Flag>("isConstant", "Is this spatially-constant?")),
              field(schema.addField<int>("field", "archive ID of the BoundedField object")) {
        schema.getCitizen().markPersistent();
    }
};

class PhotoCalibFactory : public table::io::PersistableFactory {
public:
    virtual PTR(table::io::Persistable)
            read(InputArchive const &archive, CatalogVector const &catalogs) const {
        table::BaseRecord const &record = catalogs.front().front();
        PhotoCalibSchema const &keys = PhotoCalibSchema::get();
        return std::make_shared<PhotoCalib>(record.get(keys.instFluxMag0), record.get(keys.instFluxMag0Err),
                                            archive.get<afw::math::BoundedField>(record.get(keys.field)),
                                            record.get(keys.isConstant));
    }

    PhotoCalibFactory(std::string const &name) : afw::table::io::PersistableFactory(name) {}
};

std::string getPhotoCalibPersistenceName() { return "PhotoCalib"; }

PhotoCalibFactory registration(getPhotoCalibPersistenceName());

}  // anonymous namespace

std::string PhotoCalib::getPersistenceName() const { return getPhotoCalibPersistenceName(); }

void PhotoCalib::write(OutputArchiveHandle &handle) const {
    PhotoCalibSchema const &keys = PhotoCalibSchema::get();
    table::BaseCatalog catalog = handle.makeCatalog(keys.schema);
    auto record = catalog.addNew();
    record->set(keys.instFluxMag0, _instFluxMag0);
    record->set(keys.instFluxMag0Err, _instFluxMag0Err);
    record->set(keys.isConstant, _isConstant);
    record->set(keys.field, handle.put(_zeroPoint));
    handle.saveCatalog(catalog);
}

// ------------------- private/protected helpers -------------------

void PhotoCalib::instFluxToMaggiesArray(afw::table::SourceCatalog const &sourceCatalog,
                                        std::string const &instFluxField,
                                        ndarray::Array<double, 2, 2> result) const {
    double instFlux, instFluxErr, maggies, instFluxMag0;
    auto instFluxKey = sourceCatalog.getSchema().find<double>(instFluxField + "_flux").key;
    auto instFluxErrKey = sourceCatalog.getSchema().find<double>(instFluxField + "_fluxSigma").key;
    auto iter = result.begin();
    for (auto const &rec : sourceCatalog) {
        instFlux = rec.get(instFluxKey);
        instFluxErr = rec.get(instFluxErrKey);
        if (_isConstant)
            instFluxMag0 = _instFluxMag0;
        else
            instFluxMag0 = _zeroPoint->evaluate(rec.getCentroid());
        maggies = toMaggies(instFlux, instFluxMag0);
        (*iter)[0] = maggies;
        (*iter)[1] = toMaggiesErr(instFlux, instFluxErr, instFluxMag0, _instFluxMag0Err, maggies);
        iter++;
    }
}

void PhotoCalib::instFluxToMagnitudeArray(afw::table::SourceCatalog const &sourceCatalog,
                                          std::string const &instFluxField,
                                          ndarray::Array<double, 2, 2> result) const {
    double instFlux, instFluxErr, instFluxMag0;
    auto instFluxKey = sourceCatalog.getSchema().find<double>(instFluxField + "_flux").key;
    auto instFluxErrKey = sourceCatalog.getSchema().find<double>(instFluxField + "_fluxSigma").key;
    auto iter = result.begin();
    for (auto const &rec : sourceCatalog) {
        instFlux = rec.get(instFluxKey);
        instFluxErr = rec.get(instFluxErrKey);
        if (_isConstant)
            instFluxMag0 = _instFluxMag0;
        else
            instFluxMag0 = _zeroPoint->evaluate(rec.getCentroid());
        (*iter)[0] = toMagnitude(instFlux, instFluxMag0);
        (*iter)[1] = toMagnitudeErr(instFlux, instFluxErr, instFluxMag0, _instFluxMag0Err);
        iter++;
    }
}
}
}
}
