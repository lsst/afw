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

double toMaggies(double instFlux, double scale) { return instFlux * scale; }

double toMagnitude(double instFlux, double scale) { return -2.5 * log10(instFlux * scale); }

double toMaggiesErr(double instFlux, double instFluxErr, double scale, double scaleErr, double maggies) {
    return maggies * hypot(instFluxErr / instFlux, scaleErr / scale);
}

double toMagnitudeErr(double instFlux, double instFluxErr, double scale, double scaleErr) {
    return 2.5 / log(10.0) * hypot(instFluxErr / instFlux, scaleErr / scale);
}

}  // anonymous namespace

// ------------------- Conversions to Maggies -------------------

double PhotoCalib::instFluxToMaggies(double instFlux, afw::geom::Point<double, 2> const &point) const {
    if (_isConstant)
        return toMaggies(instFlux, _calibrationMean);
    else
        return toMaggies(instFlux, _calibration->evaluate(point));
}

double PhotoCalib::instFluxToMaggies(double instFlux) const { return toMaggies(instFlux, _calibrationMean); }

Measurement PhotoCalib::instFluxToMaggies(double instFlux, double instFluxErr,
                                          afw::geom::Point<double, 2> const &point) const {
    double calibration, err, maggies;
    if (_isConstant)
        calibration = _calibrationMean;
    else
        calibration = _calibration->evaluate(point);
    maggies = toMaggies(instFlux, calibration);
    err = toMaggiesErr(instFlux, instFluxErr, calibration, _calibrationErr, maggies);
    return Measurement(maggies, err);
}

Measurement PhotoCalib::instFluxToMaggies(double instFlux, double instFluxErr) const {
    double maggies = toMaggies(instFlux, _calibrationMean);
    double err = toMaggiesErr(instFlux, instFluxErr, _calibrationMean, _calibrationErr, maggies);
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
    auto maggiesKey = sourceCatalog.getSchema().find<double>(outField + "_flux").key;
    auto maggiesErrKey = sourceCatalog.getSchema().find<double>(outField + "_fluxSigma").key;
    for (auto &record : sourceCatalog) {
        auto result =
                instFluxToMaggies(record.get(instFluxKey), record.get(instFluxErrKey), record.getCentroid());
        record.set(maggiesKey, result.value);
        record.set(maggiesErrKey, result.err);
    }
}

// ------------------- Conversions to Magnitudes -------------------

double PhotoCalib::instFluxToMagnitude(double instFlux, afw::geom::Point<double, 2> const &point) const {
    if (_isConstant)
        return toMagnitude(instFlux, _calibrationMean);
    else
        return toMagnitude(instFlux, _calibration->evaluate(point));
}

double PhotoCalib::instFluxToMagnitude(double instFlux) const {
    return toMagnitude(instFlux, _calibrationMean);
}

Measurement PhotoCalib::instFluxToMagnitude(double instFlux, double instFluxErr,
                                            afw::geom::Point<double, 2> const &point) const {
    double calibration, err, magnitude;
    if (_isConstant)
        calibration = _calibrationMean;
    else
        calibration = _calibration->evaluate(point);
    magnitude = toMagnitude(instFlux, calibration);
    err = toMagnitudeErr(instFlux, instFluxErr, calibration, _calibrationErr);
    return Measurement(magnitude, err);
}

Measurement PhotoCalib::instFluxToMagnitude(double instFlux, double instFluxErr) const {
    double magnitude = toMagnitude(instFlux, _calibrationMean);
    double err = toMagnitudeErr(instFlux, instFluxErr, _calibrationMean, _calibrationErr);
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
    for (auto &record : sourceCatalog) {
        auto result = instFluxToMagnitude(record.get(instFluxKey), record.get(instFluxErrKey),
                                          record.getCentroid());
        record.set(magKey, result.value);
        record.set(magErrKey, result.err);
    }
}

// ------------------- other utility methods -------------------

double PhotoCalib::magnitudeToInstFlux(double magnitude) const {
    return pow(10, magnitude / -2.5) / _calibrationMean;
}

std::shared_ptr<math::BoundedField> PhotoCalib::computeScaledCalibration() const {
    return *(_calibration) / _calibrationMean;
}

std::shared_ptr<math::BoundedField> PhotoCalib::computeScalingTo(std::shared_ptr<PhotoCalib> other) const {
    throw LSST_EXCEPT(pex::exceptions::LogicError, "Not Implemented: See DM-10154.");
}

bool PhotoCalib::operator==(PhotoCalib const &rhs) const {
    return (_calibrationMean == rhs._calibrationMean && _calibrationErr == rhs._calibrationErr &&
            (*_calibration) == *(rhs._calibration));
}

double PhotoCalib::computeCalibrationMean(std::shared_ptr<afw::math::BoundedField> calibration) const {
    return calibration->mean();
}

std::ostream &operator<<(std::ostream &os, PhotoCalib const &photoCalib) {
    if (photoCalib._isConstant)
        os << "spatially constant with ";
    else
        os << *(photoCalib._calibration) << " with ";
    return os << "mean: " << photoCalib._calibrationMean << " err: " << photoCalib._calibrationErr;
}

// ------------------- persistence -------------------

namespace {

class PhotoCalibSchema {
public:
    table::Schema schema;
    table::Key<double> calibrationMean;
    table::Key<double> calibrationErr;
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
              calibrationMean(schema.addField<double>(
                      "calibrationMean", "mean calibration on this PhotoCalib's domain", "count")),
              calibrationErr(
                      schema.addField<double>("calibrationErr", "1-sigma error on calibrationMean", "count")),
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
        return std::make_shared<PhotoCalib>(record.get(keys.calibrationMean), record.get(keys.calibrationErr),
                                            archive.get<afw::math::BoundedField>(record.get(keys.field)),
                                            record.get(keys.isConstant));
    }

    explicit PhotoCalibFactory(std::string const &name) : afw::table::io::PersistableFactory(name) {}
};

std::string getPhotoCalibPersistenceName() { return "PhotoCalib"; }

PhotoCalibFactory registration(getPhotoCalibPersistenceName());

}  // anonymous namespace

std::string PhotoCalib::getPersistenceName() const { return getPhotoCalibPersistenceName(); }

void PhotoCalib::write(OutputArchiveHandle &handle) const {
    PhotoCalibSchema const &keys = PhotoCalibSchema::get();
    table::BaseCatalog catalog = handle.makeCatalog(keys.schema);
    auto record = catalog.addNew();
    record->set(keys.calibrationMean, _calibrationMean);
    record->set(keys.calibrationErr, _calibrationErr);
    record->set(keys.isConstant, _isConstant);
    record->set(keys.field, handle.put(_calibration));
    handle.saveCatalog(catalog);
}

// ------------------- private/protected helpers -------------------

void PhotoCalib::instFluxToMaggiesArray(afw::table::SourceCatalog const &sourceCatalog,
                                        std::string const &instFluxField,
                                        ndarray::Array<double, 2, 2> result) const {
    double instFlux, instFluxErr, maggies, calibration;
    auto instFluxKey = sourceCatalog.getSchema().find<double>(instFluxField + "_flux").key;
    auto instFluxErrKey = sourceCatalog.getSchema().find<double>(instFluxField + "_fluxSigma").key;
    auto iter = result.begin();
    for (auto const &rec : sourceCatalog) {
        instFlux = rec.get(instFluxKey);
        instFluxErr = rec.get(instFluxErrKey);
        if (_isConstant)
            calibration = _calibrationMean;
        else
            calibration = _calibration->evaluate(rec.getCentroid());
        maggies = toMaggies(instFlux, calibration);
        (*iter)[0] = maggies;
        (*iter)[1] = toMaggiesErr(instFlux, instFluxErr, calibration, _calibrationErr, maggies);
        iter++;
    }
}

void PhotoCalib::instFluxToMagnitudeArray(afw::table::SourceCatalog const &sourceCatalog,
                                          std::string const &instFluxField,
                                          ndarray::Array<double, 2, 2> result) const {
    double instFlux, instFluxErr, calibration;
    auto instFluxKey = sourceCatalog.getSchema().find<double>(instFluxField + "_flux").key;
    auto instFluxErrKey = sourceCatalog.getSchema().find<double>(instFluxField + "_fluxSigma").key;
    auto iter = result.begin();
    for (auto const &rec : sourceCatalog) {
        instFlux = rec.get(instFluxKey);
        instFluxErr = rec.get(instFluxErrKey);
        if (_isConstant)
            calibration = _calibrationMean;
        else
            calibration = _calibration->evaluate(rec.getCentroid());
        (*iter)[0] = toMagnitude(instFlux, calibration);
        (*iter)[1] = toMagnitudeErr(instFlux, instFluxErr, calibration, _calibrationErr);
        iter++;
    }
}
}  // namespace image
}  // namespace afw
}  // namespace lsst
