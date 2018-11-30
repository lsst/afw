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

#include "lsst/geom/Point.h"
#include "lsst/afw/image/PhotoCalib.h"
#include "lsst/afw/math/BoundedField.h"
#include "lsst/afw/table/Source.h"
#include "lsst/afw/table/io/CatalogVector.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/daf/base/PropertySet.h"
#include "lsst/pex/exceptions.h"
#include "ndarray.h"
#include "lsst/afw/table/io/Persistable.cc"

namespace lsst {
namespace afw {

template std::shared_ptr<image::PhotoCalib> table::io::PersistableFacade<image::PhotoCalib>::dynamicCast(
        std::shared_ptr<table::io::Persistable> const &);

namespace image {

// ------------------- helpers -------------------

namespace {

int const SERIALIZATION_VERSION = 0;

double toMaggies(double instFlux, double scale) { return instFlux * scale; }

double toMagnitude(double instFlux, double scale) { return -2.5 * log10(instFlux * scale); }

double fromMagnitude(double magnitude, double scale) { return pow(10, magnitude / -2.5) / scale; }

double toMaggiesErr(double instFlux, double instFluxErr, double scale, double scaleErr, double maggies) {
    return maggies * hypot(instFluxErr / instFlux, scaleErr / scale);
}

/**
 * Compute the variance of an array of fluxes, for calculations on MaskedImages.
 *
 * MaskedImage stores the variance instead of the standard deviation, so we can skip a sqrt().
 * Usage in calibrateImage() is to compute maggies directly, so that the calibration scale is
 * `maggies/instFlux` (and thus not passed as an argument).
 *
 * @param instFlux[in] The instrumental flux.
 * @param instFluxVar[in] The variance of the instrumental fluxes.
 * @param scaleErr[in] The error on the calibration scale.
 * @param maggies[in] The physical fluxes calculated from the instrumental fluxes.
 * @param out[out] The output array to fill with the variance values.
 */
void toMaggiesVariance(ndarray::Array<float const, 2, 1> const &instFlux,
                       ndarray::Array<float const, 2, 1> const &instFluxVar, float scaleErr,
                       ndarray::Array<float const, 2, 1> const &maggies, ndarray::Array<float, 2, 1> out) {
    auto eigenMaggies = ndarray::asEigen<Eigen::ArrayXpr>(maggies);
    auto eigenInstFluxVar = ndarray::asEigen<Eigen::ArrayXpr>(instFluxVar);
    auto eigenInstFlux = ndarray::asEigen<Eigen::ArrayXpr>(instFlux);
    auto eigenOut = ndarray::asEigen<Eigen::ArrayXpr>(out);
    eigenOut = eigenMaggies.square() * (eigenInstFluxVar / eigenInstFlux.square() +
                                        (scaleErr / eigenMaggies * eigenInstFlux).square());
}

double toMagnitudeErr(double instFlux, double instFluxErr, double scale, double scaleErr) {
    return 2.5 / log(10.0) * hypot(instFluxErr / instFlux, scaleErr / scale);
}

}  // anonymous namespace

// ------------------- Conversions to Maggies -------------------

double PhotoCalib::instFluxToMaggies(double instFlux, lsst::geom::Point<double, 2> const &point) const {
    return toMaggies(instFlux, evaluate(point));
}

double PhotoCalib::instFluxToMaggies(double instFlux) const { return toMaggies(instFlux, _calibrationMean); }

Measurement PhotoCalib::instFluxToMaggies(double instFlux, double instFluxErr,
                                          lsst::geom::Point<double, 2> const &point) const {
    double calibration, err, maggies;
    calibration = evaluate(point);
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
    auto instFluxKey = sourceRecord.getSchema().find<double>(instFluxField + "_instFlux").key;
    auto instFluxErrKey = sourceRecord.getSchema().find<double>(instFluxField + "_instFluxErr").key;
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
    auto instFluxKey = sourceCatalog.getSchema().find<double>(instFluxField + "_instFlux").key;
    auto instFluxErrKey = sourceCatalog.getSchema().find<double>(instFluxField + "_instFluxErr").key;
    auto maggiesKey = sourceCatalog.getSchema().find<double>(outField + "_instFlux").key;
    auto maggiesErrKey = sourceCatalog.getSchema().find<double>(outField + "_instFluxErr").key;
    for (auto &record : sourceCatalog) {
        auto result =
                instFluxToMaggies(record.get(instFluxKey), record.get(instFluxErrKey), record.getCentroid());
        record.set(maggiesKey, result.value);
        record.set(maggiesErrKey, result.err);
    }
}

// ------------------- Conversions to Magnitudes -------------------

double PhotoCalib::instFluxToMagnitude(double instFlux, lsst::geom::Point<double, 2> const &point) const {
    return toMagnitude(instFlux, evaluate(point));
}

double PhotoCalib::instFluxToMagnitude(double instFlux) const {
    return toMagnitude(instFlux, _calibrationMean);
}

Measurement PhotoCalib::instFluxToMagnitude(double instFlux, double instFluxErr,
                                            lsst::geom::Point<double, 2> const &point) const {
    double calibration, err, magnitude;
    calibration = evaluate(point);
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
    auto instFluxKey = sourceRecord.getSchema().find<double>(instFluxField + "_instFlux").key;
    auto instFluxErrKey = sourceRecord.getSchema().find<double>(instFluxField + "_instFluxErr").key;
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
    auto instFluxKey = sourceCatalog.getSchema().find<double>(instFluxField + "_instFlux").key;
    auto instFluxErrKey = sourceCatalog.getSchema().find<double>(instFluxField + "_instFluxErr").key;
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
    return fromMagnitude(magnitude, _calibrationMean);
}

double PhotoCalib::magnitudeToInstFlux(double magnitude, lsst::geom::Point<double, 2> const &point) const {
    return fromMagnitude(magnitude, evaluate(point));
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

MaskedImage<float> PhotoCalib::calibrateImage(MaskedImage<float> const &maskedImage,
                                              bool includeScaleUncertainty) const {
    // Deep copy construct, as we're mutiplying in-place.
    auto result = MaskedImage<float>(maskedImage, true);

    if (_isConstant) {
        *(result.getImage()) *= _calibrationMean;
    } else {
        _calibration->multiplyImage(*(result.getImage()), true);  // only in the overlap region
    }
    if (includeScaleUncertainty) {
        toMaggiesVariance(maskedImage.getImage()->getArray(), maskedImage.getVariance()->getArray(),
                          _calibrationErr, result.getImage()->getArray(), result.getVariance()->getArray());
    } else {
        toMaggiesVariance(maskedImage.getImage()->getArray(), maskedImage.getVariance()->getArray(), 0,
                          result.getImage()->getArray(), result.getVariance()->getArray());
    }

    return result;
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
    table::Key<int> version;

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
              field(schema.addField<int>("field", "archive ID of the BoundedField object")),
              version(schema.addField<int>("version", "version of this PhotoCalib")) {
        schema.getCitizen().markPersistent();
    }
};

class PhotoCalibFactory : public table::io::PersistableFactory {
public:
    PTR(table::io::Persistable)
    read(InputArchive const &archive, CatalogVector const &catalogs) const override {
        table::BaseRecord const &record = catalogs.front().front();
        PhotoCalibSchema const &keys = PhotoCalibSchema::get();
        int version = getVersion(record);
        if (version != 0) {
            throw(pex::exceptions::TypeError("Unsupported version: " + std::to_string(version)));
        }
        return std::make_shared<PhotoCalib>(record.get(keys.calibrationMean), record.get(keys.calibrationErr),
                                            archive.get<afw::math::BoundedField>(record.get(keys.field)),
                                            record.get(keys.isConstant));
    }

    PhotoCalibFactory(std::string const &name) : afw::table::io::PersistableFactory(name) {}

protected:
    int getVersion(table::BaseRecord const &record) const {
        int version = -1;
        try {
            std::string versionName = "version";
            auto versionKey = record.getSchema().find<int>(versionName);
            version = record.get(versionKey.key);
        } catch (const pex::exceptions::NotFoundError &) {
            // un-versioned files are version 0
            version = 0;
        }
        return version;
    }
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
    record->set(keys.version, SERIALIZATION_VERSION);
    handle.saveCatalog(catalog);
}

// ------------------- private/protected helpers -------------------

double PhotoCalib::evaluate(lsst::geom::Point<double, 2> const &point) const {
    if (_isConstant)
        return _calibrationMean;
    else
        return _calibration->evaluate(point);
}

void PhotoCalib::instFluxToMaggiesArray(afw::table::SourceCatalog const &sourceCatalog,
                                        std::string const &instFluxField,
                                        ndarray::Array<double, 2, 2> result) const {
    double instFlux, instFluxErr, maggies, calibration;
    auto instFluxKey = sourceCatalog.getSchema().find<double>(instFluxField + "_instFlux").key;
    auto instFluxErrKey = sourceCatalog.getSchema().find<double>(instFluxField + "_instFluxErr").key;
    auto iter = result.begin();
    for (auto const &rec : sourceCatalog) {
        instFlux = rec.get(instFluxKey);
        instFluxErr = rec.get(instFluxErrKey);
        calibration = evaluate(rec.getCentroid());
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
    auto instFluxKey = sourceCatalog.getSchema().find<double>(instFluxField + "_instFlux").key;
    auto instFluxErrKey = sourceCatalog.getSchema().find<double>(instFluxField + "_instFluxErr").key;
    auto iter = result.begin();
    for (auto const &rec : sourceCatalog) {
        instFlux = rec.get(instFluxKey);
        instFluxErr = rec.get(instFluxErrKey);
        calibration = evaluate(rec.getCentroid());
        (*iter)[0] = toMagnitude(instFlux, calibration);
        (*iter)[1] = toMagnitudeErr(instFlux, instFluxErr, calibration, _calibrationErr);
        iter++;
    }
}

}  // namespace image
}  // namespace afw
}  // namespace lsst
