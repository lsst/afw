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

#ifndef LSST_AFW_IMAGE_PHOTOCALIB_H
#define LSST_AFW_IMAGE_PHOTOCALIB_H

/**
 * @file
 * @brief Implementation of the Photometric Calibration class.
 * @ingroup afw
 */

#include "lsst/afw/math/BoundedField.h"
#include "lsst/afw/math/ChebyshevBoundedField.h"
#include "lsst/geom/Point.h"
#include "lsst/geom/Box.h"
#include "lsst/afw/table/Source.h"
#include "lsst/afw/table/io/Persistable.h"

namespace lsst {
namespace afw {
namespace image {

/// A value and its error.
struct Measurement {
    Measurement(double value, double err) : value(value), err(err) {}
    double const value;
    double const err;
};

/**
 * @class PhotoCalib
 *
 * @brief The photometric calibration of an exposure.
 *
 * A PhotoCalib is a BoundedField (a function with a specified domain) that converts from post-ISR
 * counts-on-chip (ADU) to flux and magnitude. It is defined in terms of "maggies", which are a linear
 * unit defined in SDSS (definition given in nanomaggies):
 *     http://www.sdss.org/dr12/algorithms/magnitudes/#nmgy
 *
 * PhotoCalib is immutable.
 *
 * The spatially varying flux calibration has units of maggies/ADU, and is defined such that,
 * at a position (x,y) in the domain of the boundedField calibration and for a given measured source instFlux:
 * @f[
 *     instFlux*calibration(x,y) = flux [maggies]
 * @f]
 * while the errors (constant on the domain) are defined as:
 * @f[
 *     sqrt((instFluxErr/instFlux)^2 + (calibrationErr/calibration)^2)*flux = fluxErr [maggies]
 * @f]
 * This implies that the conversions from instFlux and instFlux error to magnitude and magnitude error
 * are as follows:
 * @f[
 *     -2.5*log_{10}(instFlux*calibration(x,y)) = magnitude
 * @f]
 * and
 * @f[
 *     2.5/log(10)*sqrt((instFluxErr/instFlux)^2 + (calibrationErr/calibration)^2) = magnitudeErr
 * @f]
 */
class PhotoCalib : public table::io::PersistableFacade<PhotoCalib>, public table::io::Persistable {
public:
    // no move or copy
    PhotoCalib(PhotoCalib const &) = delete;
    PhotoCalib(PhotoCalib &&) = delete;
    PhotoCalib &operator=(PhotoCalib const &) = delete;
    PhotoCalib &operator=(PhotoCalib &&) = delete;

    ~PhotoCalib() override = default;

    /**
     * Create a empty, zeroed calibration.
     */
    PhotoCalib() : PhotoCalib(0) {}

    /**
     * Create a non-spatially-varying calibration.
     *
     * @param[in]  calibrationMean The spatially-constant calibration.
     * @param[in]  calibrationErr  The error on the calibration.
     * @param[in]  bbox            The bounding box on which this PhotoCalib is valid. If not specified,
     *                             this PhotoCalib is valid at any point (i.e. an empty bbox).
     * @param[in]  takesSurfaceBrightness Is this PhotoCalib defined on a surface brightness image (true), or
     * a fluence image (false)?
     */
    explicit PhotoCalib(double calibrationMean, double calibrationErr = 0,
                        lsst::geom::Box2I const &bbox = lsst::geom::Box2I(),
                        bool takesSurfaceBrightness = true)
            : _calibrationMean(calibrationMean),
              _calibrationErr(calibrationErr),
              _isConstant(true),
              _takesSurfaceBrightness(takesSurfaceBrightness) {
        ndarray::Array<double, 2, 2> coeffs = ndarray::allocate(ndarray::makeVector(1, 1));
        coeffs[0][0] = calibrationMean;
        _calibration = std::make_shared<afw::math::ChebyshevBoundedField>(
                afw::math::ChebyshevBoundedField(bbox, coeffs));
    }

    /**
     * Create a spatially-varying calibration.
     *
     * @param[in]  calibration    The spatially varying photometric calibration.
     * @param[in]  calibrationErr The error on the calibration.
     */
    PhotoCalib(std::shared_ptr<afw::math::BoundedField> calibration, double calibrationErr = 0,
               bool takesSurfaceBrightness = true)
            : _calibration(calibration),
              _calibrationMean(computeCalibrationMean(calibration)),
              _calibrationErr(calibrationErr),
              _isConstant(false),
              _takesSurfaceBrightness(takesSurfaceBrightness) {}

    /**
     * Create a calibration with a pre-computed mean. Primarily for de-persistence.
     *
     * @param[in]  calibrationMean The mean of the calibration() over its bounding box.
     * @param[in]  calibration     The spatially varying photometric calibration.
     * @param[in]  calibrationErr  The error on the calibration.
     * @param[in]  isConstant      Is this PhotoCalib spatially constant?
     */
    PhotoCalib(double calibrationMean, double calibrationErr,
               std::shared_ptr<afw::math::BoundedField> calibration, bool isConstant,
               bool takesSurfaceBrightness = true)
            : _calibration(calibration),
              _calibrationMean(calibrationMean),
              _calibrationErr(calibrationErr),
              _isConstant(isConstant),
              _takesSurfaceBrightness(takesSurfaceBrightness) {}

    /**
     * Convert instFlux in ADU to maggies at a point in the BoundedField.
     *
     * If passed point, use the exact calculation at that point, otherwise, use the mean scaling factor.
     *
     * @param[in]  instFlux The source instFlux in ADU.
     * @param[in]  point    The point that instFlux is measured at.
     *
     * @returns    The flux in maggies.
     */
    double instFluxToMaggies(double instFlux, lsst::geom::Point<double, 2> const &point) const;

    /// @overload instFluxToMaggies(double, lsst::geom::Point<double, 2> const &) const;
    double instFluxToMaggies(double instFlux) const;

    /**
     * Convert instFlux and error in instFlux (ADU) to maggies and maggies error.
     *
     * If passed point, use the exact calculation at that point, otherwise, use the mean scaling factor.
     *
     * @param[in]  instFlux     The source fluxinstFlux in ADU.
     * @param[in]  instFluxErr  The instFlux error.
     * @param[in]  point        The point that instFlux is measured at.
     *
     * @returns    The flux in maggies and error.
     */
    Measurement instFluxToMaggies(double instFlux, double instFluxErr,
                                  lsst::geom::Point<double, 2> const &point) const;

    /// @overload Measurement instFluxToMaggies(double, double, lsst::geom::Point<double, 2> const &) const
    Measurement instFluxToMaggies(double instFlux, double instFluxErr) const;

    /**
     * Convert `sourceRecord[instFluxField_instFlux]` (ADU) at location
     * `(sourceRecord.get("x"), sourceRecord.get("y"))` (pixels) to maggies and maggie error.
     *
     * @param[in]  sourceRecord  The source record to get instFlux and position from.
     * @param[in]  instFluxField The instFlux field: Keys of the form "*_instFlux" and "*_instFluxErr" must
     * exist.
     *                           For example: instFluxField = "PsfFlux" -> "PsfFlux_instFlux",
     * "PsfFlux_instFluxErr"
     *
     * @returns    The flux in maggies and error for this source.
     */
    Measurement instFluxToMaggies(const afw::table::SourceRecord &sourceRecord,
                                  std::string const &instFluxField) const;

    /**
     * Convert `sourceCatalog[instFluxField_instFlux]` (ADU) at locations
     * `(sourceCatalog.get("x"), sourceCatalog.get("y"))` (pixels) to maggies.
     *
     * @param[in]  sourceCatalog  The source catalog to get instFlux and position from.
     * @param[in]  instFluxField  The instFlux field: Keys of the form "*_instFlux" and "*_instFluxErr" must
     * exist.
     *                            For example: instFluxField = "PsfFlux" -> "PsfFlux_instFlux",
     * "PsfFlux_instFluxErr"
     *
     * @returns    The flux in maggies and error for this source.
     */
    ndarray::Array<double, 2, 2> instFluxToMaggies(afw::table::SourceCatalog const &sourceCatalog,
                                                   std::string const &instFluxField) const;

    /**
     * Convert `sourceCatalog[instFluxField_instFlux]` (ADU) at locations
     * `(sourceCatalog.get("x"), sourceCatalog.get("y"))` (pixels) to maggies
     * and write the results back to `sourceCatalog[outField_mag]`.
     *
     * @param[in]  sourceCatalog  The source catalog to get instFlux and position from.
     * @param[in]  instFluxField  The instFlux field: Keys of the form "*_instFlux" and "*_instFluxErr" must
     * exist.
     *                            For example: instFluxField = "PsfFlux" -> "PsfFlux_instFlux",
     * "PsfFlux_instFluxErr"
     * @param[in]  outField       The field to write the maggies and maggie errors to.
     *                            Keys of the form "*_instFlux" and "*_instFluxErr" must exist in the schema.
     *
     * @warning Not implemented yet: See DM-10155.
     */
    void instFluxToMaggies(afw::table::SourceCatalog &sourceCatalog, std::string const &instFluxField,
                           std::string const &outField) const;

    /**
     * Convert instFlux in ADU to AB magnitude.
     *
     * If passed point, use the exact calculation at that point, otherwise, use the mean scaling factor.
     *
     * @param[in]  instFlux The source instFlux in ADU.
     * @param[in]  point    The point that instFlux is measured at.
     *
     * @returns    The AB magnitude.
     */
    double instFluxToMagnitude(double instFlux, lsst::geom::Point<double, 2> const &point) const;

    /// @overload instFluxToMagnitude(double, lsst::geom::Point<double, 2> const &) const;
    double instFluxToMagnitude(double instFlux) const;

    /**
     * Convert instFlux and error in instFlux (ADU) to AB magnitude and magnitude error.
     *
     * If passed point, use the exact calculation at that point, otherwise, use the mean scaling factor.
     *
     * @param[in]  instFlux     The source instFlux in ADU.
     * @param[in]  instFluxErr  The instFlux error (standard deviation).
     * @param[in]  point        The point that instFlux is measured at.
     *
     * @returns    The AB magnitude and error.
     */
    Measurement instFluxToMagnitude(double instFlux, double instFluxErr,
                                    lsst::geom::Point<double, 2> const &point) const;

    /// @overload instFluxToMagnitude(double, double, lsst::geom::Point<double, 2> const &) const;
    Measurement instFluxToMagnitude(double instFlux, double instFluxErr) const;

    /**
     * Convert `sourceRecord[instFluxField_instFlux]` (ADU) at location
     * `(sourceRecord.get("x"), sourceRecord.get("y"))` (pixels) to AB magnitude.
     *
     * @param[in]  sourceRecord  The source record to get instFlux and position from.
     * @param[in]  instFluxField The instFlux field: Keys of the form "*_instFlux" and "*_instFluxErr" must
     * exist.
     *                           For example: instFluxField = "PsfFlux" -> "PsfFlux_instFlux",
     * "PsfFlux_instFluxErr"
     *
     * @returns    The magnitude and magnitude error for this source.
     */
    Measurement instFluxToMagnitude(afw::table::SourceRecord const &sourceRecord,
                                    std::string const &instFluxField) const;

    /**
     * Convert `sourceCatalog[instFluxField_instFlux]` (ADU) at locations
     * `(sourceCatalog.get("x"), sourceCatalog.get("y"))` (pixels) to AB magnitudes.
     *
     * @param[in]  sourceCatalog  The source catalog to get instFlux and position from.
     * @param[in]  instFluxField  The instFlux field: Keys of the form "*_instFlux" and "*_instFluxErr" must
     * exist.
     *                            For example: instFluxField = "PsfFlux" -> "PsfFlux_instFlux",
     * "PsfFlux_instFluxErr"
     *
     * @returns    The magnitudes and magnitude errors for the sources.
     */
    ndarray::Array<double, 2, 2> instFluxToMagnitude(afw::table::SourceCatalog const &sourceCatalog,
                                                     std::string const &instFluxField) const;

    /**
     * Convert instFluxes in a catalog to AB magnitudes and write back into the catalog.
     *
     * Convert `sourceCatalog[instFluxField_instFlux]` (ADU) at
     * locations `(sourceCatalog.get("x"), sourceCatalog.get("y"))` (pixels) to AB magnitudes
     * and write the results back to `sourceCatalog[outField_mag]`.
     *
     * @param[in]  sourceCatalog  The source catalog to get instFlux and position from.
     * @param[in]  instFluxField  The instFlux field: Keys of the form "*_instFlux" and "*_instFluxErr" must
     * exist.
     *                            For example: instFluxField = "PsfFlux" -> "PsfFlux_instFlux",
     * "PsfFlux_instFluxErr"
     * @param[in]  outField       The field to write the magnitudes and magnitude errors to.
     *                            Keys of the form "*_instFlux", "*_instFluxErr", *_mag", and "*_magErr"
     *                            must exist in the schema.
     *
     * @warning Not implemented yet: See DM-10155.
     */
    void instFluxToMagnitude(afw::table::SourceCatalog &sourceCatalog, std::string const &instFluxField,
                             std::string const &outField) const;

    /**
     * Convert AB magnitude to instFlux (ADU), using the mean instFlux/magnitude scaling factor.
     *
     * @param[in]  magnitude  The AB magnitude to convert.
     *
     * @returns    Source instFlux in ADU.
     */
    double magnitudeToInstFlux(double magnitude) const;

    /**
     * Get the mean photometric calibration (equal to 1 / getInstFluxMag0()).
     *
     * This value is defined, for instFlux at (x,y), such that:
     * @f[
     *   instFlux*computeScaledCalibration()(x,y)*getCalibrationMean() = instFluxToMaggies(instFlux,
     * (x,y))
     * @f]
     *
     * @see PhotoCalib::computeScaledCalibration(), getCalibrationErr(), getInstFluxMag0()
     *
     * @returns     The spatial mean of this calibration.
     */
    double getCalibrationMean() const { return _calibrationMean; }

    /**
     * Get the mean photometric calibration error.
     *
     * This value is defined such that for some instFluxErr, instFlux, and flux:
     * @f[
     *     sqrt((instFluxErr/instFlux)^2 + (calibrationErr/calibration(x,y))^2)*flux = fluxErr (in maggies)
     * @f]
     *
     * @see PhotoCalib::computeScaledCalibration(), getCalibrationMean()
     *
     * @returns    The calibration error.
     */
    double getCalibrationErr() const { return _calibrationErr; }

    /**
     * Get the magnitude zero point (equal to 1 / mean(calibration)).
     *
     * This value is defined, for instFlux at (x,y), such that:
     * @f[
     *   instFlux*computeScaledCalibration()(x,y)/getInstFluxMag0() = instFluxToMaggies(instFlux, (x,y))
     * @f]
     *
     * @see PhotoCalib::computeScaledCalibration(), getCalibrationMean()
     *
     * @returns     The instFlux magnitude zero point.
     */
    double getInstFluxMag0() const { return 1.0 / _calibrationMean; }

    /**
     * Calculates the spatially-variable calibration, normalized by the mean in the valid domain.
     *
     * This value is defined, for instFlux at (x,y), such that:
     * @f[
     *   instFlux*computeScaledCalibration()(x,y)*getCalibrationMean() = instFluxToMaggies(instFlux,
     * (x,y))
     * @f]
     *
     * @see PhotoCalib::getCalibrationMean()
     *
     * @returns    The normalized spatially-variable calibration.
     */
    std::shared_ptr<afw::math::BoundedField> computeScaledCalibration() const;

    /**
     * Calculates the scaling between this PhotoCalib and another PhotoCalib.
     *
     * The BoundedFields of these PhotoCalibs must have the same BBoxes (or one or both must be empty).
     *
     * With:
     *   - c = instFlux at position (x,y)
     *   - this = this PhotoCalib
     *   - other = other PhotoCalib
     *   - return = BoundedField returned by this method
     * the return value from this method is defined as:
     * @f[
     *   this.instFluxToMaggies(c, (x,y))*return(x, y) = other.instFluxToMaggies(c, (x,y))
     * @f]
     *
     * @param[in]  other  The PhotoCalib to scale to.
     *
     * @returns    The BoundedField as defined above.
     *
     * @warning Not implemented yet: See DM-10154.
     */
    std::shared_ptr<afw::math::BoundedField> computeScalingTo(std::shared_ptr<PhotoCalib> other) const;

    /// Is this PhotoCalib defined on a surface brightness image or a Fluence image?
    bool takesSurfaceBrightness() { return _takesSurfaceBrightness; }
    bool takesFluence() { return !_takesSurfaceBrightness; }

    /// Two PhotoCalibs are equal if their component bounded fields and calibrationErr are equal.
    bool operator==(PhotoCalib const &rhs) const;

    /// @copydoc operator==
    bool operator!=(PhotoCalib const &rhs) const { return !(*this == rhs); }

    bool isPersistable() const noexcept override { return true; }

    friend std::ostream &operator<<(std::ostream &os, PhotoCalib const &photoCalib);

protected:
    std::string getPersistenceName() const override;

    void write(OutputArchiveHandle &handle) const override;

private:
    std::shared_ptr<afw::math::BoundedField> _calibration;

    // The "mean" calibration, defined as the geometric mean of _calibration evaluated over _calibration's
    // bbox. Computed on instantiation as a convinience. Also, the actual calibration for a spatially-constant
    // calibration.
    double _calibrationMean;

    // The standard deviation of this PhotoCalib.
    double _calibrationErr;

    // Is this spatially-constant? Used to short-circuit getting centroids.
    bool _isConstant;

    // Does this exposure apply to a surface brightness image, or a fluence image?
    // PhotoCalib always returns fluence.
    bool _takesSurfaceBrightness;

    /// Returns the spatially-constant calibration (for setting _calibrationMean)
    double computeCalibrationMean(std::shared_ptr<afw::math::BoundedField> calibration) const;

    /// Helpers for converting arrays of instFlux
    void instFluxToMaggiesArray(afw::table::SourceCatalog const &sourceCatalog,
                                std::string const &instFluxField, ndarray::Array<double, 2, 2> result) const;
    void instFluxToMagnitudeArray(afw::table::SourceCatalog const &sourceCatalog,
                                  std::string const &instFluxField,
                                  ndarray::Array<double, 2, 2> result) const;
};
}  // namespace image
}  // namespace afw
}  // namespace lsst

#endif  // LSST_AFW_IMAGE_PHOTOCALIB_H
