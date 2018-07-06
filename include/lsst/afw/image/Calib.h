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

//
/*
 * Classes to support calibration (e.g. photometric zero points, exposure times)
 */
#ifndef LSST_AFW_IMAGE_CALIB_H
#define LSST_AFW_IMAGE_CALIB_H

#include <cmath>
#include <utility>
#include <memory>
#include "ndarray_fwd.h"
#include "lsst/base.h"
#include "lsst/daf/base/DateTime.h"
#include "lsst/pex/exceptions.h"
#include "lsst/geom/Point.h"
#include "lsst/afw/table/io/Persistable.h"

namespace lsst {
namespace daf {
namespace base {
class PropertySet;
}
}  // namespace daf

namespace afw {
namespace cameraGeom {
class Detector;
}
namespace image {

static double const JanskysPerABFlux = 3631.0;

/// Compute AB magnitude from flux in Janskys
inline double abMagFromFlux(double flux) { return -2.5 * std::log10(flux / JanskysPerABFlux); }

/// Compute AB magnitude error from flux and flux error in Janskys
inline double abMagErrFromFluxErr(double fluxErr, double flux) {
    return std::abs(fluxErr / (-0.4 * flux * std::log(10)));
}

/// Compute flux in Janskys from AB magnitude
inline double fluxFromABMag(double mag) noexcept { return std::pow(10.0, -0.4 * mag) * JanskysPerABFlux; }

/// Compute flux error in Janskys from AB magnitude error and AB magnitude
inline double fluxErrFromABMagErr(double magErr, double mag) noexcept {
    return std::abs(-0.4 * magErr * fluxFromABMag(mag) * std::log(10.0));
}

/// Compute AB magnitude from flux in Janskys
template <typename T>
ndarray::Array<T, 1> abMagFromFlux(ndarray::Array<T const, 1> const& flux);

/// Compute AB magnitude error from flux and flux error in Janskys
template <typename T>
ndarray::Array<T, 1> abMagErrFromFluxErr(ndarray::Array<T const, 1> const& fluxErr,
                                         ndarray::Array<T const, 1> const& flux);

/// Compute flux in Janskys from AB magnitude
template <typename T>
ndarray::Array<T, 1> fluxFromABMag(ndarray::Array<T const, 1> const& mag);

/// Compute flux error in Janskys from AB magnitude error and AB magnitude
template <typename T>
ndarray::Array<T, 1> fluxErrFromABMagErr(ndarray::Array<T const, 1> const& magErr,
                                         ndarray::Array<T const, 1> const& mag);

/**
 * Describe an exposure's calibration
 */

class Calib : public table::io::PersistableFacade<Calib>, public table::io::Persistable {
public:
    /**
     * ctor
     */
    explicit Calib() noexcept;
    /**
     * ctor from a given fluxMagnitude zero point
     */
    explicit Calib(double fluxMag0);
    /**
     * ctor from a vector of Calibs
     *
     * @param calibs Set of calibs to be merged
     *
     * @note All the input calibs must have the same zeropoint; throw InvalidParameterError if this isn't true
     */
    explicit Calib(std::vector<std::shared_ptr<Calib const>> const& calibs);
    /**
     * ctor
     */
    explicit Calib(std::shared_ptr<lsst::daf::base::PropertySet const>);

    Calib(Calib const&) noexcept;
    Calib(Calib&&) noexcept;
    Calib& operator=(Calib const&) noexcept;
    Calib& operator=(Calib&&) noexcept;
    virtual ~Calib() noexcept;

    /**
     * Set the flux of a zero-magnitude object
     *
     * @param fluxMag0 The flux in question (ADUs)
     * @param fluxMag0Sigma The error in the flux (ADUs)
     */
    void setFluxMag0(double fluxMag0, double fluxMag0Sigma = 0.0);
    /**
     * @param fluxMag0AndSigma The flux and error (ADUs)
     */
    void setFluxMag0(std::pair<double, double> fluxMag0AndSigma);
    /**
     * Return the flux, and error in flux, of a zero-magnitude object
     */
    std::pair<double, double> getFluxMag0() const;

    /**
     * Return a flux (in ADUs) given a magnitude
     *
     * @param mag the magnitude of the object
     */
    double getFlux(double const mag) const;

    /**
     * Return a flux and flux error (in ADUs) given a magnitude and magnitude error
     *
     * Assumes that the errors are small and uncorrelated.
     *
     * @param mag the magnitude of the object
     * @param magErr the error in the magnitude
     */
    std::pair<double, double> getFlux(double const mag, double const magErr) const;

    ndarray::Array<double, 1> getFlux(ndarray::Array<double const, 1> const& mag) const;

    std::pair<ndarray::Array<double, 1>, ndarray::Array<double, 1>> getFlux(
            ndarray::Array<double const, 1> const& mag, ndarray::Array<double const, 1> const& magErr) const;

    /**
     * Return a magnitude given a flux
     *
     * @param flux the measured flux of the object (ADUs)
     */
    double getMagnitude(double const flux) const;

    /**
     * Return a magnitude and magnitude error given a flux and flux error
     *
     * @param flux the measured flux of the object (ADUs)
     * @param fluxErr the error in the measured flux (ADUs)
     */
    std::pair<double, double> getMagnitude(double const flux, double const fluxErr) const;

    ndarray::Array<double, 1> getMagnitude(ndarray::Array<double const, 1> const& flux) const;

    std::pair<ndarray::Array<double, 1>, ndarray::Array<double, 1>> getMagnitude(
            ndarray::Array<double const, 1> const& flux,
            ndarray::Array<double const, 1> const& fluxErr) const;

    /**
     * Set whether Calib should throw an exception when asked to convert a flux to a magnitude
     *
     * @param raiseException Should the exception be raised?
     */
    static void setThrowOnNegativeFlux(bool raiseException) noexcept;
    /**
     * Tell me whether Calib will throw an exception if asked to convert a flux to a magnitude
     */
    static bool getThrowOnNegativeFlux() noexcept;
    /**
     * Are two Calibs identical?
     *
     * @note Maybe this should be an approximate comparison
     */
    bool operator==(Calib const& rhs) const noexcept;
    bool operator!=(Calib const& rhs) const noexcept { return !(*this == rhs); }

    Calib& operator*=(double const scale);
    Calib& operator/=(double const scale) {
        (*this) *= 1.0 / scale;
        return *this;
    }

    bool isPersistable() const noexcept override { return true; }

protected:
    virtual std::string getPersistenceName() const;

    virtual void write(OutputArchiveHandle& handle) const;

private:
    double _fluxMag0;
    double _fluxMag0Sigma;
    /**
     * Control whether we throw an exception when faced with a negative flux
     */
    static bool _throwOnNegativeFlux;
};

namespace detail {
/**
 * Remove Calib-related keywords from the metadata
 *
 * @param[in, out] metadata Metadata to be stripped
 * @returns Number of keywords stripped
 */
int stripCalibKeywords(std::shared_ptr<lsst::daf::base::PropertySet> metadata);
}  // namespace detail
}  // namespace image
}  // namespace afw
}  // namespace lsst

#endif  // LSST_AFW_IMAGE_CALIB_H
