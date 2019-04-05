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
namespace afw {
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

}  // namespace image
}  // namespace afw
}  // namespace lsst

#endif  // LSST_AFW_IMAGE_CALIB_H
