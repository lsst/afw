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

#include "ndarray.h"
#include "lsst/pex/exceptions.h"
#include "lsst/afw/image/Calib.h"

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
        throw LSST_EXCEPT(pex::exceptions::LengthError, (boost::format("Length mismatch: %d vs %d") %
                                                         flux.getNumElements() % fluxErr.getNumElements())
                                                                .str());
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
        throw LSST_EXCEPT(pex::exceptions::LengthError, (boost::format("Length mismatch: %d vs %d") %
                                                         mag.getNumElements() % magErr.getNumElements())
                                                                .str());
    }
    ndarray::Array<T, 1> out = ndarray::allocate(mag.getShape());
    for (std::size_t ii = 0; ii < mag.getNumElements(); ++ii) {
        out[ii] = fluxErrFromABMagErr(magErr[ii], mag[ii]);
    }
    return out;
}

// Explicit instantiation
#define INSTANTIATE(TYPE)                                                                              \
    template ndarray::Array<TYPE, 1> abMagFromFlux(ndarray::Array<TYPE const, 1> const& flux);         \
    template ndarray::Array<TYPE, 1> abMagErrFromFluxErr(ndarray::Array<TYPE const, 1> const& fluxErr, \
                                                         ndarray::Array<TYPE const, 1> const& flux);   \
    template ndarray::Array<TYPE, 1> fluxFromABMag(ndarray::Array<TYPE const, 1> const& mag);          \
    template ndarray::Array<TYPE, 1> fluxErrFromABMagErr(ndarray::Array<TYPE const, 1> const& magErr,  \
                                                         ndarray::Array<TYPE const, 1> const& mag);

INSTANTIATE(float);
INSTANTIATE(double);

}  // namespace image
}  // namespace afw
}  // namespace lsst
