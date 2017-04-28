// -*- LSST-C++ -*-
/*
 * LSST Data Management System
 * Copyright 2008-2014 LSST Corporation.
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

#include "lsst/afw/math/detail/TrapezoidalPacker.h"

namespace lsst {
namespace afw {
namespace math {
namespace detail {

TrapezoidalPacker::TrapezoidalPacker(ChebyshevBoundedFieldControl const& ctrl)
        : nx(ctrl.orderX + 1), ny(ctrl.orderY + 1) {
    if (ctrl.triangular) {
        if (nx >= ny) {
            m = 0;
            size = (nx - ny) * ny + (ny * (ny + 1)) / 2;
        } else {
            m = ny - nx;
            size = m * nx + (nx * (nx + 1)) / 2;
        }
    } else {
        m = ny;
        size = nx * ny;
    }
}

void TrapezoidalPacker::pack(ndarray::Array<double, 1, 1> const& out,
                             ndarray::Array<double const, 1, 1> const& tx,
                             ndarray::Array<double const, 1, 1> const& ty) const {
    double* outIter = out.begin();
    for (int i = 0; i < m; ++i) {  // loop over rectangular part
        for (int j = 0; j < nx; ++j, ++outIter) {
            *outIter = ty[i] * tx[j];
        }
    }
    for (int i = m; i < ny; ++i) {  // loop over wide trapezoidal part
        for (int j = 0, nj = nx + m - i; j < nj; ++j, ++outIter) {
            *outIter = ty[i] * tx[j];
        }
    }
}

void TrapezoidalPacker::pack(ndarray::Array<double, 1, 1> const& out,
                             ndarray::Array<double const, 2, 2> const& unpacked) const {
    double* outIter = out.begin();
    for (int i = 0; i < m; ++i) {  // loop over rectangular part
        ndarray::Array<double const, 1, 1> unpackedRow = unpacked[i];
        for (int j = 0; j < nx; ++j, ++outIter) {
            *outIter = unpackedRow[j];
        }
    }
    for (int i = m; i < ny; ++i) {  // loop over wide trapezoidal part
        ndarray::Array<double const, 1, 1> unpackedRow = unpacked[i];
        for (int j = 0, nj = nx + m - i; j < nj; ++j, ++outIter) {
            *outIter = unpackedRow[j];
        }
    }
}

void TrapezoidalPacker::unpack(ndarray::Array<double, 2, 2> const& out,
                               ndarray::Array<double const, 1, 1> const& packed) const {
    out.deep() = 0.0;
    double const* packedIter = packed.begin();
    for (int i = 0; i < m; ++i) {  // loop over rectangular part
        ndarray::Array<double, 1, 1> outRow = out[i];
        for (int j = 0; j < nx; ++j, ++packedIter) {
            outRow[j] = *packedIter;
        }
    }
    for (int i = m; i < ny; ++i) {  // loop over wide trapezoidal part
        ndarray::Array<double, 1, 1> outRow = out[i];
        for (int j = 0, nj = nx + m - i; j < nj; ++j, ++packedIter) {
            outRow[j] = *packedIter;
        }
    }
}

ndarray::Array<double, 2, 2> TrapezoidalPacker::unpack(
        ndarray::Array<double const, 1, 1> const& packed) const {
    ndarray::Array<double, 2, 2> out = ndarray::allocate(ny, nx);
    unpack(out, packed);
    return out;
}
}
}
}
}  // namespace lsst::afw::math::detail
