// -*- LSST-C++ -*-
/*
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */

#include "lsst/afw/math/detail/TrapezoidalPacker.h"

namespace lsst { namespace afw { namespace math { namespace detail {

TrapezoidalPacker::TrapezoidalPacker(ChebyshevBoundedFieldControl const & ctrl)
  : nx(ctrl.orderX + 1), ny(ctrl.orderY+1)
{
    if (ctrl.triangular) {
        if (nx >= ny) {
            m = 0;
            size = (nx - ny)*ny + (ny*(ny + 1))/2;
        } else {
            m = ny - nx;
            size = m*nx + (nx*(nx + 1))/2;
        }
    } else {
        m = ny;
        size = nx*ny;
    }
}

void TrapezoidalPacker::pack(
    ndarray::Array<double,1,1> const & out,
    ndarray::Array<double const,1,1> const & tx,
    ndarray::Array<double const,1,1> const & ty
) const {
    double * outIter = out.begin();
    for (int i = 0; i < m; ++i) {  // loop over rectangular part
        for (int j = 0; j < nx; ++j, ++outIter) {
            *outIter = ty[i]*tx[j];
        }
    }
    for (int i = m; i < ny; ++i) {  // loop over wide trapezoidal part
        for (int j = 0, nj = nx + m - i; j < nj; ++j, ++outIter) {
            *outIter = ty[i]*tx[j];
        }
    }
}

void TrapezoidalPacker::pack(
    ndarray::Array<double,1,1> const & out,
    ndarray::Array<double const,2,2> const & unpacked
) const {
    double * outIter = out.begin();
    for (int i = 0; i < m; ++i) {  // loop over rectangular part
        ndarray::Array<double const,1,1> unpackedRow = unpacked[i];
        for (int j = 0; j < nx; ++j, ++outIter) {
            *outIter = unpackedRow[j];
        }
    }
    for (int i = m; i < ny; ++i) {  // loop over wide trapezoidal part
        ndarray::Array<double const,1,1> unpackedRow = unpacked[i];
        for (int j = 0, nj = nx + m - i; j < nj; ++j, ++outIter) {
            *outIter = unpackedRow[j];
        }
    }
}

void TrapezoidalPacker::unpack(
    ndarray::Array<double,2,2> const & out,
    ndarray::Array<double const,1,1> const & packed
) const {
    out.deep() = 0.0;
    double const * packedIter = packed.begin();
    for (int i = 0; i < m; ++i) {  // loop over rectangular part
        ndarray::Array<double,1,1> outRow = out[i];
        for (int j = 0; j < nx; ++j, ++packedIter) {
            outRow[j] = *packedIter;
        }
    }
    for (int i = m; i < ny; ++i) {  // loop over wide trapezoidal part
        ndarray::Array<double,1,1> outRow = out[i];
        for (int j = 0, nj = nx + m - i; j < nj; ++j, ++packedIter) {
            outRow[j] = *packedIter;
        }
    }
}

ndarray::Array<double,2,2> TrapezoidalPacker::unpack(
    ndarray::Array<double const,1,1> const & packed
) const {
    ndarray::Array<double,2,2> out = ndarray::allocate(ny, nx);
    unpack(out, packed);
    return out;
}

}}}} // namespace lsst::afw::math::detail
