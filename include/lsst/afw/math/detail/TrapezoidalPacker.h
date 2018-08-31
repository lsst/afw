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

#ifndef LSST_AFW_MATH_DETAIL_TrapezoidalPacker_h_INCLUDED
#define LSST_AFW_MATH_DETAIL_TrapezoidalPacker_h_INCLUDED

#include "lsst/afw/math/ChebyshevBoundedField.h"

namespace lsst {
namespace afw {
namespace math {
namespace detail {

/**
 *  A helper class ChebyshevBoundedField, for mapping trapezoidal matrices to 1-d arrays.
 *
 *  This class is not Swigged, and should not be included by any other .h files (including
 *  lsst/afw/math/detail.h); it's for internal use by ChebyshevBoundedField only, and it's
 *  only in a header file instead of that .cc file only so it can be unit tested.
 *
 *  We characterize the matrices by their number of columns (nx) and rows (ny),
 *  and the number of complete rows minus one (m).
 *
 *  This splits up the matrix into a rectangular part, in which the number of columns
 *  is the same for each row, and a wide trapezoidal or triangular part, in which the
 *  number of columns decreases by one for each row.
 *
 *  Here are some examples of how this class handles different kinds of matrices:
 *
 *  A wide trapezoidal matrix with orderX=4, orderY=3:
 *     nx=5, ny=4, m=0
 *
 *      0   1   2   3   4
 *      5   6   7   8
 *      9  10  11
 *      12 13
 *
 *  A tall trapezoidal matrix with orderX=2, orderY=4
 *     nx=3, ny=5, m=2
 *
 *      0   1   2
 *      3   4   5
 *      6   7   8
 *      9  10
 *     11
 *
 *  A triangular matrix with orderX=3, orderY=3
 *     nx=4, ny=5, m=0
 *
 *      0   1   2   3
 *      4   5   6
 *      7   8
 *      9
 *
 *  A wide rectangular matrix with orderX=3, orderY=2
 *     nx=4, ny=3, m=3
 *
 *      0   1   2   3
 *      4   5   6   7
 *      8   9  10  11
 *
 *  A tall rectangular matrix with orderX=2, orderY=3
 *     nx=3, ny=4, m=4
 *
 *      0   1   2
 *      3   4   5
 *      6   7   8
 *      9  10  11
 */
struct TrapezoidalPacker final {
    explicit TrapezoidalPacker(ChebyshevBoundedFieldControl const& ctrl);

    void pack(ndarray::Array<double, 1, 1> const& out, ndarray::Array<double const, 1, 1> const& tx,
              ndarray::Array<double const, 1, 1> const& ty) const;

    void pack(ndarray::Array<double, 1, 1> const& out,
              ndarray::Array<double const, 2, 2> const& unpacked) const;

    void unpack(ndarray::Array<double, 2, 2> const& out,
                ndarray::Array<double const, 1, 1> const& packed) const;

    ndarray::Array<double, 2, 2> unpack(ndarray::Array<double const, 1, 1> const& packed) const;

    int nx;
    int ny;
    int m;
    int size;
};
}  // namespace detail
}  // namespace math
}  // namespace afw
}  // namespace lsst

#endif  // !LSST_AFW_MATH_DETAIL_TrapezoidalPacker_h_INCLUDED
