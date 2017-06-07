// -*- LSST-C++ -*-
/*
 * LSST Data Management System
 * See COPYRIGHT file at the top of the source tree.
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the LSST License Statement and
 * the GNU General Public License along with this program. If not,
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */

#include <sstream>
#include <cmath>

#include "astshim.h"

#include "Eigen/Core"

#include "lsst/afw/geom/Endpoint.h"
#include "lsst/afw/geom/transformFactory.h"
#include "lsst/pex/exceptions.h"

#include "ndarray.h"
#include "ndarray/eigen.h"

namespace lsst {
namespace afw {
namespace geom {

namespace {
/**
 * Print a vector to a stream.
 *
 * The exact details of the representation are unspecified and subject to
 * change, but the following may be regarded as typical:
 *
 *     [1.0, -3.560, 42.0]
 *
 * @tparam T the element type. Must support stream output.
 */
template <typename T>
std::ostream &operator<<(std::ostream &os, std::vector<T> const &v) {
    os << '[';
    bool first = true;
    for (T element : v) {
        if (first) {
            first = false;
        } else {
            os << ", ";
        }
        os << element;
    }
    os << ']';
    return os;
}

/**
 * Convert a Matrix to the equivalent ndarray.
 *
 * @param matrix The matrix to convert.
 * @returns an ndarray containing a copy of the data in `matrix`
 */
template <class T, int Rows, int Cols>
ndarray::Array<T, 2, 2> toNdArray(Eigen::Matrix<T, Rows, Cols> const &matrix) {
    ndarray::Array<T, 2, 2> array = ndarray::allocate(ndarray::makeVector(matrix.rows(), matrix.cols()));
    array.asEigen() = matrix;
    return array;
}
}  // namespace {}

template <class From, class To>
Transform<From, To> linearizeTransform(Transform<From, To> const &original,
                                       typename Transform<From, To>::FromPoint const &inPoint) {
    using Transform = Transform<From, To>;
    auto fromEndpoint = original.getFromEndpoint();
    auto toEndpoint = original.getToEndpoint();

    auto outPoint = original.applyForward(inPoint);
    auto jacobian = original.getJacobian(inPoint);
    for (int i = 0; i < toEndpoint.getNAxes(); ++i) {
        if (!std::isfinite(outPoint[i])) {
            std::ostringstream buffer;
            buffer << "Transform ill-defined: " << inPoint << " -> " << outPoint;
            throw LSST_EXCEPT(pex::exceptions::InvalidParameterError, buffer.str());
        }
    }
    if (!jacobian.allFinite()) {
        std::ostringstream buffer;
        buffer << "Transform not continuous at " << inPoint << ": J = " << jacobian;
        throw LSST_EXCEPT(pex::exceptions::InvalidParameterError, buffer.str());
    }

    // y(x) = J (x - x0) + y0
    auto map = ast::ShiftMap(fromEndpoint.dataFromPoint(inPoint))
                       .getInverse()
                       ->then(ast::MatrixMap(toNdArray(jacobian)))
                       .then(ast::ShiftMap(toEndpoint.dataFromPoint(outPoint)));
    // TODO: remove false flag as part of DM-10947
    return Transform(map, false);
}

#define INSTANTIATE_FACTORIES(From, To)                                                             \
    template Transform<From, To> linearizeTransform<From, To>(Transform<From, To> const &transform, \
                                                              Transform<From, To>::FromPoint const &point);

// explicit instantiations
INSTANTIATE_FACTORIES(GenericEndpoint, GenericEndpoint);
INSTANTIATE_FACTORIES(GenericEndpoint, Point2Endpoint);
INSTANTIATE_FACTORIES(Point2Endpoint, GenericEndpoint);
INSTANTIATE_FACTORIES(Point2Endpoint, Point2Endpoint);
}
}
} /* namespace lsst::afw::geom */
