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

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE TransformCpp

#include <array>

#include "boost/test/unit_test.hpp"

#include "lsst/afw/geom/Transform.h"

/*
 * Unit tests for C++-only functionality in Transform.
 *
 * See test_transform.py for remaining unit tests.
 */
namespace lsst {
namespace afw {
namespace geom {

/**
 * Make an ast::PolyMap suitable for testing.
 *
 * The forward transform is as follows:
 * @f[ f_j(\vec{x}) = \sum_{i} C_{ij} x_{i}^2 @f]
 * where @f$ C_{ij} = 0.001 (i+j+1) @f$.
 *
 * The equation is chosen for the following reasons:
 * - It is well defined for any value of `nIn`, `nOut`.
 * - It stays small for small `x`, to avoid wraparound of angles for
 *      SpherePoint endpoints.
 *
 * @param nIn,nOut The input and output dimensions of the desired PolyMap.
 * @returns a Mapping with a forward transform described by the equation above
 *          and no inverse transform.
 */
ast::PolyMap makeForwardPolyMap(size_t nIn, size_t nOut) {
    using namespace ndarray;

    double const baseCoeff = 0.001;
    Array<double, 2, 2> forwardCoeffs = allocate(ndarray::makeVector(nOut * nIn, 2 + nIn));
    for (size_t iOut = 0; iOut < nOut; ++iOut) {
        double const coeffOffset = baseCoeff * iOut;
        for (size_t iIn = 0; iIn < nIn; ++iIn) {
            auto row = forwardCoeffs[iOut * nIn + iIn];
            row[0] = baseCoeff * (iIn + 1) + coeffOffset;  // Coefficient
            row[1] = iOut + 1;                             // Compute f_iOut
            row[view(2, 2 + nIn)] = 0;                     // Ignore most variables
            row[2 + iIn] = 2;                              // Square x_iIn
        }
    }

    return ast::PolyMap(forwardCoeffs, nOut, "IterInverse=0");
}

/**
 * Tests whether the result of Transform::getJacobian(FromPoint const&)
 * has the specified dimensions.
 *
 * The Python version of this method follows a slightly different spec to
 * conform to the numpy convention that length=1 dimensions do not exist.
 */
BOOST_AUTO_TEST_CASE(getJacobianDimensions) {
    using GenericTransform = Transform<GenericEndpoint, GenericEndpoint>;
    std::array<size_t, 6> const dimensions = {{1, 2, 3, 4, 5, 6}};

    for (auto nIn : dimensions) {
        for (auto nOut : dimensions) {
            std::string msg = " [nIn=" + std::to_string(nIn) + ", nOut=" + std::to_string(nOut) + "]";
            auto polyMap = makeForwardPolyMap(nIn, nOut);
            GenericTransform transform(polyMap);

            // Don't care about elements, so zero initialization is ok
            auto inPoint = std::vector<double>(nIn);
            Eigen::MatrixXd jacobian = transform.getJacobian(inPoint);

            auto fromAxes = transform.getFromEndpoint().getNAxes();
            auto toAxes = transform.getToEndpoint().getNAxes();
            BOOST_TEST(jacobian.rows() == toAxes, "Matrix has " << jacobian.rows() << " rows, expected "
                                                                << toAxes << msg);
            BOOST_TEST(jacobian.cols() == fromAxes, "Matrix has " << jacobian.cols() << " columns, expected "
                                                                  << fromAxes << msg);
        }
    }
}
}
}
} /* namespace lsst::afw::geom */
