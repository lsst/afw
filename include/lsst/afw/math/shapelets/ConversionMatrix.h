// -*- LSST-C++ -*-

/* 
 * LSST Data Management System
 * Copyright 2008, 2009, 2010, 2011 LSST Corporation.
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
 
#ifndef LSST_AFW_MATH_SHAPELETS_CONVERSIONMATRIX_H
#define LSST_AFW_MATH_SHAPELETS_CONVERSIONMATRIX_H


#include "lsst/afw/geom/ellipses.h"
#include "ndarray.h"
#include "lsst/afw/math/shapelets/constants.h"

namespace lsst {
namespace afw {
namespace math {
namespace shapelets {

/**
 *  @brief Conversions between shapelet basis types.
 *
 *  The basis conversion matrix is block-diagonal and only needs to be computed once, so we cache
 *  the blocks in a hidden singleton and provide operations that act on shapelet matrices while
 *  taking advantage of the sparseness of the conversion.
 */
class ConversionMatrix {
public:

    /// @brief Return a block of the block-diagonal conversion matrix.
    Eigen::MatrixXd getBlock(int n) const;

    /// @brief Construct the full conversion matrix (should just be used for testing).
    Eigen::MatrixXd buildDenseMatrix() const;

    /// @brief Multiply the given array by the conversion matrix on the left in-place.
    void multiplyOnLeft(ndarray::Array<lsst::afw::math::shapelets::Pixel,1> const & array) const;
    
    /// @brief Multiply the given array by the conversion matrix on the right in-place.
    void multiplyOnRight(ndarray::Array<lsst::afw::math::shapelets::Pixel,1> const & array) const;

    /// @brief Construct a conversion matrix that maps the input basis to the output basis.
    explicit ConversionMatrix(BasisTypeEnum input, BasisTypeEnum output, int order);

    /// @brief Convert a coefficient vector between basis types in-place.
    static void convertCoefficientVector(
        ndarray::Array<lsst::afw::math::shapelets::Pixel,1> const & array,
        BasisTypeEnum input,
        BasisTypeEnum output, int order
    );

    /// @brief Convert an operation (evaluation, integration) vector between basis types in-place.
    static void convertOperationVector(
        ndarray::Array<lsst::afw::math::shapelets::Pixel,1> const & array,
        BasisTypeEnum input,
        BasisTypeEnum output, int order
    );
    
private:
    int _order;
    BasisTypeEnum _input;
    BasisTypeEnum _output;
};


}}}}   // lsst::afw::math::shapelets

#endif // !defined(LSST_AFW_MATH_SHAPELETS_CONVERSIONMATRIX_H)
