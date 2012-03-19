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

/**
 * @file
 *
 * @brief Conversions between shapelet basis types.
 *
 * @todo
 *
 * @author Jim Bosch
 */

#include "lsst/afw/geom/ellipses.h"
#include "ndarray.h"
#include "lsst/afw/math/shapelets/constants.h"

namespace lsst {
namespace afw {
namespace math {
namespace shapelets {

class ConversionMatrix {
public:

    Eigen::MatrixXd getBlock(int n) const;

    Eigen::MatrixXd buildDenseMatrix() const;

    void multiplyOnLeft(ndarray::Array<lsst::afw::math::shapelets::Pixel,1> const & array) const;
    
    void multiplyOnRight(ndarray::Array<lsst::afw::math::shapelets::Pixel,1> const & array) const;

    explicit ConversionMatrix(BasisTypeEnum input,
        BasisTypeEnum output, int order);

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
