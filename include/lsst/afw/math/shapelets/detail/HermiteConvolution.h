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

#ifndef LSST_AFW_MATH_SHAPELETS_DETAIL_HermiteConvolution
#define LSST_AFW_MATH_SHAPELETS_DETAIL_HermiteConvolution

#include "ndarray.h"
#include "lsst/afw/geom/ellipses.h"
#include "lsst/afw/math/shapelets/constants.h"

#include <boost/scoped_ptr.hpp>

namespace lsst { namespace afw { namespace math { namespace shapelets {

class ShapeletFunction;

namespace detail {

/**
 *  @brief A parametrized matrix that performs a convolution in shapelet space.
 *
 *  HermiteConvolution is defined only for the HERMITE basis type.
 */
class HermiteConvolution : private boost::noncopyable {
public:

    typedef boost::shared_ptr<HermiteConvolution> Ptr;

    /**
     *  @brief Evaluate a shapelet convolution matrix in the given array.
     *
     *  @param[in,out] ellipse   On input, the ellipse core of the unconvolved shapelet expansion.
     *                           On output, the ellipse core of the convolved shapelet expansion.
     */
    ndarray::Array<Pixel const,2,2> evaluate(geom::ellipses::Ellipse & ellipse) const;

    int getColOrder() const;

    int getRowOrder() const;

    HermiteConvolution(int colOrder, ShapeletFunction const & psf);

    ~HermiteConvolution();

private:
    class Impl;

    boost::scoped_ptr<Impl> _impl;
};


}}}}} // namespace lsst::afw::math::shapelets::detail

#endif // !LSST_AFW_MATH_SHAPELETS_DETAIL_HermiteConvolution
