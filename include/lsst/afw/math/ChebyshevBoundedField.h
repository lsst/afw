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

#ifndef LSST_AFW_MATH_ChebyshevBoundedField_h_INCLUDED
#define LSST_AFW_MATH_ChebyshevBoundedField_h_INCLUDED

#include "ndarray.h"

#include "lsst/pex/config.h"
#include "lsst/afw/geom/AffineTransform.h"
#include "lsst/afw/math/BoundedField.h"

namespace lsst { namespace afw { namespace math {

/// A control object used when fitting ChebyshevBoundedField to data (see ChebyshevBoundedField::fit)
class ChebyshevBoundedFieldControl {
public:

    ChebyshevBoundedFieldControl() : orderX(2), orderY(2), triangular(true) {}

    LSST_CONTROL_FIELD(orderX, int, "maximum Chebyshev function order in x");

    LSST_CONTROL_FIELD(orderY, int, "maximum Chebyshev function order in y");

    LSST_CONTROL_FIELD(
        triangular, bool,
        "if true, only include terms where the sum of the x and y order "
        "is less than or equal to max(orderX, orderY)"
    );

    /// Return the number of nonzero coefficients in the Chebyshev function defined by this object
    int computeSize() const;

};

/**
 *  @brief A BoundedField based on 2-d Chebyshev polynomials of the first kind.
 *
 *  ChebyshevBoundedField supports fitting to gridded and non-gridded data,
 *  as well coefficient matrices with different x- and y-order.
 *
 *  There is currently quite a bit of duplication of functionality between
 *  ChebyshevBoundedField, ApproximateChebyshev, and Chebyshev1Function2;
 *  the intent is that ChebyshevBoundedField will ultimately replace
 *  ApproximateChebyshev and should be preferred over Chebyshev1Function2
 *  when the parametrization interface that is part of the Function2 class
 *  is not needed.
 */
class ChebyshevBoundedField :
        public table::io::PersistableFacade<ChebyshevBoundedField>,
        public BoundedField
{
public:

    typedef ChebyshevBoundedFieldControl Control;

    /**
     *  @brief Initialize the field from its bounding box an coefficients.
     *
     *  This constructor is mostly intended for testing purposes and persistence,
     *  but it also provides a way to initialize the object from Chebyshev coefficients
     *  derived from some external source.
     *
     *  Note that because the bounding box provided is always an integer bounding box,
     *  and LSST convention puts the center of each pixel at an integer, the actual
     *  floating-point domain of the Chebyshev functions is Box2D(bbox), that is, the
     *  box that contains the entirety of all the pixels included in the integer
     *  bounding box.
     *
     *  The coefficients are ordered [y,x], so the shape is (orderY+1, orderX+1),
     *  and the arguments to the Chebyshev functions are transformed such that
     *  the region Box2D(bbox) is mapped to [-1, 1]x[-1, 1].
     */
    ChebyshevBoundedField(
        afw::geom::Box2I const & bbox,
        ndarray::Array<double const,2,2> const & coefficients
    );

    /**
     *  @brief Fit a Chebyshev approximation to non-gridded data with equal weights.
     *
     *  @param[in]  bbox     Integer bounding box of the resulting approximation.  All
     *                       given points must lie within Box2D(bbox).
     *  @param[in]  x        Array of x coordinate values.
     *  @param[in]  y        Array of y coordinate values.
     *  @param[in]  z        Array of field values to be fit at each (x,y) point.
     *  @param[in]  ctrl     Specifies the orders and triangularity of the coefficient matrix.
     */
    static PTR(ChebyshevBoundedField) fit(
        afw::geom::Box2I const & bbox,
        ndarray::Array<double const,1> const & x,
        ndarray::Array<double const,1> const & y,
        ndarray::Array<double const,1> const & z,
        Control const & ctrl
    );

    /**
     *  @brief Fit a Chebyshev approximation to non-gridded data with unequal weights.
     *
     *  @param[in]  bbox     Integer bounding box of the resulting approximation.  All
     *                       given points must lie within Box2D(bbox).
     *  @param[in]  x        Array of x coordinate values.
     *  @param[in]  y        Array of y coordinate values.
     *  @param[in]  z        Array of field values to be fit at each (x,y) point.
     *  @param[in]  w        Array of weights for each point in the fit.  For points with Gaussian
     *                       noise, w = 1/sigma.
     *  @param[in]  ctrl     Specifies the orders and triangularity of the coefficient matrix.
     */
    static PTR(ChebyshevBoundedField) fit(
        afw::geom::Box2I const & bbox,
        ndarray::Array<double const,1> const & x,
        ndarray::Array<double const,1> const & y,
        ndarray::Array<double const,1> const & z,
        ndarray::Array<double const,1> const & w,
        Control const & ctrl
    );

    /**
     *  @brief Fit a Chebyshev approximation to gridded data with equal weights.
     *
     *  @param[in]  image    The Image containing the data to fit.  image.getBBox(PARENT) is
     *                       used as the bounding box of the BoundedField.
     *  @param[in]  ctrl     Specifies the orders and triangularity of the coefficient matrix.
     *
     *  Instantiated for float and double.
     *
     *  @note if the image to be fit is a binned version of the actual image the field should
     *        correspond to, call relocate() with the unbinned image's bounding box after
     *        fitting.
     */
    template <typename T>
    static PTR(ChebyshevBoundedField) fit(
        image::Image<T> const & image,
        Control const & ctrl
    );

    /**
     *  @brief Return the coefficient matrix.
     *
     *  The coefficients are ordered [y,x], so the shape is (orderY+1, orderX+1).
     */
    ndarray::Array<double const,2,2> getCoefficients() const { return _coefficients; }

    /// Return a new ChebyshevBoudedField with maximum orders set by the given control object.
    PTR(ChebyshevBoundedField) truncate(Control const & ctrl) const;

    /**
     *  Return a new ChebyshevBoundedField with domain set to the given bounding box.
     *
     *  Because this leaves the coefficients unchanged, it is equivalent to transforming the function
     *  by the affine transform that maps the old box to the new one.
     */
    PTR(ChebyshevBoundedField) relocate(geom::Box2I const & bbox) const;

    /// @copydoc BoundedField::evaluate
    virtual double evaluate(geom::Point2D const & position) const;

    using BoundedField::evaluate;

    /// ChebyshevBoundedField is always persistable.
    virtual bool isPersistable() const { return true; }

protected:

    virtual std::string getPersistenceName() const;

    virtual std::string getPythonModule() const;

    virtual void write(OutputArchiveHandle & handle) const;

private:

    // Internal constructor for fit() routines: just initializes the transform,
    // leaves coefficients empty.
    explicit ChebyshevBoundedField(afw::geom::Box2I const & bbox);

    geom::AffineTransform _toChebyshevRange; // maps points from the bbox to [-1,1]x[-1,1]
    ndarray::Array<double const,2,2> _coefficients;  // shape=(orderY+1, orderX+1)
};


}}} // namespace lsst::afw::math

#endif // !LSST_AFW_MATH_ChebyshevBoundedField_h_INCLUDED
