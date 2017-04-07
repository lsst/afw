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

#ifndef LSST_AFW_MATH_BoundedField_h_INCLUDED
#define LSST_AFW_MATH_BoundedField_h_INCLUDED

#include "ndarray.h"

#include "lsst/base.h"
#include "lsst/afw/geom/Point.h"
#include "lsst/afw/image/Image.h"
#include "lsst/afw/table/io/Persistable.h"

namespace lsst { namespace afw { namespace math {

/**
 *  An abstract base class for 2-d functions defined on an integer bounding boxes
 *
 *  Integer bounding boxes (afw.geom.Box2I) are inclusive of the end pixels (integer positions correspond
 *  to the centers of the pixels and include the entirety of those pixels). Thus a BoundedField defined on
 *  the box [x0, x1] x [y0, y1] actually covers the range [x0 - 0.5, x1 + 0.5] x [y0 - 0.5, y1 + 0.5].
 *
 *  BoundedField provides a number of ways of accessing the function, all delegating to a single
 *  evaluate-at-a-point implementation.  The base class does not mandate anything about how the field
 *  is constructed, so it's appropriate for use with e.g. model-fitting results, interpolation results
 *  points, or functions known a priori.
 *
 *  Usually, BoundedField will be used to represent functions that correspond to images, for quantities
 *  such as aperture corrections, photometric scaling, PSF model parameters, or backgrounds, and its
 *  bounding box will be set to match the PARENT bounding box of the image.
 */
class BoundedField : public table::io::PersistableFacade<BoundedField>, public table::io::Persistable {
public:

    /**
     *  Evaluate the field at the given point.
     *
     *  This is the only abstract method to be implemented by subclasses.
     *
     *  Subclasses should not provide bounds checking on the given position; this is the responsibility
     *  of the user, who can almost always do it more efficiently.
     */
    virtual double evaluate(geom::Point2D const & position) const = 0;

    /**
     *  Evaluate the field at the given point.
     *
     *  This delegates to the evaluate() method that takes geom::Point2D.
     *
     *  There is no bounds-checking on the given position; this is the responsibility
     *  of the user, who can almost always do it more efficiently.
     */
    double evaluate(double x, double y) const { return evaluate(geom::Point2D(x, y)); }

    /**
     *  Evaluate the field at multiple arbitrary points
     *
     *  @param[in]  x         array of x coordinates, same shape as y
     *  @param[in]  y         array of y coordinates, same shape as x
     *  @returns an array of output values, same shape as x and y
     *
     *  There is no bounds-checking on the given positions; this is the responsibility
     *  of the user, who can almost always do it more efficiently.
     */
    ndarray::Array<double,1,1> evaluate(
        ndarray::Array<double const,1> const & x,
        ndarray::Array<double const,1> const & y
    ) const;

    /**
     * Compute the integral of this function over its bounding-box.
     *
     * @returns The value of the integral.
     */
    virtual double integrate() const;

    /**
     * Compute the mean of this function over its bounding-box.
     *
     * @returns The value of the mean.
     */
    virtual double mean() const;

    /**
     *  Return the bounding box that defines the region where the field is valid
     *
     *  Because this is an integer bounding box, its minimum and maximum positions are the
     *  centers of the pixels where the field is valid, but the field can be assumed to be
     *  valid to the edges of those pixels, which is the boundary you'd get by converting
     *  the returned Box2I into a Box2D.
     */
    geom::Box2I getBBox() const { return _bbox; }

    /**
     *  Assign the field to an image, overwriting values already present.
     *
     *  @param[out]   image         Image to fill.
     *  @param[in]    overlapOnly   If true, only modify the region in the intersection of
     *                              image.getBBox(image::PARENT) and this->getBBox().
     *
     *  @throws pex::exceptions::RuntimeError if the bounding boxes do not overlap
     *         and overlapOnly=false.
     */
    template <typename T>
    void fillImage(image::Image<T> & image, bool overlapOnly=false) const;

    /**
     *  Add the field or a constant multiple of it to an image in-place
     *
     *  @param[out]   image         Image to add to.
     *  @param[in]    scaleBy       Multiply the field by this before adding it to the image.
     *  @param[in]    overlapOnly   If true, only modify the region in the intersection of
     *                              image.getBBox(image::PARENT) and this->getBBox().
     *
     *  @throws pex::exceptions::RuntimeError if the bounding boxes do not overlap
     *         and overlapOnly=false.
     */
    template <typename T>
    void addToImage(image::Image<T> & image, double scaleBy=1.0, bool overlapOnly=false) const;

    /**
     *  Multiply an image by the field in-place.
     *
     *  @param[out]   image         Image to fill.
     *  @param[in]    overlapOnly   If true, only modify the region in the intersection of
     *                              image.getBBox(image::PARENT) and this->getBBox().
     *
     *  @throws pex::exceptions::RuntimeError if the bounding boxes do not overlap
     *         and overlapOnly=false.
     */
    template <typename T>
    void multiplyImage(image::Image<T> & image, bool overlapOnly=false) const;

    /**
     *  Divide an image by the field in-place.
     *
     *  @param[out]   image         Image to fill.
     *  @param[in]    overlapOnly   If true, only modify the region in the intersection of
     *                              image.getBBox(image::PARENT) and this->getBBox().
     *
     *  @throws pex::exceptions::RuntimeError if the bounding boxes do not overlap
     *         and overlapOnly=false.
     */
    template <typename T>
    void divideImage(image::Image<T> & image, bool overlapOnly=false) const;

    /**
     *  Return a scaled BoundedField
     *
     *  @param[in]  scale    Scaling factor
     */
    virtual PTR(BoundedField) operator*(double const scale) const = 0;
    PTR(BoundedField) operator/(double scale) const {
        return (*this)*(1.0/scale);
    }

    virtual ~BoundedField() {}

protected:

    explicit BoundedField(geom::Box2I const & bbox) : _bbox(bbox) {}

private:
    geom::Box2I const _bbox;
};

PTR(BoundedField) operator*(double const scale, CONST_PTR(BoundedField) bf);

}}} // namespace lsst::afw::math

#endif // !LSST_AFW_MATH_BoundedField_h_INCLUDED
