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

#include "lsst/pex/exceptions.h"
#include "lsst/afw/math/BoundedField.h"

namespace lsst { namespace afw { namespace math {

ndarray::Array<double,1,1> BoundedField::evaluate(
    ndarray::Array<double const,1> const & x,
    ndarray::Array<double const,1> const & y
) const {
    ndarray::Array<double,1,1> out = ndarray::allocate(x.getSize<0>());
    for (int i = 0, n = x.getSize<0>(); i < n; ++i) {
        out[i] = evaluate(x[i], y[i]);
    }
    return out;
}

namespace {

// We use these operator-based functors to implement the various image-modifying routines
// in BoundedField.  I don't think this is necessarily the best way to add interoperability
// with images, but it seems like a reasonable point on the simplicity vs. featurefulness
// scale, and I don't have time to explore more complex solutions (expression templates!).
// Of course, in C++11, we could replace all these with lambdas.

struct Assign {

    template <typename T>
    void operator()(T & out, double a) const { out = a; }

};

struct ScaledAdd {

    explicit ScaledAdd(double s) : scaleBy(s) {}

    template <typename T>
    void operator()(T & out, double a) const { out += scaleBy * a; }

    double scaleBy;
};

struct Multiply {

    template <typename T>
    void operator()(T & out, double a) const { out *= a; }

};

struct Divide {

    template <typename T>
    void operator()(T & out, double a) const { out /= a; }

};

template <typename T, typename F>
void applyToImage(BoundedField const & field, image::Image<T> & img, F functor, bool overlapOnly) {
    geom::Box2I region(field.getBBox());
    if (overlapOnly) {
        region.clip(img.getBBox(image::PARENT));
    } else if (region != img.getBBox(image::PARENT)) {
        throw LSST_EXCEPT(
            pex::exceptions::RuntimeErrorException,
            "Image bounding box does not match field bounding box"
        );
    }
    for (int y = region.getBeginY(), yEnd = region.getEndY(); y < yEnd; ++y) {
        typename image::Image<T>::x_iterator rowIter = img.x_at(
            region.getBeginX() - img.getX0(), y - img.getY0()
        );
        for (int x = region.getBeginX(), xEnd = region.getEndY(); x < xEnd; ++x, ++rowIter) {
            functor(*rowIter, field.evaluate(x, y));
        }
    }
}

} // anonymous

template <typename T>
void BoundedField::fillImage(image::Image<T> & img, bool overlapOnly) const {
    applyToImage(*this, img, Assign(), overlapOnly);
}

template <typename T>
void BoundedField::addToImage(image::Image<T> & img, double scaleBy, bool overlapOnly) const {
    applyToImage(*this, img, ScaledAdd(scaleBy), overlapOnly);
}

template <typename T>
void BoundedField::multiplyImage(image::Image<T> & img, bool overlapOnly) const {
    applyToImage(*this, img, Multiply(), overlapOnly);
}

template <typename T>
void BoundedField::divideImage(image::Image<T> & img, bool overlapOnly) const {
    applyToImage(*this, img, Divide(), overlapOnly);
}

#define INSTANTIATE(T)                          \
    template void BoundedField::fillImage(image::Image<T> &, bool) const; \
    template void BoundedField::addToImage(image::Image<T> &, double, bool) const; \
    template void BoundedField::multiplyImage(image::Image<T> &, bool) const; \
    template void BoundedField::divideImage(image::Image<T> &, bool) const

INSTANTIATE(float);
INSTANTIATE(double);

}}} // namespace lsst::afw::math
