// -*- LSST-C++ -*-

/*
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
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
            pex::exceptions::RuntimeError,
            "Image bounding box does not match field bounding box"
        );
    }
    for (int y = region.getBeginY(), yEnd = region.getEndY(); y < yEnd; ++y) {
        typename image::Image<T>::x_iterator rowIter = img.x_at(
            region.getBeginX() - img.getX0(), y - img.getY0()
        );
        for (int x = region.getBeginX(), xEnd = region.getEndX(); x < xEnd; ++x, ++rowIter) {
            functor(*rowIter, field.evaluate(x, y));
        }
    }
}

} // anonymous

PTR(BoundedField) operator*(double const scale, CONST_PTR(BoundedField) bf) {
    return *bf*scale;
}

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
