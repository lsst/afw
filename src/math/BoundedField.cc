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

namespace lsst {
namespace afw {
namespace math {

ndarray::Array<double, 1, 1> BoundedField::evaluate(ndarray::Array<double const, 1> const &x,
                                                    ndarray::Array<double const, 1> const &y) const {
    ndarray::Array<double, 1, 1> out = ndarray::allocate(x.getSize<0>());
    for (int i = 0, n = x.getSize<0>(); i < n; ++i) {
        out[i] = evaluate(x[i], y[i]);
    }
    return out;
}

double BoundedField::integrate() const { throw LSST_EXCEPT(pex::exceptions::LogicError, "Not Implemented"); }

double BoundedField::mean() const { throw LSST_EXCEPT(pex::exceptions::LogicError, "Not Implemented"); }

namespace {

// We use these operator-based functors to implement the various image-modifying routines
// in BoundedField.  I don't think this is necessarily the best way to add interoperability
// with images, but it seems like a reasonable point on the simplicity vs. featurefulness
// scale, and I don't have time to explore more complex solutions (expression templates!).
// Of course, in C++11, we could replace all these with lambdas.

struct Assign {
    template <typename T>
    void operator()(T &out, double a) const {
        out = a;
    }
};

struct ScaledAdd {
    explicit ScaledAdd(double s) : scaleBy(s) {}

    template <typename T>
    void operator()(T &out, double a) const {
        out += scaleBy * a;
    }

    double scaleBy;
};

struct Multiply {
    template <typename T>
    void operator()(T &out, double a) const {
        out *= a;
    }
};

struct Divide {
    template <typename T>
    void operator()(T &out, double a) const {
        out /= a;
    }
};

template <typename T, typename F>
void applyToImage(BoundedField const &field, image::Image<T> &img, F functor, bool overlapOnly, int xStep,
                  int yStep) {
    geom::Box2I region(field.getBBox());
    if (overlapOnly) {
        region.clip(img.getBBox(image::PARENT));
    } else if (region != img.getBBox(image::PARENT)) {
        throw LSST_EXCEPT(pex::exceptions::RuntimeError,
                          "Image bounding box does not match field bounding box");
    }

    if (yStep > 1 || xStep > 1) {

        // Bilinear interpolation with no dynamic memory allocation:
        // iterate over (y0, y1) intervals, then (x0, x1) intervals to evaluate
        // the field at the corners of each box, then iterate over pixels in
        // the box (in y, then x) to fill interpolated values.
        // This means we evaluate the function multiple times at some points
        // (when we move to a new interval in y, to be specific).
        // Could probably be optimized for the cases where either step is 1,
        // but that seems premature.

        // Local function to do bilinear interpolation using points
        // [(x0, y0), (x0, y1), (x1, y0), (x1, y1)] with values
        // [z00, z01, z10, z11], in pixels [x0:xEnd, y0:yEnd]
        auto interpolate = [&img, &functor](
            int x0, int x1, int xEnd,
            int y0, int y1, int yEnd,
            double z00, double z01, double z10, double z11
        ) {
            int dy = y1 - y0;
            int dx = x1 - x0;
            // First iteration of each of the for loops below has been
            // split into an explicit block.  This avoids a few instructions
            // when no interpolation is necessary, but more importantly it
            // allows us to handle dx==0 and dy==0 with no divide-by-zero
            // problems.
            { // y=y0
                auto rowIter =
                    img.x_at(x0 - img.getX0(), y0 - img.getY0());
                { // x=x0
                    functor(*rowIter, z00);
                    ++rowIter;
                }
                for (int x = x0 + 1; x < xEnd; ++x, ++rowIter) {
                    functor(*rowIter, ((x1 - x)*z00 + (x - x0)*z10)/dx);
                }
            }
            for (int y = y0 + 1; y < yEnd; ++y) {
                auto rowIter =
                    img.x_at(x0 - img.getX0(), y - img.getY0());
                double z0 = ((y1 - y)*z00 + (y - y0)*z01)/dy;
                double z1 = ((y1 - y)*z10 + (y - y0)*z11)/dy;
                { // x=x0
                    functor(*rowIter, z0);
                    ++rowIter;
                }
                for (int x = x0 + 1; x < xEnd; ++x, ++rowIter) {
                    functor(*rowIter, ((x1 - x)*z0 + (x - x0)*z1)/dx);
                }
            }
        };

        auto processRowBlock = [&region, &field, &interpolate, xStep](int y0, int y1, int yEnd) {
            int x0 = region.getBeginX();
            int x1 = x0 + xStep;
            double z00 = field.evaluate(x0, y0);
            double z01 = field.evaluate(x0, y1);
            double z10 = 0.0;
            double z11 = 0.0;
            while (x1 < region.getEndX()) {
                z10 = field.evaluate(x1, y0);
                z11 = field.evaluate(x1, y1);
                interpolate(x0, x1, x1, y0, y1, yEnd, z00, z01, z10, z11);
                x0 = x1;
                x1 += xStep;
                z00 = z10;
                z01 = z11;
            }
            {
                x1 = region.getMaxX();
                z10 = field.evaluate(x1, y0);
                z11 = field.evaluate(x1, y1);
                interpolate(x0, x1, region.getEndX(), y0, y1, yEnd, z00, z01, z10, z11);
            }
        };

        int y0 = region.getBeginY();
        int y1 = y0 + yStep;
        while (y1 < region.getEndY()) {
            processRowBlock(y0, y1, y1);
            y0 = y1;
            y1 += yStep;
        }
        {
            y1 = region.getMaxY();
            processRowBlock(y0, y1, region.getEndY());
        }

    } else {
        for (int y = region.getBeginY(), yEnd=region.getEndY(); y < yEnd; ++y) {
            auto rowIter = img.x_at(region.getBeginX() - img.getX0(), y - img.getY0());
            for (int x = region.getBeginX(), xEnd=region.getEndX(); x < xEnd; ++x, ++rowIter) {
                functor(*rowIter, field.evaluate(x, y));
            }
        }
    }
}

}  // anonymous

std::shared_ptr<BoundedField> operator*(double const scale, std::shared_ptr<BoundedField const> bf) {
    return *bf * scale;
}

template <typename T>
void BoundedField::fillImage(image::Image<T> &img, bool overlapOnly, int xStep, int yStep) const {
    applyToImage(*this, img, Assign(), overlapOnly, xStep, yStep);
}

template <typename T>
void BoundedField::addToImage(image::Image<T> &img, double scaleBy, bool overlapOnly,
                              int xStep, int yStep) const {
    applyToImage(*this, img, ScaledAdd(scaleBy), overlapOnly, xStep, yStep);
}

template <typename T>
void BoundedField::multiplyImage(image::Image<T> &img, bool overlapOnly, int xStep, int yStep) const {
    applyToImage(*this, img, Multiply(), overlapOnly, xStep, yStep);
}

template <typename T>
void BoundedField::divideImage(image::Image<T> &img, bool overlapOnly, int xStep, int yStep) const {
    applyToImage(*this, img, Divide(), overlapOnly, xStep, yStep);
}

#define INSTANTIATE(T)                                                             \
    template void BoundedField::fillImage(image::Image<T> &, bool, int, int) const;          \
    template void BoundedField::addToImage(image::Image<T> &, double, bool, int, int) const; \
    template void BoundedField::multiplyImage(image::Image<T> &, bool, int, int) const;      \
    template void BoundedField::divideImage(image::Image<T> &, bool, int, int) const

INSTANTIATE(float);
INSTANTIATE(double);

}
}
}  // namespace lsst::afw::math
