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

#include <numeric>

#include "lsst/pex/exceptions.h"
#include "lsst/afw/math/BoundedField.h"
#include "lsst/afw/table/io/Persistable.cc"
#include "lsst/afw/image/ImageUtils.h"

namespace lsst {
namespace afw {

template std::shared_ptr<math::BoundedField> table::io::PersistableFacade<math::BoundedField>::dynamicCast(
        std::shared_ptr<table::io::Persistable> const &);

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
    template <typename T, typename U>
    void operator()(T &out, U a) const {
        out = a;
    }
};

struct ScaledAdd {
    explicit ScaledAdd(double s) : scaleBy(s) {}

    template <typename T, typename U>
    void operator()(T &out, U a) const {
        out += scaleBy * a;
    }

    double scaleBy;
};

struct Multiply {
    template <typename T, typename U>
    void operator()(T &out, U a) const {
        out *= a;
    }
};

struct Divide {
    template <typename T, typename U>
    void operator()(T &out, U a) const {
        out /= a;
    }
};

// Helper class to do bilinear interpolation with no dynamic memory
// allocation. This means we evaluate the function multiple times at some
// points (when we move to a new interval in y, to be specific). Could
// probably be optimized for the cases where either step is 1, but that seems
// premature.
class Interpolator {
public:
    // Description of a cell to interpolate in one dimension.
    struct Bounds {
        int const step;  // step between min and max, except in the special last row/column
        int min;  // lower-bound of cell (coordinate of known value and one before first point to fill in)
        int max;  // upper-bound of cell (coordinate of known value)
        int end;  // upper-bound of cell (one after last point to fill in)

        // Construct from step only.
        //
        // Other variables are initialized (and re-initialized) by calls to reset().
        explicit Bounds(int step_) : step(step_), min(0), max(0), end(0) {}

        // Reset all points (aside from the step) to the first cell in this dimension.
        void reset(int min_) {
            min = min_;
            max = min_ + step;
            end = min_ + step;
        }
    };

    // Construct an object to interpolate the given BoundedField on an evenly-spaced grid within a region.
    Interpolator(BoundedField const *field, lsst::geom::Box2I const *region, int xStep, int yStep)
            : _field(field),
              _region(region),
              _x(xStep),
              _y(yStep),
              _z00(std::numeric_limits<double>::quiet_NaN()),
              _z01(std::numeric_limits<double>::quiet_NaN()),
              _z10(std::numeric_limits<double>::quiet_NaN()),
              _z11(std::numeric_limits<double>::quiet_NaN()) {}

    // Actually do the interpolation.
    //
    // This method iterates over rows of cells in y, calling _runRow()
    // to iterate over cells in x.
    template <typename T, typename F>
    void run(image::Image<T> &img, F functor) {
        _y.reset(_region->getBeginY());
        while (_y.end < _region->getEndY()) {
            _runRow(img, functor);
            _y.min = _y.max;
            _y.max += _y.step;
            _y.end = _y.max;
        }
        {  // special-case last iteration in y
            _y.max = _region->getMaxY();
            _y.end = _region->getEndY();
            _runRow(img, functor);
        }
    }

private:
    // Process a row of cells, calling _runCell() on each one.
    template <typename T, typename F>
    void _runRow(image::Image<T> &img, F functor) {
        _x.reset(_region->getBeginX());
        _z00 = _field->evaluate(_x.min, _y.min);
        _z01 = _field->evaluate(_x.min, _y.max);
        while (_x.max < _region->getEndX()) {
            _z10 = _field->evaluate(_x.max, _y.min);
            _z11 = _field->evaluate(_x.max, _y.max);
            _runCell(img, functor);
            _x.min = _x.max;
            _x.max += _x.step;
            _x.end = _x.max;
            _z00 = _z10;
            _z01 = _z11;
        }
        {  // special-case last iteration in x
            _x.max = _region->getMaxX();
            _x.end = _region->getEndX();
            _z10 = _field->evaluate(_x.max, _y.min);
            _z11 = _field->evaluate(_x.max, _y.max);
            _runCell(img, functor);
        }
    }

    // Interpolate all points in a cell, which is defined as a rectangle for which
    // the BoundedField has been evaluated at all four corners.  The main complication
    // comes from the need to special-case the unusually-sized final cells and final
    // rows/columns within cells in each dimension.
    template <typename T, typename F>
    void _runCell(image::Image<T> const &img, F functor) const {
        int dy = _y.max - _y.min;
        int dx = _x.max - _x.min;
        // First iteration of each of the for loops below has been
        // split into an explicit block.  This avoids a few instructions
        // when no interpolation is necessary, but more importantly it
        // allows us to handle dx==0 and dy==0 with no divide-by-zero
        // problems.
        {  // y=_y.min
            auto rowIter = img.x_at(_x.min - img.getX0(), _y.min - img.getY0());
            {  // x=_x.min
                functor(*rowIter, _z00);
                ++rowIter;
            }
            for (int x = _x.min + 1; x < _x.end; ++x, ++rowIter) {
                functor(*rowIter, ((_x.max - x) * _z00 + (x - _x.min) * _z10) / dx);
            }
        }
        for (int y = _y.min + 1; y < _y.end; ++y) {
            auto rowIter = img.x_at(_x.min - img.getX0(), y - img.getY0());
            double z0 = ((_y.max - y) * _z00 + (y - _y.min) * _z01) / dy;
            double z1 = ((_y.max - y) * _z10 + (y - _y.min) * _z11) / dy;
            {  // x=_x.min
                functor(*rowIter, z0);
                ++rowIter;
            }
            for (int x = _x.min + 1; x < _x.end; ++x, ++rowIter) {
                functor(*rowIter, ((_x.max - x) * z0 + (x - _x.min) * z1) / dx);
            }
        }
    }

    BoundedField const *_field;
    lsst::geom::Box2I const *_region;
    Bounds _x;
    Bounds _y;
    double _z00, _z01, _z10, _z11;
};

template <typename T>
struct ImageRowGetter {
    int y0;
    ndarray::Array<T, 2, 1> array;

    ndarray::ArrayRef<T, 1, 1> get_row(int y) const {
        return array[y - y0];
    }
};

template <typename T>
struct MaskedImageRow {
    ndarray::ArrayRef<T, 1, 1> image_row;
    ndarray::ArrayRef<float, 1, 1> variance_row;

    template <typename U>
    MaskedImageRow & operator*=(U a) {
        image_row *= a;
        variance_row *= a;
        variance_row *= a;
        return *this;
    }

    template <typename U>
    MaskedImageRow & operator/=(U a) {
        image_row /= a;
        variance_row /= a;
        variance_row /= a;
        return *this;
    }
};

template <typename T>
struct MaskedImageRowGetter {
    int y0;
    ndarray::Array<T, 2, 1> image_array;
    ndarray::Array<float, 2, 1> variance_array;

    MaskedImageRow<T> get_row(int y) const {
        return MaskedImageRow<T> { image_array[y - y0], variance_array[y - y0] };
    }
};

template <typename T>
ImageRowGetter<T> make_row_getter(image::Image<T> & im) {
    return ImageRowGetter<T> { im.getY0(), im.getArray() };
}

template <typename T>
MaskedImageRowGetter<T> make_row_getter(image::MaskedImage<T> & mi) {
    return MaskedImageRowGetter<T> { mi.getY0(), mi.getImage()->getArray(), mi.getVariance()->getArray() };
}

template <typename ImageT, typename F>
void applyToImage(BoundedField const &field, ImageT &img, F functor, bool overlapOnly, int xStep,
                  int yStep) {
    lsst::geom::Box2I region(field.getBBox());
    if (overlapOnly) {
        region.clip(img.getBBox(image::PARENT));
    } else if (region != img.getBBox(image::PARENT)) {
        throw LSST_EXCEPT(pex::exceptions::RuntimeError,
                          "Image bounding box does not match field bounding box");
    }

    if (yStep > 1 || xStep > 1) {
        if constexpr (std::is_scalar_v<typename ImageT::Pixel>) {
            Interpolator interpolator(&field, &region, xStep, yStep);
            interpolator.run(img, functor);
        } else {
            throw LSST_EXCEPT(
                pex::exceptions::LogicError,
                "No interpolation with MaskedImage."
            );
        }
    } else {
        // We iterate over rows as a significant optimization for AST-backed bounded fields
        // (it's also slightly faster for other bounded fields, too).
        auto subImage = img.subset(region);
        auto size = region.getWidth();
        ndarray::Array<double, 1> xx = ndarray::allocate(ndarray::makeVector(size));
        ndarray::Array<double, 1> yy = ndarray::allocate(ndarray::makeVector(size));
        // y gets incremented each outer loop, x is always xMin->xMax
        std::iota(xx.begin(), xx.end(), region.getBeginX());
        auto row_getter = make_row_getter(subImage);
        for (int y = region.getBeginY(); y < region.getEndY(); ++y) {
            yy.deep() = y;  // don't need indexToPosition, as we're already working in the right box (region).
            // We need to make 'row' a temporary variable so it can be an
            // lvalue reference in the functor; it needs to be an lvalue
            // reference functor for the interpolation code path that passes
            // it a single value.
            auto row = row_getter.get_row(y);
            functor(row, field.evaluate(xx, yy));
        }
    }
}

}  // namespace

std::shared_ptr<BoundedField> operator*(double const scale, std::shared_ptr<BoundedField const> bf) {
    return *bf * scale;
}

template <typename T>
void BoundedField::fillImage(image::Image<T> &img, bool overlapOnly, int xStep, int yStep) const {
    applyToImage(*this, img, Assign(), overlapOnly, xStep, yStep);
}

template <typename T>
void BoundedField::addToImage(image::Image<T> &img, double scaleBy, bool overlapOnly, int xStep,
                              int yStep) const {
    applyToImage(*this, img, ScaledAdd(scaleBy), overlapOnly, xStep, yStep);
}

template <typename T>
void BoundedField::multiplyImage(image::Image<T> &img, bool overlapOnly, int xStep, int yStep) const {
    applyToImage(*this, img, Multiply(), overlapOnly, xStep, yStep);
}

template <typename T>
void BoundedField::multiplyImage(image::MaskedImage<T> &img, bool overlapOnly) const {
    applyToImage(*this, img, Multiply(), overlapOnly, 1, 1);
}

template <typename T>
void BoundedField::divideImage(image::Image<T> &img, bool overlapOnly, int xStep, int yStep) const {
    applyToImage(*this, img, Divide(), overlapOnly, xStep, yStep);
}

template <typename T>
void BoundedField::divideImage(image::MaskedImage<T> &img, bool overlapOnly) const {
    applyToImage(*this, img, Divide(), overlapOnly, 1, 1);
}

#define INSTANTIATE(T)                                                                       \
    template void BoundedField::fillImage(image::Image<T> &, bool, int, int) const;          \
    template void BoundedField::addToImage(image::Image<T> &, double, bool, int, int) const; \
    template void BoundedField::multiplyImage(image::Image<T> &, bool, int, int) const;      \
    template void BoundedField::multiplyImage(image::MaskedImage<T> &, bool) const;      \
    template void BoundedField::divideImage(image::Image<T> &, bool, int, int) const; \
    template void BoundedField::divideImage(image::MaskedImage<T> &, bool) const

INSTANTIATE(float);
INSTANTIATE(double);

}  // namespace math
}  // namespace afw
}  // namespace lsst
