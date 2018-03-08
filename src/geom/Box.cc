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

#include <cmath>
#include <limits>

#include "lsst/afw/geom/Box.h"

namespace lsst {
namespace afw {
namespace geom {

Box2I::Box2I(Point2I const& minimum, Point2I const& maximum, bool invert)
        : _minimum(minimum), _dimensions(maximum - minimum) {
    for (int n = 0; n < 2; ++n) {
        if (_dimensions[n] < 0) {
            if (invert) {
                _minimum[n] += _dimensions[n];
                _dimensions[n] = -_dimensions[n];
            } else {
                *this = Box2I();
                return;
            }
        }
    }
    _dimensions += Extent2I(1);
}

Box2I::Box2I(Point2I const& minimum, Extent2I const& dimensions, bool invert)
        : _minimum(minimum), _dimensions(dimensions) {
    for (int n = 0; n < 2; ++n) {
        if (_dimensions[n] == 0) {
            *this = Box2I();
            return;
        } else if (_dimensions[n] < 0) {
            if (invert) {
                _minimum[n] += (_dimensions[n] + 1);
                _dimensions[n] = -_dimensions[n];
            } else {
                *this = Box2I();
                return;
            }
        }
    }
    if (!isEmpty() && any(getMin().gt(getMax()))) {
        throw LSST_EXCEPT(pex::exceptions::OverflowError,
                          "Box dimensions too large; integer overflow detected.");
    }
}

Box2I::Box2I(Box2D const& other, EdgeHandlingEnum edgeHandling) : _minimum(), _dimensions() {
    if (other.isEmpty()) {
        *this = Box2I();
        return;
    }
    if (!std::isfinite(other.getMinX()) || !std::isfinite(other.getMinY()) ||
        !std::isfinite(other.getMaxX()) || !std::isfinite(other.getMaxY())) {
        throw LSST_EXCEPT(pex::exceptions::InvalidParameterError, "Cannot convert non-finite Box2D to Box2I");
    }
    Point2D fpMin(other.getMin() + Extent2D(0.5));
    Point2D fpMax(other.getMax() - Extent2D(0.5));
    switch (edgeHandling) {
        case EXPAND:
            for (int n = 0; n < 2; ++n) {
                _minimum[n] = static_cast<int>(std::floor(fpMin[n]));
                _dimensions[n] = static_cast<int>(std::ceil(fpMax[n])) + 1 - _minimum[n];
            }
            break;
        case SHRINK:
            for (int n = 0; n < 2; ++n) {
                _minimum[n] = static_cast<int>(std::ceil(fpMin[n]));
                _dimensions[n] = static_cast<int>(std::floor(fpMax[n])) + 1 - _minimum[n];
            }
            break;
    }
}

ndarray::View<boost::fusion::vector2<ndarray::index::Range, ndarray::index::Range> > Box2I::getSlices()
        const {
    return ndarray::view(getBeginY(), getEndY())(getBeginX(), getEndX());
}

bool Box2I::contains(Point2I const& point) const {
    return all(point.ge(this->getMin())) && all(point.le(this->getMax()));
}

bool Box2I::contains(Box2I const& other) const {
    return other.isEmpty() ||
           (all(other.getMin().ge(this->getMin())) && all(other.getMax().le(this->getMax())));
}

bool Box2I::overlaps(Box2I const& other) const {
    return !(other.isEmpty() || this->isEmpty() || any(other.getMax().lt(this->getMin())) ||
             any(other.getMin().gt(this->getMax())));
}

void Box2I::grow(Extent2I const& buffer) {
    if (isEmpty()) return;  // should we throw an exception here instead of a no-op?
    _minimum -= buffer;
    _dimensions += buffer * 2;
    if (any(_dimensions.le(0))) *this = Box2I();
}

void Box2I::shift(Extent2I const& offset) {
    if (isEmpty()) return;  // should we throw an exception here instead of a no-op?
    _minimum += offset;
}

void Box2I::flipLR(int xextent) {
    if (isEmpty()) return;  // should we throw an exception here instead of a no-op?
    // Apply flip about y-axis assumine parent coordinate system
    _minimum[0] = xextent - (_minimum[0] + _dimensions[0]);
    // _dimensions should remain unchanged
}

void Box2I::flipTB(int yextent) {
    if (isEmpty()) return;  // should we throw an exception here instead of a no-op?
    // Apply flip about y-axis assumine parent coordinate system
    _minimum[1] = yextent - (_minimum[1] + _dimensions[1]);
    // _dimensions should remain unchanged
}

void Box2I::include(Point2I const& point) {
    if (isEmpty()) {
        _minimum = point;
        _dimensions = Extent2I(1);
        return;
    }
    Point2I maximum(getMax());
    for (int n = 0; n < 2; ++n) {
        if (point[n] < _minimum[n]) {
            _minimum[n] = point[n];
        } else if (point[n] > maximum[n]) {
            maximum[n] = point[n];
        }
    }
    _dimensions = Extent2I(1) + maximum - _minimum;
}

void Box2I::include(Box2I const& other) {
    if (other.isEmpty()) return;
    if (this->isEmpty()) {
        *this = other;
        return;
    }
    Point2I maximum(getMax());
    Point2I const& otherMin = other.getMin();
    Point2I const otherMax = other.getMax();
    for (int n = 0; n < 2; ++n) {
        if (otherMin[n] < _minimum[n]) {
            _minimum[n] = otherMin[n];
        }
        if (otherMax[n] > maximum[n]) {
            maximum[n] = otherMax[n];
        }
    }
    _dimensions = Extent2I(1) + maximum - _minimum;
}

void Box2I::clip(Box2I const& other) {
    if (isEmpty()) return;
    if (other.isEmpty()) {
        *this = Box2I();
        return;
    }
    Point2I maximum(getMax());
    Point2I const& otherMin = other.getMin();
    Point2I const otherMax = other.getMax();
    for (int n = 0; n < 2; ++n) {
        if (otherMin[n] > _minimum[n]) {
            _minimum[n] = otherMin[n];
        }
        if (otherMax[n] < maximum[n]) {
            maximum[n] = otherMax[n];
        }
    }
    if (any(maximum.lt(_minimum))) {
        *this = Box2I();
        return;
    }
    _dimensions = Extent2I(1) + maximum - _minimum;
}

bool Box2I::operator==(Box2I const& other) const {
    return other._minimum == this->_minimum && other._dimensions == this->_dimensions;
}

bool Box2I::operator!=(Box2I const& other) const {
    return other._minimum != this->_minimum || other._dimensions != this->_dimensions;
}

std::vector<Point2I> Box2I::getCorners() const {
    std::vector<Point2I> retVec;
    retVec.push_back(getMin());
    retVec.push_back(Point2I(getMaxX(), getMinY()));
    retVec.push_back(getMax());
    retVec.push_back(Point2I(getMinX(), getMaxY()));
    return retVec;
}

double const Box2D::EPSILON = std::numeric_limits<double>::epsilon() * 2;

double const Box2D::INVALID = std::numeric_limits<double>::quiet_NaN();

Box2D::Box2D() : _minimum(INVALID), _maximum(INVALID) {}

Box2D::Box2D(Point2D const& minimum, Point2D const& maximum, bool invert)
        : _minimum(minimum), _maximum(maximum) {
    for (int n = 0; n < 2; ++n) {
        if (_minimum[n] == _maximum[n]) {
            *this = Box2D();
            return;
        } else if (_minimum[n] > _maximum[n]) {
            if (invert) {
                std::swap(_minimum[n], _maximum[n]);
            } else {
                *this = Box2D();
                return;
            }
        }
    }
}

Box2D::Box2D(Point2D const& minimum, Extent2D const& dimensions, bool invert)
        : _minimum(minimum), _maximum(minimum + dimensions) {
    for (int n = 0; n < 2; ++n) {
        if (_minimum[n] == _maximum[n]) {
            *this = Box2D();
            return;
        } else if (_minimum[n] > _maximum[n]) {
            if (invert) {
                std::swap(_minimum[n], _maximum[n]);
            } else {
                *this = Box2D();
                return;
            }
        }
    }
}

Box2D::Box2D(Box2I const& other)
        : _minimum(Point2D(other.getMin()) - Extent2D(0.5)),
          _maximum(Point2D(other.getMax()) + Extent2D(0.5)) {
    if (other.isEmpty()) *this = Box2D();
}

bool Box2D::contains(Point2D const& point) const {
    return all(point.ge(this->getMin())) && all(point.lt(this->getMax()));
}

bool Box2D::contains(Box2D const& other) const {
    return other.isEmpty() ||
           (all(other.getMin().ge(this->getMin())) && all(other.getMax().le(this->getMax())));
}

bool Box2D::overlaps(Box2D const& other) const {
    return !(other.isEmpty() || this->isEmpty() || any(other.getMax().le(this->getMin())) ||
             any(other.getMin().ge(this->getMax())));
}

void Box2D::grow(Extent2D const& buffer) {
    if (isEmpty()) return;  // should we throw an exception here instead of a no-op?
    _minimum -= buffer;
    _maximum += buffer;
    if (any(_minimum.ge(_maximum))) *this = Box2D();
}

void Box2D::shift(Extent2D const& offset) {
    if (isEmpty()) return;  // should we throw an exception here instead of a no-op?
    _minimum += offset;
    _maximum += offset;
}

void Box2D::flipLR(float xextent) {
    if (isEmpty()) return;  // should we throw an exception here instead of a no-op?
    // Swap min and max values for x dimension
    _minimum[0] += _maximum[0];
    _maximum[0] = _minimum[0] - _maximum[0];
    _minimum[0] -= _maximum[0];
    // Apply flip assuming coordinate system of parent.
    _minimum[0] = xextent - _minimum[0];
    _maximum[0] = xextent - _maximum[0];
    // _dimensions should remain unchanged
}

void Box2D::flipTB(float yextent) {
    if (isEmpty()) return;  // should we throw an exception here instead of a no-op?
    // Swap min and max values for y dimension
    _minimum[1] += _maximum[1];
    _maximum[1] = _minimum[1] - _maximum[1];
    _minimum[1] -= _maximum[1];
    // Apply flip assuming coordinate system of parent.
    _minimum[1] = yextent - _minimum[1];
    _maximum[1] = yextent - _maximum[1];
    // _dimensions should remain unchanged
}

void Box2D::include(Point2D const& point) {
    if (isEmpty()) {
        _minimum = point;
        _maximum = point;
        _tweakMax(0);
        _tweakMax(1);
        return;
    }
    for (int n = 0; n < 2; ++n) {
        if (point[n] < _minimum[n]) {
            _minimum[n] = point[n];
        } else if (point[n] >= _maximum[n]) {
            _maximum[n] = point[n];
            _tweakMax(n);
        }
    }
}

void Box2D::include(Box2D const& other) {
    if (other.isEmpty()) return;
    if (this->isEmpty()) {
        *this = other;
        return;
    }
    Point2D const& otherMin = other.getMin();
    Point2D const& otherMax = other.getMax();
    for (int n = 0; n < 2; ++n) {
        if (otherMin[n] < _minimum[n]) {
            _minimum[n] = otherMin[n];
        }
        if (otherMax[n] > _maximum[n]) {
            _maximum[n] = otherMax[n];
        }
    }
}

void Box2D::clip(Box2D const& other) {
    if (isEmpty()) return;
    if (other.isEmpty()) {
        *this = Box2D();
        return;
    }
    Point2D const& otherMin = other.getMin();
    Point2D const& otherMax = other.getMax();
    for (int n = 0; n < 2; ++n) {
        if (otherMin[n] > _minimum[n]) {
            _minimum[n] = otherMin[n];
        }
        if (otherMax[n] < _maximum[n]) {
            _maximum[n] = otherMax[n];
        }
    }
    if (any(_maximum.le(_minimum))) {
        *this = Box2D();
        return;
    }
}

bool Box2D::operator==(Box2D const& other) const {
    return (other.isEmpty() && this->isEmpty()) ||
           (other._minimum == this->_minimum && other._maximum == this->_maximum);
}

bool Box2D::operator!=(Box2D const& other) const {
    return !(other.isEmpty() && other.isEmpty()) &&
           (other._minimum != this->_minimum || other._maximum != this->_maximum);
}

std::vector<Point2D> Box2D::getCorners() const {
    std::vector<Point2D> retVec;
    retVec.push_back(getMin());
    retVec.push_back(Point2D(getMaxX(), getMinY()));
    retVec.push_back(getMax());
    retVec.push_back(Point2D(getMinX(), getMaxY()));
    return retVec;
}

std::ostream& operator<<(std::ostream& os, Box2I const& box) {
    if (box.isEmpty()) return os << "Box2I()";
    return os << "Box2I(Point2I" << box.getMin() << ", Extent2I" << box.getDimensions() << ")";
}

std::ostream& operator<<(std::ostream& os, Box2D const& box) {
    if (box.isEmpty()) return os << "Box2D()";
    return os << "Box2D(Point2D" << box.getMin() << ", Extent2D" << box.getDimensions() << ")";
}
}  // namespace geom
}  // namespace afw
}  // namespace lsst
