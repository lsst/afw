#include <cmath>

#include "lsst/afw/geom/Box.h"

namespace geom = lsst::afw::geom;

/**
 *  @brief Construct a box from its minimum and maximum points (inclusive).
 *
 *  @param(in) minimum   Minimum (lower left) coordinate.
 *  @param(in) maximum   Maximum (upper right) coordinate.
 *  @param(in) invert    If true (default), swap the minimum and maximum coordinates if
 *                       minimum > maximum instead of creating an empty box.
 */
geom::BoxI::BoxI(PointI const & minimum, PointI const & maximum, bool invert) :
    _minimum(minimum), _dimensions(maximum - minimum)
{
    for (int n=0; n<2; ++n) {
        if (_dimensions[n] < 0) {
            if (invert) {
                _minimum[n] += _dimensions[n];
                _dimensions[n] = -_dimensions[n];
            } else {
                *this = BoxI();
                return;
            }
        }
    }
    _dimensions += ExtentI(1);
}

/**
 *  @brief Construct a box from its minimum point and dimensions.
 *
 *  @param(in) minimum    Minimum (lower left) coordinate.
 *  @param(in) dimensions Box dimensions.  If either dimension coordinate is 0, the box will be empty.
 *  @param(in) invert     If true (default), invert any negative dimensions instead of creating 
 *                        an empty box.
 */
geom::BoxI::BoxI(PointI const & minimum, ExtentI const & dimensions, bool invert) :
    _minimum(minimum), _dimensions(dimensions)
{
    for (int n=0; n<2; ++n) {
        if (_dimensions[n] == 0) {
            *this = BoxI();
            return;
        } else if (_dimensions[n] < 0) {
            if (invert) {
                _minimum[n] += (_dimensions[n] + 1);
                _dimensions[n] = -_dimensions[n];
            } else {
                *this = BoxI();
                return;
            }
        }
    }
}

/**
 *  @brief Construct an integer box from a floating-point box.
 *
 *  Floating-point to integer box conversion is based on the concept that a pixel
 *  is not an infinitesimal point but rather a square of unit size centered on
 *  integer-valued coordinates.  Converting a floating-point box to an integer box
 *  thus requires a choice on how to handle pixels which are only partially contained
 *  by the input floating-point box.
 *
 *  @param(in) other          A floating-point box to convert.
 *  @param(in) edgeHandling   If EXPAND, the integer box will contain any pixels that
 *                            overlap the floating-point box.  If SHRINK, the integer
 *                            box will contain only pixels completely contained by
 *                            the floating-point box.
 */
geom::BoxI::BoxI(BoxD const & other, EdgeHandlingEnum edgeHandling) : _minimum(), _dimensions() {
    PointD fpMin(other.getMin() + ExtentD(0.5));
    PointD fpMax(other.getMax() - ExtentD(0.5));
    switch (edgeHandling) {
    case EXPAND:
        for (int n=0; n<2; ++N) {
            _minimum[n] = static_cast<int>(std::floor(fpMin[n]));
            _dimensions[n] = static_cast<int>(std::ceil(fpMax)) + 1 - _minimum[n];
        }
        break;
    case SHRINK:
        for (int n=0; n<2; ++N) {
            _minimum[n] = static_cast<int>(std::ceil(fpMin[n]));
            _dimensions[n] = static_cast<int>(std::floor(fpMax)) + 1 - _minimum[n];
        }
        break;
    }
}

/// \brief Return true if the box contains the point.
bool geom::BoxI::contains(PointI const & point) const {
    return all(point >= this->getMin()) && all(point <= this->getMax());
}

/**
 *  \brief Return true if all points contained by other are also contained by this.
 *
 *  An empty box is contained by every other box, including other empty boxes.
 */
bool geom::BoxI::contains(BoxI const & other) const {
    return other.isEmpty() || 
        (all(other.getMin() >= this->getMin()) && all(other.getMax() <= this->getMax()));
}

/**
 *  \brief Return true if any points in other are also in this.
 *
 *  Any overlap operation involving an empty box returns false.
 */
bool geom::BoxI::overlaps(BoxI const & other) const {
    return !(
        other.isEmpty() || this->isEmpty() 
        || any(other.getMax() < this->getMin()) 
        || any(other.getMin() > this->getMax())
    );
}

/**
 *  \brief Increase the size of the box by the given buffer amount in all directions.
 *
 *  If a negative buffer is passed and the final size of the box is less than or
 *  equal to zero, the box will be made empty.
 */
void geom::BoxI::grow(ExtentI const & buffer) {
    if (isEmpty()) return; // should we throw an exception here instead of a no-op?
    _minimum -= buffer;
    _dimensions += buffer * 2;
    if (any(_dimensions <= 0)) *this = BoxI();
}

/**
 *  \brief Increase the size of the box by the given buffer amount in each direction.
 *
 *  If a negative buffer is passed and the final size of the box is less than or
 *  equal to zero, the box will be made empty.
 */
void geom::BoxI::shift(ExtentI const & offset) {
    if (isEmpty()) return; // should we throw an exception here instead of a no-op?
    _minimum += offset;
}

/// \brief Expand this to ensure that this->contains(point).
void geom::BoxI::include(PointI const & point) {
    if (isEmpty()) {
        _minimum = point;
        _dimensions = ExtentI(1);
        return;
    }
    PointI maximum(getMax());
    for (int n=0; n<2; ++n) {
        if (point[n] < _minimum[n]) {
            _minimum[n] = point[n];
        } else if (point[n] > maximum[n]) {
            maximum[n] = point[n];
        }
    }
    _dimensions = Extent(1) + maximum - _minimum;
}

/// \brief Expand this to ensure that this->contains(other).
void geom::BoxI::include(BoxI const & other) {
    if (other.isEmpty()) return;
    if (this->isEmpty()) {
        *this = other;
        return;
    }
    PointI maximum(getMax());
    PointI const & otherMin = other.getMin();
    PointI const otherMax = other.getMax();
    for (int n=0; n<2; ++n) {
        if (otherMin[n] < _minimum[n]) {
            _minimum[n] = otherMin[n];
        } else if (otherMax[n] > maximum[n]) {
            maximum[n] = otherMax[n];
        }
    }
    _dimensions = Extent(1) + maximum - _minimum;    
}

/// \brief Shrink this to ensure that other.contains(*this).
void geom::BoxI::clip(BoxI const & other) {
    if (isEmpty()) return;
    if (other.isEmpty()) {
        *this = BoxI();
        return;
    }
    PointI maximum(getMax());
    PointI const & otherMin = other.getMin();
    PointI const otherMax = other.getMax();
    for (int n=0; n<2; ++n) {
        if (otherMin[n] > _minimum[n]) {
            _minimum[n] = otherMin[n];
        } else if (otherMax[n] < maximum[n]) {
            maximum[n] = otherMax[n];
        }
    }
    if (any(maximum < _minimum)) {
        *this = BoxI();
        return;
    }                     
    _dimensions = Extent(1) + maximum - _minimum;    
}
