// -*- lsst-c++ -*-
/**
 * \file
 * \brief A coordinate class intended to represent absolute positions.
 */
#ifndef LSST_AFW_GEOM_BOX_H
#define LSST_AFW_GEOM_BOX_H

#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/Extent.h"

namespace lsst { namespace afw { namespace geom {

class BoxD;

/**
 *  \brief An integer coordinate rectangle.
 *
 *  BoxI is an inclusive box that represents a rectangular region of pixels.  A box 
 *  never has negative dimensions; the empty box is defined to have zero-size dimensions,
 *  and is treated as though it does not have a well-defined position (regardless of the
 *  return value of getMin() or getMax() for an empty box).
 *
 *  \internal
 *
 *  BoxI internally stores its minimum point and dimensions, because we expect
 *  these will be the most commonly accessed quantities.
 *
 *  BoxI sets the minimum point to the origin for an empty box, and returns -1 for both
 *  elements of the maximum point in that case.
 */
class BoxI {
public:

    enum EdgeHandlingEnum { EXPAND, SHRINK };

    /// \brief Construct an empty box.
    BoxI() : _minimum(0), _dimensions(0) {}

    BoxI(PointI const & minimum, PointI const & maximum, bool invert=true);
    BoxI(PointI const & minimum, ExtentI const & dimensions, bool invert=true);

    explicit BoxI(BoxD const & other, EdgeHandlingEnum edgeHandling=EXPAND);

    /// \brief Standard copy constructor.
    BoxI(BoxI const & other) : _minimum(other._minimum), _dimensions(other._dimensions) {}

    /// \brief Standard assignment operator.
    BoxI & operator=(BoxI const & other) {
        _minimum = other._minimum;
        _dimensions = other._dimensions;
        return *this;
    }

    /**
     *  @name Min/Max Accessors
     *
     *  @brief Return the minimum and maximum coordinates of the box (inclusive).
     */
    //@{
    PointI const & getMin() const { return _minimum; }
    int getMinX() const { return _minimum.getX(); }
    int getMinY() const { return _minimum.getY(); }

    PointI const getMax() const { return _minimum + _dimensions - ExtentI(1); }
    int getMaxX() const { return _minimum.getX() + _dimensions.getX() - 1; }
    int getMaxY() const { return _minimum.getY() + _dimensions.getY() - 1; }
    //@}

    /**
     *  @name Begin/End Accessors
     *
     *  \brief Return STL-style begin (inclusive) and end (exclusive) coordinates for the box.
     */
    //@{
    PointI const & getBegin() const { return _minimum; }
    int getBeginX() const { return _minimum.getX(); }
    int getBeginY() const { return _minimum.getY(); }

    PointI const getEnd() const { return _minimum + _dimensions; }
    int getEndX() const { return _minimum.getX() + _dimensions.getX(); }
    int getEndY() const { return _minimum.getY() + _dimensions.getY(); }
    //@}

    /**
     *  @name Size Accessors
     *
     *  \brief Return the size of the box in pixels.
     */
    //@{
    ExtentI const & getDimensions() const { return _dimensions; }
    int getWidth() const { return _dimensions.getX(); }
    int getHeight() const { return _dimensions.getY(); }
    int getArea() const { return getWidth() * getHeight(); }
    //@}

    /// \brief Return true if the box contains no points.
    bool isEmpty() const { return _dimensions.getX() == 0; }

    bool contains(PointI const & point) const;
    bool contains(BoxI const & other) const;
    bool overlaps(BoxI const & other) const;

    void grow(int buffer) { grow(ExtentI(buffer)); }
    void grow(ExtentI const & buffer);
    void shift(ExtentI const & offset);
    void include(PointI const & point);
    void include(BoxI const & other);
    void clip(BoxI const & other);

private:
    PointI _minimum;
    ExtentI _dimensions;
};

/**
 *  \brief A floating-point coordinate rectangle geometry.
 *
 *  BoxD is a half-open (minimum is inclusive, maximum is exclusive) box. A box 
 *  never has negative dimensions; the empty box is defined to zero-size dimensions
 *  and its minimum and maximum values set to NaN.  Only the empty box may have
 *  zero-size dimensions.
 *
 *  \internal
 *
 *  BoxD internally stores its minimum point and maximum point, instead of
 *  minimum point and dimensions, to ensure roundoff error does not affect
 *  whether points are contained by the box.
 *
 *  Despite some recommendations to the contrary, BoxD sets the minimum and maximum
 *  points to NaN for an empty box.  In almost every case, special checks for
 *  emptiness would have been necessary anyhow, so there was little to gain in
 *  using the minimum > maximum condition to denote an empty box, as was used in BoxI.
 */
class BoxD {
public:

    /**
     *  Value the maximum coordinate is multiplied by to increase it by the smallest
     *  possible amount.
     */
    static double const ONE_PLUS_EPSILON;

    /// Value used to specify undefined coordinate values.
    static double const INVALID;

    /// \brief Construct an empty box.
    BoxD();

    BoxD(PointD const & minimum, PointD const & maximum, bool invert=true);    
    BoxD(PointD const & minimum, ExtentD const & dimensions, bool invert=true);
    
    explicit BoxD(BoxI const & other);
    
    /// \brief Standard copy constructor.
    BoxD(BoxD const & other) : _minimum(other._minimum), _maximum(other._maximum) {}

    /// \brief Standard assignment operator.
    BoxD & operator=(BoxD const & other) {
        _minimum = other._minimum;
        _maximum = other._maximum;
        return *this;
    }

    /**
     *  @name Min/Max Accessors
     *
     *  @brief Return the minimum (inclusive) and maximum (exclusive) coordinates of the box.
     */
    //@{
    PointD const & getMin() const { return _minimum; }
    double getMinX() const { return _minimum.getX(); }
    double getMinY() const { return _minimum.getY(); }

    PointD const & getMax() const { return _maximum; }
    double getMaxX() const { return _maximum.getX(); }
    double getMaxY() const { return _maximum.getY(); }
    //@}

    /**
     *  @name Size Accessors
     *
     *  \brief Return the size of the box.
     */
    //@{
    ExtentD const getDimensions() const { return isEmpty() ? ExtentD(0.0) : _maximum - _minimum; }
    double getWidth() const { return isEmpty() ? 0 : _maximum.getX() - _minimum.getX(); }
    double getHeight() const { return isEmpty() ? 0 : _maximum.getY() - _minimum.getY(); }
    double getArea() const {
        ExtentD dim(getDimensions());
        return dim.getX() * dim.getY();
    }
    //@}

    /**
     *  @name Center Accessors
     *
     *  \brief Return the center coordinate of the box.
     */
    //@{
    PointD const getCenter() const { return PointD((_minimum.asVector() + _maximum.asVector())*0.5); }
    double getCenterX() const { return (_minimum.getX() + _maximum.getX())*0.5; }
    double getCenterY() const { return (_minimum.getY() + _maximum.getY())*0.5; }
    //@}

    /// \brief Return true if the box contains no points.
    bool isEmpty() const { return _minimum.getX() != _minimum.getX(); }

    bool contains(PointD const & point) const;
    bool contains(BoxD const & other) const;
    bool overlaps(BoxD const & other) const;

    void grow(double buffer) { grow(ExtentD(buffer)); }
    void grow(ExtentD const & buffer);
    void shift(ExtentD const & offset);
    void include(PointD const & point);
    void include(BoxD const & other);
    void clip(BoxD const & other);

private:
    PointD _minimum;
    PointD _maximum;
};

}}}

#endif
