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

class BoxI {
public:

    enum EdgeHandlingEnum { EXPAND, SHRINK };

    /// \brief Construct an empty box.
    BoxI() : _minimum(0), _dimensions(0) {}

    BoxI(PointI const & minimum, PointI const & maximum, bool invert=true);
    BoxI(PointI const & minimum, ExtentI const & dimensions, bool invert=true);

    explicit BoxI(BoxD const & other, EdgeHandlingEnum edgeHandling=EXPAND);
    
    PointI const & getMin() const { return _minimum; }
    int getMinX() const { return _minimum.getX(); }
    int getMinY() const { return _minimum.getY(); }

    PointI const getMax() const { return _minimum + _dimensions - ExtentI(1); }
    int getMinX() const { return _minimum.getX() + _dimensions.getX() - 1; }
    int getMinY() const { return _minimum.getY() + _dimensions.getY() - 1; }

    PointI const & getBegin() const { return _minimum; }
    int getBeginX() const { return _minimum.getX(); }
    int getBeginY() const { return _minimum.getY(); }

    PointI const getEnd() const { return _minimum + _dimensions; }
    int getEndX() const { return _minimum.getX() + _dimensions.getX(); }
    int getEndY() const { return _minimum.getY() + _dimensions.getY(); }

    ExtentI const & getDimensions() const { return _dimensions; }
    int getWidth() const { return _dimensions.getX(); }
    int getHeight() const { return _dimensions.getY(); }
    int getArea() const { return _getWidth() * _getHeight(); }

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
    void normalize(bool invert);

    PointI _minimum;
    ExtentI _dimensions;
};

class BoxD {
public:

    BoxD();
    BoxD(PointD const & minimum, PointD const & maximum, bool invert=true);    
    BoxD(PointD const & minimum, ExtentD const & dimensions, bool invert=true);
    
    explicit BoxD(BoxI const & box);
    
    PointD const & getMin() const { return _minimum; }
    double getMinX() const { return _minimum.getX(); }
    double getMinY() const { return _minimum.getY(); }

    PointD const getMax() const { return _minimum + _dimensions - ExtentD(1); }
    double getMinX() const { return _minimum.getX() + _dimensions.getX(); }
    double getMinY() const { return _minimum.getY() + _dimensions.getY(); }

    ExtentD const & getDimensions() const { return _dimensions; }
    double getWidth() const { return _dimensions.getX(); }
    double getHeight() const { return _dimensions.getY(); }
    double getArea() const { return getWidth() * getHeight(); }

    PointD const getCenter() const { return _minimum + _dimensions*0.5; }
    double getCenterX() const { return _minimum.getX() + _dimensions.getX()*0.5; }
    double getCenterY() const { return _minimum.getY() + _dimensions.getY()*0.5; }

    bool isEmpty() const { return _dimensions.getX() == 0; }

    bool contains(PointD const & point) const;
    bool contains(BoxD const & other) const;
    bool overlaps(BoxD const & other) const;

    void grow(double buffer);
    void grow(ExtentD const & buffer);
    void scale(double factor);
    void shift(ExtentD const & offset);
    void include(PointD const & point);
    void include(BoxD const & other);
    void clip(BoxD const & other);

private:
    void normalize(bool invert);

    PointD _minimum;
    ExtentD _dimensions;
};

}}}

#endif
