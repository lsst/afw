// -*- lsst-c++ -*-
#ifndef LSST_AFW_REGION_Span_h_INCLUDED
#define LSST_AFW_REGION_Span_h_INCLUDED

#include "lsst/afw/geom/Point.h"

namespace lsst { namespace afw { namespace region {

class Span {
public:

    /// @brief Construct an empty span.
    Span() : _min(), _width(0) {}

    /**
     *  @brief Construct from y and inclusive x bounds.
     *
     *  x0 and x1 will be swapped if x1 < x0.
     */
    Span(int y, int x0, int x1) : _min(x0, y), _width((x1 - x0) + 1) {
        if (x1 < x0) {
            _min.setX(x1);
            _width = (x0 - x1) + 1;
        }
    }

    /**
     *  @brief Construct from the minimum point and width.
     *
     *  Passing a negative width will make an empty Span (which may not preserve
     *  the minimum point and exact width passed in);
     */
    Span(geom::Point2I const & min, int width) : _min(min), _width(width) {
        if (_width <= 0) {
            _min = geom::Point2I();
            _width = 0;
        }
    }

    //@{
    /// @brief Accessors to match the original Span class.
    int getX0() const { return _min.getX(); }
    int getX1() const { return _min.getX() + _width - 1; }
    int getY() const { return _min.getY(); }
    int getWidth() const { return _width; }
    //@}

    //@{
    /// @brief Accessors to match afw::geom::Box2I.
    geom::Point2I const & getMin() const { return _min; }
    geom::Point2I const getMax() const { return Point2I(_min.getX() + _width - 1, _min.getY()); }
    geom::Point2I const & getBegin() const { return _min; }
    geom::Point2I const getEnd() const { return Point2I(_min.getX() + _width, _min.getY()); }
    int getMinX() const { return _min.getX(); }
    int getMaxX() const { return _min.getX() + _width - 1; }
    int getBeginX() const { return _min.get(); }
    int getEndX() const { return _min.get() + _width; }
    int getMinY() const { return _min.getY(); }
    int getMaxY() const { return _min.getY(); }
    int getBeginY() const { return _min.getY(); }
    int getEndY() const { return _min.getY() + 1; }
    //@}

    /// @brief Shift the position of the span by the given offset.
    void shift(geom::Extent2I const & offset) { _min += offset; }
    
    /// @brief Return true if the span contains the point.
    bool contains(geom::Point2I const & point) {
        return (point.getY() == _min.getY())
            && (point.getX() >= _min.getX())
            && (point.getX() < _min.getX() + _width);
    }

    /// @brief Return true if the span contains no points.
    bool isEmpty() const { _width == 0; }

    /// @brief Equality operator.
    bool operator==(Span const & other) const {
        return _min == other._min && _width == other._width;
    }

    /// @brief Inequality operator.
    bool operator!=(Span const & other) const {
        return !this->operator==(other);
    }

private:
    afw::geom::Point2I _min;
    int _width;
};

}}} // namespace lsst::afw::region

#endif // !LSST_AFW_REGION_Span_h_INCLUDED
