#if !defined(FPPOSITION_H)
#define FPPOSITION_H

#include <limits>
#include <iostream>
#include <cmath>

#include "lsst/afw/geom.h"

namespace lsst { namespace afw { namespace cameraGeom {

/**
 * A class representing a position in the focal plane
 * This is a lightweight wrapper for a Point2D to enforce typesafety
 * where focal plane coordinates (mm vs. pixels) are concerned.
 */
class FpPoint {
public:
    /** Construct an FpPoint with the specified value (interpreted in the given units) */
    explicit FpPoint(lsst::afw::geom::Point2D p) : _p(p) {}
    explicit FpPoint(lsst::afw::geom::Extent2D p) : _p(p) {}
    FpPoint(double x, double y) : _p(x, y) {}
    FpPoint() : _p(lsst::afw::geom::Point2D()) {}

    // Return a position in mm from the boresight
    lsst::afw::geom::Point2D getMm() const { return _p; }
    // Return a position in pixels wrt the boresight
    lsst::afw::geom::Point2D getPixels(double pixelSize) const {
        lsst::afw::geom::Point2D pix(_p);
        pix.scale(1.0/pixelSize);
        return pix;
    }

    FpPoint operator+(FpPoint const &pos2) const {
        return FpPoint(this->_p + lsst::afw::geom::Extent2D(pos2.getMm()));
    }
    FpPoint operator-(FpPoint const &pos2) const {
        return FpPoint(this->_p - lsst::afw::geom::Extent2D(pos2.getMm()));
    }
    FpPoint operator*(double const val) const {
        return FpPoint(this->_p.getX()*val, this->_p.getY()*val);
    }
    FpPoint operator/(double const val) const {
        return FpPoint(this->_p.getX()/val, this->_p.getY()/val);
    }
private:
    lsst::afw::geom::Point2D _p; // store this in mm
};


/**
 * A class representing a dimension in the focal plane
 * This is a lightweight wrapper for an Extent2D to enforce typesafety
 * where focal plane coordinates (mm vs. pixels) are concerned.
 */
class FpExtent {
public:
    /** Construct an FpExtent with the specified value (interpreted in the given units) */
    explicit FpExtent(lsst::afw::geom::Point2D p) : _e(p) {}
    explicit FpExtent(lsst::afw::geom::Extent2D e) : _e(e) {}
    FpExtent(double x, double y) : _e(x, y) {}
    FpExtent() : _e(lsst::afw::geom::Extent2D()) {}

    // Return a position in mm from the boresight
    lsst::afw::geom::Extent2D getMm() const { return _e; }
    // Return a position in pixels wrt the boresight
    lsst::afw::geom::Extent2D getPixels(double pixelSize) const {
        return _e/pixelSize;
    }

    FpExtent operator+(FpExtent const &pos2) const {
        return FpExtent(this->_e + lsst::afw::geom::Extent2D(pos2.getMm()));
    }
    FpExtent operator-(FpExtent const &pos2) const {
        return FpExtent(this->_e - lsst::afw::geom::Extent2D(pos2.getMm()));
    }
    FpExtent operator*(double const val) const {
        return FpExtent(this->_e*val);
    }
    FpExtent operator/(double const val) const {
        return FpExtent(this->_e/val);
    }
private:
    lsst::afw::geom::Extent2D _e; // store this in mm
};
            

}}}
#endif
