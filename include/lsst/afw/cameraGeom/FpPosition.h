#if !defined(FPPOSITION_H)
#define FPPOSITION_H

#include <limits>
#include <iostream>
#include <cmath>

#include "lsst/afw/geom.h"

namespace lsst { namespace afw { namespace cameraGeom {

/**
 * A class representing a position in the focal plane
 * This is a lightweight wrapper for a Point2D to ensure that we know what sort of 
 */
class FpPosition {
public:
    /** Construct an Angle with the specified value (interpreted in the given units) */
    explicit FpPosition(lsst::afw::geom::Point2D p) : _p(p) {}
    explicit FpPosition(lsst::afw::geom::Extent2D p) : _p(p) {}
    FpPosition(double x, double y) : _p(x, y) {}
    FpPosition() : _p(lsst::afw::geom::Point2D()) {}

    // Return a position in mm from the boresight
    lsst::afw::geom::Point2D getMm() const { return _p; }
    // Return a position in pixels wrt the boresight
    lsst::afw::geom::Point2D getPixels(double pixelSize) const {
        lsst::afw::geom::Point2D pix(_p);
        pix.scale(1.0/pixelSize);
        return pix;
    }

    FpPosition operator+(FpPosition const &pos2) const {
        return FpPosition(this->_p + lsst::afw::geom::Extent2D(pos2.getMm()));
    }
    FpPosition operator-(FpPosition const &pos2) const {
        return FpPosition(this->_p - lsst::afw::geom::Extent2D(pos2.getMm()));
    }
private:
    lsst::afw::geom::Point2D _p; // store this in mm
};


}}}
#endif
