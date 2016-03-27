// -*- lsst-c++ -*-
/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */

#include "lsst/utils/ieee.h"
#include "lsst/afw/geom/ellipses/PixelRegion.h"
#include "lsst/afw/geom/ellipses/Quadrupole.h"

namespace lsst { namespace afw { namespace geom { namespace ellipses {

PixelRegion::PixelRegion(Ellipse const & ellipse) :
    _center(ellipse.getCenter()),
    _bbox(ellipse.computeBBox(), Box2I::EXPAND)
{
    Quadrupole::Matrix q = Quadrupole(ellipse.getCore()).getMatrix();
    _detQ = q(0,0) * q(1,1) - q(0, 1) * q(0, 1);
    _invQxx = q(1,1) / _detQ;
    _alpha = q(0,1) / _detQ / _invQxx; // == -invQxy / invQxx
    if (lsst::utils::isnan(_alpha)) {
        _alpha = 0.0;
    }
}

Span const PixelRegion::getSpanAt(int y) const {
    double yt = y - _center.getY();
    double d = _invQxx - yt*yt/_detQ;
    double x0 = _center.getX() + yt * _alpha;
    double x1 = x0;
    if (d > 0.0) {
        d = std::sqrt(d) / _invQxx;
        x0 -= d;
        x1 += d;
    } // Note that we return an empty span when d <= 0.0 or d is NaN.
    return Span(y, std::ceil(x0), std::floor(x1));
}

}}}} // namespace lsst::afw::geom::ellipses
