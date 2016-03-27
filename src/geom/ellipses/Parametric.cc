// -*- lsst-c++ -*-

/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */
#include "lsst/afw/geom/ellipses/Parametric.h"

namespace lsst { namespace afw { namespace geom { namespace ellipses {

Parametric::Parametric(Ellipse const & ellipse) : _center(ellipse.getCenter()) {
    double a, b, theta;
    ellipse.getCore()._assignToAxes(a, b, theta);
    double c = std::cos(theta);
    double s = std::sin(theta);
    _u = Extent2D(a*c, a*s);
    _v = Extent2D(-b*s, b*c);
}

}}}} // namespace lsst::afw::geom::ellipses
