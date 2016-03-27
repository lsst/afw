// -*- lsst-c++ -*-

/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */

#ifndef LSST_AFW_GEOM_ELLIPSES_Parametric_h_INCLUDED
#define LSST_AFW_GEOM_ELLIPSES_Parametric_h_INCLUDED

#include "lsst/afw/geom/ellipses/Ellipse.h"

namespace lsst { namespace afw { namespace geom { namespace ellipses {

/** 
 *  @brief A functor that returns points on the boundary of the ellipse as a function
 *         of a parameter that runs between 0 and 2 pi (but is not angle).
 */
class Parametric {
public:

    Parametric(Ellipse const & ellipse);

    Point2D operator()(double t) const {
        return _center + _u*std::cos(t) + _v*std::sin(t); 
    }

private:
    Point2D _center;
    Extent2D _u;
    Extent2D _v;
};

}}}} // namespace lsst::afw::geom::ellipses

#endif // !LSST_AFW_GEOM_ELLIPSES_Parametric_h_INCLUDED
