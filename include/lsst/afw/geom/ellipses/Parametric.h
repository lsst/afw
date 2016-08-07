// -*- lsst-c++ -*-

/*
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
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
