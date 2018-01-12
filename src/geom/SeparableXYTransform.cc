// -*- lsst-c++ -*-

/*
 * LSST Data Management System
 * Copyright 2015 LSST Corporation.
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

#include <cmath>

#include "lsst/afw/geom/Functor.h"
#include "lsst/afw/geom/SeparableXYTransform.h"

namespace lsst {
namespace afw {
namespace geom {

SeparableXYTransform::SeparableXYTransform(Functor const& xfunctor, Functor const& yfunctor)
        : XYTransform(), _xfunctor(xfunctor.clone()), _yfunctor(yfunctor.clone()) {}

SeparableXYTransform::SeparableXYTransform(SeparableXYTransform const&) = default;
SeparableXYTransform::SeparableXYTransform(SeparableXYTransform&&) = default;
SeparableXYTransform& SeparableXYTransform::operator=(SeparableXYTransform const&) = default;
SeparableXYTransform& SeparableXYTransform::operator=(SeparableXYTransform&&) = default;

std::shared_ptr<XYTransform> SeparableXYTransform::clone() const {
    return std::make_shared<SeparableXYTransform>(*_xfunctor, *_yfunctor);
}

Point2D SeparableXYTransform::forwardTransform(Point2D const& point) const {
    double xin = point.getX();
    double yin = point.getY();
    double xout = (*_xfunctor)(xin);
    double yout = (*_yfunctor)(yin);
    return Point2D(xout, yout);
}

Point2D SeparableXYTransform::reverseTransform(Point2D const& point) const {
    double xout = point.getX();
    double yout = point.getY();
    double xin = 0;
    double yin = 0;
    try {
        xin = _xfunctor->inverse(xout);
        yin = _yfunctor->inverse(yout);
    } catch (lsst::pex::exceptions::Exception& e) {
        LSST_EXCEPT_ADD(e, "Called from SeparableXYTransform::reverseTransform");
        throw;
    }
    return Point2D(xin, yin);
}

Functor const& SeparableXYTransform::getXfunctor() const { return *_xfunctor; }

Functor const& SeparableXYTransform::getYfunctor() const { return *_yfunctor; }

}  // namespace geom
}  // namespace afw
}  // namespace lsst
