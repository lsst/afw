// -*- lsst-c++ -*-

/* 
 * LSST Data Management System
 * Copyright 2014 LSST Corporation.
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

#include "lsst/afw/geom/XYTransform.h"
#include "boost/make_shared.hpp"

namespace lsst {
namespace afw {
namespace geom {

AffineXYTransform::AffineXYTransform(AffineTransform const &affineTransform)
    : XYTransform(), _forwardAffineTransform(affineTransform), 
      _reverseAffineTransform(_forwardAffineTransform.invert())
{ }

PTR(XYTransform) AffineXYTransform::clone() const
{
    return std::make_shared<AffineXYTransform> (_forwardAffineTransform);
}

Point2D AffineXYTransform::forwardTransform(Point2D const &position) const
{
    return _forwardAffineTransform(position);
}

Point2D AffineXYTransform::reverseTransform(Point2D const &position) const
{
    return _reverseAffineTransform(position);
}

AffineTransform AffineXYTransform::linearizeForwardTransform(Point2D const &) const
{
    return _forwardAffineTransform;
}

AffineTransform AffineXYTransform::linearizeReverseTransform(Point2D const &) const
{
    return _reverseAffineTransform; 
}

AffineTransform AffineXYTransform::getForwardTransform() const
{
    return _forwardAffineTransform;
}

AffineTransform AffineXYTransform::getReverseTransform() const
{
    return _reverseAffineTransform;
}

}}}
