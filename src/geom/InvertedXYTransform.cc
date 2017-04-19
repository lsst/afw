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

#include "lsst/afw/geom/XYTransform.h"
#include <memory>

namespace lsst {
namespace afw {
namespace geom {

InvertedXYTransform::InvertedXYTransform(std::shared_ptr<XYTransform const> base)
        : XYTransform(), _base(base) {}

std::shared_ptr<XYTransform> InvertedXYTransform::clone() const {
    // deep copy
    return std::make_shared<InvertedXYTransform>(_base->clone());
}

std::shared_ptr<XYTransform> InvertedXYTransform::invert() const { return _base->clone(); }

Point2D InvertedXYTransform::forwardTransform(Point2D const &point) const {
    return _base->reverseTransform(point);
}

Point2D InvertedXYTransform::reverseTransform(Point2D const &point) const {
    return _base->forwardTransform(point);
}

AffineTransform InvertedXYTransform::linearizeForwardTransform(Point2D const &point) const {
    return _base->linearizeReverseTransform(point);
}

AffineTransform InvertedXYTransform::linearizeReverseTransform(Point2D const &point) const {
    return _base->linearizeForwardTransform(point);
}
}
}
}
