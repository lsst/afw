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

XYTransform::XYTransform() : daf::base::Citizen(typeid(this)) {}

XYTransform::XYTransform(XYTransform const &) = default;
XYTransform::XYTransform(XYTransform &&) = default;
XYTransform &XYTransform::operator=(XYTransform const &) = default;
XYTransform &XYTransform::operator=(XYTransform &&) = default;

AffineTransform XYTransform::linearizeForwardTransform(Point2D const &p) const {
    Point2D px = p + Extent2D(1, 0);
    Point2D py = p + Extent2D(0, 1);

    return makeAffineTransformFromTriple(p, px, py, this->forwardTransform(p), this->forwardTransform(px),
                                         this->forwardTransform(py));
}

AffineTransform XYTransform::linearizeReverseTransform(Point2D const &p) const {
    Point2D px = p + Extent2D(1, 0);
    Point2D py = p + Extent2D(0, 1);

    return makeAffineTransformFromTriple(p, px, py, this->reverseTransform(p), this->reverseTransform(px),
                                         this->reverseTransform(py));
}

std::shared_ptr<XYTransform> XYTransform::invert() const {
    return std::make_shared<InvertedXYTransform>(this->clone());
}
}  // namespace geom
}  // namespace afw
}  // namespace lsst
