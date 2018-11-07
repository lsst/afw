/*
 * This file is part of afw.
 *
 * Developed for the LSST Data Management System.
 * This product includes software developed by the LSST Project
 * (https://www.lsst.org).
 * See the COPYRIGHT file at the top-level directory of this distribution
 * for details of code ownership.
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
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "lsst/afw/math/PixelScaleBoundedField.h"

namespace lsst {
namespace afw {
namespace math {

double PixelScaleBoundedField::evaluate(lsst::geom::Point2D const &position) const {
    return std::pow(_skyWcs.getPixelScale(position).asDegrees(), 2) * _inverseScale;
}

bool PixelScaleBoundedField::operator==(BoundedField const &rhs) const {
    auto rhsCasted = dynamic_cast<PixelScaleBoundedField const *>(&rhs);
    if (!rhsCasted) return false;

    return getBBox() == rhsCasted->getBBox() && getSkyWcs() == rhsCasted->getSkyWcs();
}

std::string PixelScaleBoundedField::toString() const {
    std::ostringstream os;
    os << "PixelScaleBoundedField(" << _skyWcs << ")";
    return os.str();
}

}  // namespace math
}  // namespace afw
}  // namespace lsst
