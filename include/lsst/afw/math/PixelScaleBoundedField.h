// -*- LSST-C++ -*-
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

/** \file
 * A BoundedField that gives the amount a pixel is distorted at each point.
 */
#ifndef LSST_AFW_MATH_PixelScaleBoundedField_h_INCLUDED
#define LSST_AFW_MATH_PixelScaleBoundedField_h_INCLUDED

#include "lsst/afw/geom/SkyWcs.h"
#include "lsst/afw/math/BoundedField.h"
#include "lsst/pex/exceptions.h"

namespace lsst {
namespace afw {
namespace math {

/**
 * A BoundedField that gives the amount a pixel is distorted at each point.
 *
 * This is ratio of the SkyWcs-transformed pixel area to the transformed pixel area at the SkyWcs center, or
 * equivalently the determinant of the Jacobian of the SkyWcs Transform.
 *
 * Typically used to move an image or source flux between surface brightness and fluence space.
 */
class PixelScaleBoundedField : public BoundedField {
public:
    /**
     *  Create a PixelScaleBoundedField from a bounding box and SkyWcs.
     */
    PixelScaleBoundedField(lsst::geom::Box2I const &bbox, geom::SkyWcs const &skyWcs)
            : BoundedField(bbox),
              _skyWcs(skyWcs),
              _inverseScale(1.0 / std::pow(skyWcs.getPixelScale().asDegrees(), 2)) {}

    ~PixelScaleBoundedField() override = default;

    PixelScaleBoundedField(PixelScaleBoundedField const &) = default;
    PixelScaleBoundedField(PixelScaleBoundedField &&) = default;
    PixelScaleBoundedField &operator=(PixelScaleBoundedField const &) = delete;
    PixelScaleBoundedField &operator=(PixelScaleBoundedField &&) = delete;

    /// Get the contained SkyWcs
    geom::SkyWcs const &getSkyWcs() const { return _skyWcs; }
    /// Get the cached inverse pixel scale
    double getInverseScale() const { return _inverseScale; }

    /// @copydoc BoundedField::evaluate
    double evaluate(lsst::geom::Point2D const &position) const override;

    /// TransformBoundedField is not persistable.
    bool isPersistable() const noexcept override { return false; }

    /// @copydoc BoundedField::operator*
    std::shared_ptr<BoundedField> operator*(double const scale) const override {
        throw LSST_EXCEPT(pex::exceptions::LogicError, "Not implemented");
    }

    /// @copydoc BoundedField::operator==
    bool operator==(BoundedField const &rhs) const override;

private:
    geom::SkyWcs const _skyWcs;
    // Inverse pixel scale (square degrees), for convenience.
    double const _inverseScale;

    std::string toString() const override;
};

}  // namespace math
}  // namespace afw
}  // namespace lsst

#endif  // !LSST_AFW_MATH_PixelScaleBoundedField_h_INCLUDED
