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
#ifndef LSST_AFW_MATH_PixelAreaBoundedField_h_INCLUDED
#define LSST_AFW_MATH_PixelAreaBoundedField_h_INCLUDED


#include "lsst/geom/Angle.h"
#include "lsst/afw/geom/SkyWcs.h"
#include "lsst/afw/math/BoundedField.h"
#include "lsst/pex/exceptions.h"

namespace lsst {
namespace afw {
namespace math {

/**
 * A BoundedField that evaluate the pixel area of a SkyWcs in angular units.
 *
 * Typically used to move an image or source flux between surface brightness
 * and fluence.
 */
class PixelAreaBoundedField : public BoundedField {
public:

    /**
     *  Create a PixelAreaBoundedField from a SkyWcs.
     *
     *  @param[in] bbox      Pixel bounding box over which the WCS is valid.
     *  @param[in] skyWcs    WCS that maps pixels to and from sky coordinates.
     *  @param[in] unit      Angular unit that is used (squared) for the fields
     *                       values.
     *  @param[in] scaling   Factor all field values should be scaled by.
     *
     *  @throw pex::exception::InvalidParameterError  Thrown if `skyWcs` is
     *      null.
     */
    PixelAreaBoundedField(
        lsst::geom::Box2I const &bbox,
        std::shared_ptr<geom::SkyWcs const> skyWcs,
        lsst::geom::AngleUnit const & unit = lsst::geom::radians,
        double scaling = 1.0
    );

    ~PixelAreaBoundedField() override = default;

    PixelAreaBoundedField(PixelAreaBoundedField const &) = default;
    PixelAreaBoundedField(PixelAreaBoundedField &&) = default;
    PixelAreaBoundedField &operator=(PixelAreaBoundedField const &) = delete;
    PixelAreaBoundedField &operator=(PixelAreaBoundedField &&) = delete;

    /// @copydoc BoundedField::evaluate(lsst::geom::Point2D const &) const
    double evaluate(lsst::geom::Point2D const &position) const override;

    /// @copydoc BoundedField::evaluate(ndarray::Array<double const, 1> const &x, ndarray::Array<double const, 1> const &y) const
    ndarray::Array<double, 1, 1> evaluate(ndarray::Array<double const, 1> const & x,
                                          ndarray::Array<double const, 1> const & y) const override;

    /// PixelAreaBoundedField is persistable if and only if the nested SkyWcs
    /// is.
    bool isPersistable() const noexcept override;

    /// @copydoc BoundedField::operator*
    std::shared_ptr<BoundedField> operator*(double const scale) const override;

    /// @copydoc BoundedField::operator==
    bool operator==(BoundedField const &rhs) const override;

protected:

    std::string getPersistenceName() const override;
    std::string getPythonModule() const override;
    void write(OutputArchiveHandle &handle) const override;

private:

    std::string toString() const override;

    std::shared_ptr<geom::SkyWcs const> _skyWcs;
    double _scaling;
};

}  // namespace math
}  // namespace afw
}  // namespace lsst

#endif  // !LSST_AFW_MATH_PixelAreaBoundedField_h_INCLUDED
