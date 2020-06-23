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

#ifndef LSST_AFW_DETECTION_PYTHON_H
#define LSST_AFW_DETECTION_PYTHON_H

#include "pybind11/pybind11.h"
#include "lsst/afw/detection/Psf.h"
#include "lsst/afw/typehandling/python.h"

using lsst::afw::typehandling::StorableHelper;

namespace lsst {
namespace afw {
namespace detection {

/**
 * "Trampoline" for Psf to let it be used as a base class in Python.
 *
 * Subclasses of Psf that are wrapped in %pybind11 should have a similar
 * helper that subclasses `PsfTrampoline<subclass>`. This helper can be
 * skipped if the subclass neither adds any virtual methods nor implements
 * any abstract methods.
 *
 * @tparam Base the exact (most specific) class being wrapped
 *
 * @see [pybind11 documentation](https://pybind11.readthedocs.io/en/stable/advanced/classes.html)
 */
template <typename Base = Psf>
class PsfTrampoline : public StorableHelper<Base> {
public:
    using Image = typename Base::Image;

    /**
     * Delegating constructor for wrapped class.
     *
     * While we would like to simply inherit base class constructors, when doing so, we cannot
     * change their access specifiers.  One consequence is that it's not possible to use inheritance
     * to expose a protected constructor to python.  The alternative, used here, is to create a new
     * public constructor that delegates to the base class public or protected constructor with the
     * same signature.
     *
     * @tparam Args  Variadic type specification
     * @param ...args  Arguments to forward to the Base class constructor.
     */
    template<typename... Args>
    PsfTrampoline<Base>(Args... args) : StorableHelper<Base>(args...) {}

    std::shared_ptr<Psf> clone() const override {
        /* __deepcopy__ takes an optional dict, but PYBIND11_OVERLOAD_* won't
         * compile unless you give it arguments that work for the C++ method
         */
        PYBIND11_OVERLOAD_PURE_NAME(std::shared_ptr<Psf>, Base, "__deepcopy__", clone,);
    }

    std::shared_ptr<Psf> resized(int width, int height) const override {
        PYBIND11_OVERLOAD_PURE(std::shared_ptr<Psf>, Base, resized, width, height);
    }

    lsst::geom::Point2D getAveragePosition() const override {
        PYBIND11_OVERLOAD(lsst::geom::Point2D, Base, getAveragePosition,);
    }

    // Private and protected c++ members are overloaded to python using underscores.
    std::shared_ptr<Image> doComputeImage(
        lsst::geom::Point2D const& position,
        image::Color const& color
    ) const override {
        PYBIND11_OVERLOAD_NAME(
            std::shared_ptr<Image>, Base, "_doComputeImage", doComputeImage, position, color
        );
    }

    std::shared_ptr<Image> doComputeKernelImage(
        lsst::geom::Point2D const& position,
        image::Color const& color
    ) const override {
        PYBIND11_OVERLOAD_PURE_NAME(
            std::shared_ptr<Image>, Base, "_doComputeKernelImage", doComputeKernelImage, position, color
        );
    }

    double doComputeApertureFlux(
        double radius, lsst::geom::Point2D const& position,
        image::Color const& color
    ) const override {
        PYBIND11_OVERLOAD_PURE_NAME(
            double, Base, "_doComputeApertureFlux", doComputeApertureFlux, radius, position, color
        );
    }

    geom::ellipses::Quadrupole doComputeShape(
        lsst::geom::Point2D const& position,
        image::Color const& color
    ) const override {
        PYBIND11_OVERLOAD_PURE_NAME(
            geom::ellipses::Quadrupole, Base, "_doComputeShape", doComputeShape, position, color
        );
    }

    lsst::geom::Box2I doComputeBBox(
        lsst::geom::Point2D const& position,
        image::Color const& color
    ) const override {
        PYBIND11_OVERLOAD_PURE_NAME(
            lsst::geom::Box2I, Base, "_doComputeBBox", doComputeBBox, position, color
        );
    }
};

}  // namespace detection
}  // namespace afw
}  // namespace lsst

#endif
