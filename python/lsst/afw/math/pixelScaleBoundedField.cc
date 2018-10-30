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
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "lsst/afw/geom/SkyWcs.h"
#include "lsst/afw/math/BoundedField.h"
#include "lsst/afw/math/PixelScaleBoundedField.h"

namespace py = pybind11;
using namespace py::literals;

namespace lsst {
namespace afw {
namespace math {
namespace {

using ClsField = py::class_<PixelScaleBoundedField, std::shared_ptr<PixelScaleBoundedField>, BoundedField>;

PYBIND11_MODULE(pixelScaleBoundedField, mod) {
    py::module::import("lsst.afw.geom.skyWcs");

    /* Module level */
    ClsField cls(mod, "PixelScaleBoundedField");

    cls.def(py::init<lsst::geom::Box2I const &, afw::geom::SkyWcs const &>(), "bbox"_a, "skyWcs"_a);

    cls.def("getSkyWcs", &PixelScaleBoundedField::getSkyWcs);
    cls.def("getInverseScale", &PixelScaleBoundedField::getInverseScale);

    cls.def("__repr__", [](PixelScaleBoundedField const &self) {
        std::ostringstream os;
        os << "lsst.afw.math.PixelScaleBoundedField(" << self << ")";
        return os.str();
    });
}

}  // namespace
}  // namespace math
}  // namespace afw
}  // namespace lsst
