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

#include "pybind11/pybind11.h"
#include "lsst/cpputils/python.h"

#include <limits>
#include <string>

#include "lsst/afw/image/Color.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst {
namespace afw {
namespace image {

using PyColor = py::class_<Color>;

void wrapColor(lsst::cpputils::python::WrapperCollection & wrappers) {
    PyColor cls(wrappers.module, "Color");

    /* Constructors */
    cls
        // default ctor → indeterminate color
        .def(py::init<>())
        // fully-specified ctor: both ColorValue and ColorType required
        .def(py::init<double, std::string>(),
            "colorValue"_a, "colorType"_a);

    /* Operators */
    cls.def(
        "__eq__", [](Color const & self, Color const & other) { return self == other; },
        py::is_operator());
    cls.def(
        "__ne__", [](Color const & self, Color const & other) { return self != other; },
        py::is_operator());

    /* Members */
    cls.def("isIndeterminate", &Color::isIndeterminate);
    cls.def("getColorValue", &Color::getColorValue);
    cls.def("getColorType", &Color::getColorType);
}

}  // namespace image
}  // namespace afw
}  // namespace lsst
