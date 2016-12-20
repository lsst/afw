/* 
 * LSST Data Management System
 * Copyright 2008-2016  AURA/LSST.
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
 * see <https://www.lsstcorp.org/LegalNotices/>.
 */

#include <limits>

#include <pybind11/pybind11.h>
//#include <pybind11/stl.h>

#include "lsst/afw/image/Color.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst {
namespace afw {
namespace image {

PYBIND11_PLUGIN(_color) {
    py::module mod("_color", "Python wrapper for afw _color library");

    /* Module level */
    py::class_<Color> cls(mod, "Color");

    /* Member types and enums */

    /* Constructors */
    cls.def(py::init<double>(), "g_r"_a=std::numeric_limits<double>::quiet_NaN());

    /* Operators */
    cls.def("__eq__",
            [](Color const & self, Color const & other) { return self == other; },
            py::is_operator());
    cls.def("__ne__",
            [](Color const & self, Color const & other) { return self != other; },
            py::is_operator());

    /* Members */
    cls.def("isIndeterminate", &Color::isIndeterminate);
    cls.def("getLambdaEff", &Color::getLambdaEff, "filter"_a);

    return mod.ptr();
}

}}}  // namespace lsst::afw::image
