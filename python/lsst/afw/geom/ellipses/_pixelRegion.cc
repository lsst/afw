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
#include <lsst/utils/python.h>

#include "lsst/afw/geom/ellipses/PixelRegion.h"

namespace py = pybind11;

namespace lsst {
namespace afw {
namespace geom {
namespace ellipses {
void wrapPixelRegion(lsst::utils::python::WrapperCollection &wrappers) {
    wrappers.wrapType(py::class_<PixelRegion>(wrappers.module, "PixelRegion"), [](auto &mod, auto &cls) {
        /* Constructors */
        cls.def(py::init<Ellipse const &>());

        /* Members */
        cls.def("getBBox", &PixelRegion::getBBox, py::return_value_policy::copy);
        cls.def("getSpanAt", &PixelRegion::getSpanAt);
        cls.def(
                "__iter__",
                [](const PixelRegion &self) { return py::make_iterator(self.begin(), self.end()); },
                py::keep_alive<0, 1>() /* Essential: keep object alive while iterator exists */);
    });
}
}  // namespace ellipses
}  // namespace geom
}  // namespace afw
}  // namespace lsst