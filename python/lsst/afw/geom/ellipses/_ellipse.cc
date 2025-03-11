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

#include "nanobind/nanobind.h"
#include "nanobind/eigen/dense.h"
#include "nanobind/stl/vector.h"
#include <lsst/cpputils/python.h>

#include "ndarray/nanobind.h"

#include "lsst/afw/geom/ellipses/BaseCore.h"
#include "lsst/afw/geom/ellipses/Convolution.h"
#include "lsst/afw/geom/ellipses/GridTransform.h"
#include "lsst/afw/geom/ellipses/Ellipse.h"
#include "lsst/afw/geom/ellipses/Transformer.h"

namespace nb = nanobind;
using namespace nanobind::literals;

namespace lsst {
namespace afw {
namespace geom {
namespace ellipses {
void wrapEllipse(lsst::cpputils::python::WrapperCollection &wrappers) {
    wrappers.wrapType(
            nb::class_<Ellipse>(wrappers.module, "Ellipse"),
            [](auto &mod, auto &cls) {
                /* Constructors */
                cls.def(nb::init<BaseCore const &, lsst::geom::Point2D const &>(), "core"_a,
                        "center"_a = lsst::geom::Point2D());
                cls.def(nb::init<Ellipse const &>());
                cls.def(nb::init<Ellipse::Convolution const &>());

                nb::implicitly_convertible<Ellipse::Convolution, Ellipse>();

                /* Operators */

                /* Members */
                cls.def("getCore", [](Ellipse &ellipse) { return ellipse.getCorePtr(); });
                cls.def("getCenter", (lsst::geom::Point2D & (Ellipse::*)()) & Ellipse::getCenter);
                cls.def("setCenter", &Ellipse::setCenter);
                cls.def("setCore", &Ellipse::setCore);
                cls.def("normalize", &Ellipse::normalize);
                cls.def("grow", &Ellipse::grow);
                cls.def("scale", &Ellipse::scale);
                cls.def("shift", &Ellipse::shift);
                cls.def("getParameterVector", &Ellipse::getParameterVector);
                cls.def("setParameterVector", &Ellipse::setParameterVector);
                cls.def(
                        "transform",
                        [](Ellipse const &self, lsst::geom::AffineTransform const &t) -> Ellipse {
                            return self.transform(t);
                        },
                        "transform"_a);
                cls.def("getGridTransform", [](Ellipse &self) -> lsst::geom::AffineTransform {
                    return self.getGridTransform();  // delibarate conversion to lsst::geom::AffineTransform
                });
                cls.def("computeBBox", &Ellipse::computeBBox);
            });
}
}  // namespace ellipses
}  // namespace geom
}  // namespace afw
}  // namespace lsst
