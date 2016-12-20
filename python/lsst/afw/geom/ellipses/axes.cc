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

#include <pybind11/pybind11.h>
//#include <pybind11/operators.h>
//#include <pybind11/stl.h>

#include "lsst/afw/geom/ellipses/Axes.h"

namespace py = pybind11;

using namespace py::literals;

using namespace lsst::afw::geom::ellipses;

PYBIND11_PLUGIN(_axes) {
    py::module mod("_axes", "Python wrapper for afw _axes library");

    py::class_<Axes, std::shared_ptr<Axes>, BaseCore> clsAxes(mod, "Axes");

    /* Constructors */
    clsAxes.def(py::init<double, double, double, bool>(),
            "a"_a=1.0, "b"_a=1.0, "theta"_a=0.0, "normalize"_a=false);
    clsAxes.def(py::init<Axes const &>());
    clsAxes.def(py::init<BaseCore const &>());

    /* Operators */
    clsAxes.def("__eq__", [](Axes & self, Axes & other) { return self == other; }, py::is_operator());
    clsAxes.def("__neq__", [](Axes & self, Axes & other) { return self != other; }, py::is_operator());

    /* Members */
    clsAxes.def("getA", &Axes::getA);
    clsAxes.def("setA", &Axes::setA);
    clsAxes.def("getB", &Axes::getB);
    clsAxes.def("setB", &Axes::setB);
    clsAxes.def("getTheta", &Axes::getTheta);
    clsAxes.def("setTheta", &Axes::setTheta);
    clsAxes.def("clone", &Axes::clone);
    clsAxes.def("getName", &Axes::getName);
    clsAxes.def("normalize", &Axes::normalize);
    clsAxes.def("readParameters", &Axes::readParameters);
    clsAxes.def("writeParameters", &Axes::writeParameters);
    clsAxes.def("assign", [](Axes & self, Axes & other) { self = other; });
    clsAxes.def("assign", [](Axes & self, BaseCore & other) { self = other; });
    clsAxes.def("transform", [](Axes & self, lsst::afw::geom::LinearTransform const & t) {
        return std::static_pointer_cast<Axes>(self.transform(t).copy());
    });
    clsAxes.def("transformInPlace", [](Axes & self, lsst::afw::geom::LinearTransform const & t) {
       self.transform(t).inPlace();
    });

    return mod.ptr();
}