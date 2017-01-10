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
#include <pybind11/stl.h>

#include "lsst/afw/geom/ellipses/BaseCore.h"
#include "lsst/afw/geom/ellipses/Quadrupole.h"

namespace py = pybind11;
using namespace pybind11::literals;

using namespace lsst::afw::geom::ellipses;

PYBIND11_PLUGIN(_quadrupole) {
    py::module mod("_quadrupole", "Python wrapper for afw _quadrupole library");

    /* Module level */
    py::class_<Quadrupole, std::shared_ptr<Quadrupole>, BaseCore> clsQuadrupole(mod, "Quadrupole");

    /* Member types and enums */
    typedef Eigen::Matrix<double,2,2,Eigen::DontAlign> Matrix;

    /* Constructors */
    clsQuadrupole.def(py::init<double, double, double, bool>(),
                      "ixx"_a=1.0, "iyy"_a=1.0, "ixy"_a=0.0, "normalize"_a=false);
    clsQuadrupole.def(py::init<BaseCore::ParameterVector const &, bool>(),
                      "vector"_a, "normalize"_a=false);
    clsQuadrupole.def(py::init<Matrix const &, bool>(),
                      "matrix"_a, "normalize"_a=true);
    clsQuadrupole.def(py::init<Quadrupole const &>());
    clsQuadrupole.def(py::init<BaseCore const &>());
    clsQuadrupole.def(py::init<BaseCore::Convolution const &>());

    py::implicitly_convertible<BaseCore::Convolution, Quadrupole>();

    /* Operators */
    clsQuadrupole.def("__eq__", [](Quadrupole & self, Quadrupole & other) { return self == other; }, py::is_operator());
    clsQuadrupole.def("__neq__", [](Quadrupole & self, Quadrupole & other) { return self != other; }, py::is_operator());

    /* Members */
    clsQuadrupole.def("getIxx", &Quadrupole::getIxx);
    clsQuadrupole.def("getIyy", &Quadrupole::getIyy);
    clsQuadrupole.def("getIxy", &Quadrupole::getIxy);
    clsQuadrupole.def("setIxx", &Quadrupole::setIxx);
    clsQuadrupole.def("setIyy", &Quadrupole::setIyy);
    clsQuadrupole.def("setIxy", &Quadrupole::setIxy);
    clsQuadrupole.def("assign", [](Quadrupole & self, Quadrupole & other) { self = other; });
    clsQuadrupole.def("assign", [](Quadrupole & self, BaseCore & other) { self = other; });
    clsQuadrupole.def("transform", [](Quadrupole & self, lsst::afw::geom::LinearTransform const & t) {
        return std::static_pointer_cast<Quadrupole>(self.transform(t).copy());
    });
    clsQuadrupole.def("transformInPlace", [](Quadrupole & self, lsst::afw::geom::LinearTransform const & t) {
       self.transform(t).inPlace();
    });
    // TODO: pybind11 based on swig wrapper for now. Hopefully can be removed once pybind11 gets smarter
    // handling of implicit conversions
    clsQuadrupole.def("convolve", [](Quadrupole & self, BaseCore const & other) {
        return Quadrupole(self.convolve(other));
    });

    return mod.ptr();
}