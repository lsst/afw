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
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include "lsst/afw/image/Calib.h"

namespace py = pybind11;

using namespace lsst::afw::image;

PYBIND11_PLUGIN(_calib) {
    py::module mod("_calib", "Python wrapper for afw _calib library");

    /* Module level */
    mod.def("abMagFromFlux", abMagFromFlux);
    mod.def("abMagErrFromFluxErr", abMagErrFromFluxErr);
    mod.def("fluxFromABMag", fluxFromABMag);
    mod.def("fluxErrFromABMagErr", fluxErrFromABMagErr);

//    py::class_<Calib> cls(mod, "Calib");
//
//    /* Constructors */
//    cls.def(py::init<>());
//    cls.def(py::init<double>();
//    cls.def(py::init<std::vector<CONST_PTR(Calib) const &>());
//    cls.def(py::init<CONST_PTR(lsst::daf::base::PropertySet)>());
//
//    /* Operators */
//    cls.def(py::self == py::self);
//    cls.def(py::self != py::self);
//    cls.def(py::self *= py::double_);
//    cls.def(py::self /= py::double_);
//
//    /* Members */
//    cls.def("setMidTime", &Calib::setMidTime);
//    cls.def("getMidTime", (lsst::daf::base::DateTime (Calib::*)() const) &Calib::getMidTime);
//    cls.def("setExptime", &Calib::setExptime);
//    cls.def("getExptime", &Calib::getExptime);
//    cls.def("setFluxMag0", (void (Calib::*)(double, double)) &Calib::setFluxMag0,
//        py::arg("fluxMag0"), py::arg("fluxMag0Sigma")=0.0);
//    cls.def("setFluxMag0", (void (Calib::*)(std::pair<double, double>)) &Calib::setFluxMag0);
//    cls.def("getFluxMag0", &Calib::getFluxMag0);
//    cls.def("getFlux", (double (Calib::*)(double const) const) &Calib::getFlux);
//    cls.def("getFlux", (std::pair<double, double> (Calib::*)(double const, double const) const) &Calib::getFlux);
//    cls.def("getFlux", (ndarray::Array<double, 1> (Calib::*)(ndarray::Array<double const, 1> const &) const) &Calib::getFlux);
//    cls.def("getFlux", (std::pair<ndarray::Array<double, 1>, ndarray::Array<double, 1>> (Calib::*)(ndarray::Array<double const, 1> const &, ndarray::Array<double const, 1> const &) const) &Calib::getFlux);
//    cls.def("getMagnitude", (double (Calib::*)(double const) const) &Calib::getMagnitude);
//    cls.def("getMagnitude", (std::pair<double, double> (Calib::*)(double const, double const) const) &Calib::getMagnitude);
//    cls.def("getMagnitude", (ndarray::Array<double, 1> (Calib::*)(ndarray::Array<double const, 1> const &) const) &Calib::getMagnitude);
//    cls.def("getMagnitude", (std::pair<ndarray::Array<double, 1>, ndarray::Array<double, 1>> (Calib::*)(ndarray::Array<double const, 1> const &, ndarray::Array<double const, 1> const &) const) &Calib::getMagnitude);
//    cls.def_static("setThrowOnNegativeFlux", Calib::setThrowOnNegativeFlux);
//    cls.def_static("getThrowOnNegativeFlux", Calib::getThrowOnNegativeFlux);
//    cls.def("isPersistable", &Calib::isPersistable);

    return mod.ptr();
}