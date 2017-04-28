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

#include "lsst/afw/math/minimize.h"

namespace py = pybind11;

using namespace lsst::afw::math;

template <typename ReturnT>
void declareMinimize(py::module &mod) {
    mod.def("minimize", (FitResults(*)(lsst::afw::math::Function1<ReturnT> const &,
                                       std::vector<double> const &, std::vector<double> const &,
                                       std::vector<double> const &, std::vector<double> const &,
                                       std::vector<double> const &, double))minimize<ReturnT>);
    mod.def("minimize",
            (FitResults(*)(lsst::afw::math::Function2<ReturnT> const &, std::vector<double> const &,
                           std::vector<double> const &, std::vector<double> const &,
                           std::vector<double> const &, std::vector<double> const &,
                           std::vector<double> const &, double))minimize<ReturnT>);
};

PYBIND11_PLUGIN(_minimize) {
    py::module mod("_minimize", "Python wrapper for afw _minimize library");

    /* Module level */

    /* Member types and enums */

    /* Constructors */

    /* Operators */

    /* Members */
    py::class_<FitResults> clsFitResults(mod, "FitResults");
    clsFitResults.def_readwrite("isValid", &FitResults::isValid);
    clsFitResults.def_readwrite("chiSq", &FitResults::chiSq);
    clsFitResults.def_readwrite("parameterList", &FitResults::parameterList);
    clsFitResults.def_readwrite("parameterErrorList", &FitResults::parameterErrorList);

    declareMinimize<double>(mod);
    declareMinimize<float>(mod);

    return mod.ptr();
}