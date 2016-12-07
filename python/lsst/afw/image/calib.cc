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
#include <memory>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "numpy/arrayobject.h"
#include "ndarray/pybind11.h"
#include "ndarray/converter.h"
#include "ndarray_fwd.h"

#include "lsst/afw/table/io/Persistable.h"
#include "lsst/afw/table/io/pybind11.h"
#include "lsst/afw/image/Calib.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst {
namespace afw {
namespace image {

PYBIND11_PLUGIN(_calib) {
    py::module mod("_calib", "Python wrapper for afw _calib library");

    if (_import_array() < 0) {
            PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
            return nullptr;
    };

    // TODO: Commented-out code is waiting until needed and is untested.
    // Add tests for it and enable it or remove it before the final pybind11 merge.

    /* Module level */
    mod.def("abMagFromFlux", abMagFromFlux);
    mod.def("abMagErrFromFluxErr", abMagErrFromFluxErr);
    mod.def("fluxFromABMag", fluxFromABMag);
    mod.def("fluxErrFromABMagErr", fluxErrFromABMagErr);

    table::io::declarePersistableFacade<Calib>(mod, "Calib");

    py::class_<Calib, std::shared_ptr<Calib>,
               table::io::PersistableFacade<Calib>,
               table::io::Persistable> cls(mod, "Calib");

    /* Constructors */
    cls.def(py::init<>());
    cls.def(py::init<double>(), "fluxMag0"_a);
    // cls.def(py::init<std::vector<std::shared_ptr<const Calib>> const &>(), "calibs"_a);
    // cls.def(py::init<std::shared_ptr<const lsst::daf::base::PropertySet>>(), "metadata"_a);

    /* Operators */
    cls.def("__eq__",
            [](Calib const & self, Calib const & other) { return self == other; },
            py::is_operator());
    cls.def("__ne__",
            [](Calib const & self, Calib const & other) { return self != other; },
            py::is_operator());

    // /* Members */
    cls.def("setFluxMag0", (void (Calib::*)(double, double)) &Calib::setFluxMag0,
            "fluxMag0"_a, "fluxMag0Sigma"_a=0.0);
    cls.def("setFluxMag0", (void (Calib::*)(std::pair<double, double>)) &Calib::setFluxMag0,
            "fluxMag0AndSigma"_a);
    cls.def("getFluxMag0", &Calib::getFluxMag0);
    cls.def("getFlux", (double (Calib::*)(double const) const) &Calib::getFlux, "mag"_a);
    cls.def("getFlux",
            (std::pair<double, double> (Calib::*)(double const, double const) const) &Calib::getFlux,
            "mag"_a, "magErr"_a);
    cls.def("getFlux",
            (ndarray::Array<double, 1> (Calib::*)(ndarray::Array<double const, 1> const &) const)
            &Calib::getFlux, "mag"_a);
    cls.def("getFlux",
            (std::pair<ndarray::Array<double, 1>, ndarray::Array<double, 1>>
             (Calib::*)(ndarray::Array<double const, 1> const &,
                        ndarray::Array<double const, 1> const &) const) &Calib::getFlux,
            "mag"_a, "magErr"_a);
    cls.def("getMagnitude", (double (Calib::*)(double const) const) &Calib::getMagnitude, "flux"_a);
    cls.def("getMagnitude",
            (std::pair<double, double> (Calib::*)(double const, double const) const) &Calib::getMagnitude,
            "flux"_a, "fluxErr"_a);
    cls.def("getMagnitude",
            (ndarray::Array<double, 1> (Calib::*)(ndarray::Array<double const, 1> const &) const)
            &Calib::getMagnitude, "flux"_a);
    cls.def("getMagnitude",
            (std::pair<ndarray::Array<double, 1>, ndarray::Array<double, 1>>
             (Calib::*)(ndarray::Array<double const, 1> const &,
                        ndarray::Array<double const, 1> const &) const) &Calib::getMagnitude,
            "flux"_a, "fluxMag"_a);
    cls.def_static("setThrowOnNegativeFlux", Calib::setThrowOnNegativeFlux, "raiseException"_a);
    cls.def_static("getThrowOnNegativeFlux", Calib::getThrowOnNegativeFlux);
    // cls.def("isPersistable", &Calib::isPersistable);

    return mod.ptr();
}

}}}  // namespace lsst::afw::image
