/*
 * LSST Data Management System
 * Copyright 2008-2017 AURA/LSST.
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

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include <memory>
#include <vector>

#include "numpy/arrayobject.h"
#include "ndarray/pybind11.h"

#include "lsst/daf/base/PropertySet.h"
#include "lsst/afw/table/io/python.h"  // for addPersistableMethods
#include "lsst/afw/image/Calib.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst {
namespace afw {
namespace image {
namespace {

using PyCalib = py::class_<Calib, std::shared_ptr<Calib>>;

template <typename T>
void declareVectorOperations(py::module & mod)
{
    typedef ndarray::Array<T, 1> Array;
    typedef ndarray::Array<T const, 1> ConstArray;
    mod.def("abMagFromFlux", (Array(*)(ConstArray const&))&abMagFromFlux<T>, "flux"_a);
    mod.def("abMagErrFromFluxErr", (Array(*)(ConstArray const&, ConstArray const&))&abMagErrFromFluxErr<T>,
            "fluxErr"_a, "flux"_a);
    mod.def("fluxFromABMag", (Array(*)(ConstArray const&))&fluxFromABMag<T>, "mag"_a);
    mod.def("fluxErrFromABMagErr", (Array(*)(ConstArray const&, ConstArray const&))&fluxErrFromABMagErr<T>,
            "magErr"_a, "mag"_a);
}


PYBIND11_PLUGIN(calib) {
    py::module mod("calib");

    if (_import_array() < 0) {
        PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
        return nullptr;
    };

    /* Module level */
    mod.def("abMagFromFlux", (double(*)(double))&abMagFromFlux, "flux"_a);
    mod.def("abMagErrFromFluxErr", (double(*)(double, double))&abMagErrFromFluxErr, "fluxErr"_a, "flux"_a);
    mod.def("fluxFromABMag", (double(*)(double))&fluxFromABMag, "mag"_a);
    mod.def("fluxErrFromABMagErr", (double(*)(double, double))&fluxErrFromABMagErr, "magErr"_a, "mag"_a);
    declareVectorOperations<float>(mod);
    declareVectorOperations<double>(mod);
    mod.def("stripCalibKeywords", &detail::stripCalibKeywords, "metadata"_a);

    PyCalib cls(mod, "Calib");

    /* Constructors */
    cls.def(py::init<>());
    cls.def(py::init<double>(), "fluxMag0"_a);
    cls.def(py::init<std::vector<std::shared_ptr<const Calib>> const &>(), "calibs"_a);
    cls.def(py::init<std::shared_ptr<const daf::base::PropertySet>>(), "metadata"_a);

    table::io::python::addPersistableMethods<Calib>(cls);

    /* Operators */
    cls.def("__eq__", &Calib::operator==, py::is_operator());
    cls.def("__ne__", &Calib::operator!=, py::is_operator());
    cls.def("__imul__", &Calib::operator*=);
    cls.def("__itruediv__", &Calib::operator/=);
    cls.def("__idiv__", &Calib::operator/=);

    /* Members */
    cls.def("setFluxMag0", (void (Calib::*)(double, double)) & Calib::setFluxMag0, "fluxMag0"_a,
            "fluxMag0Sigma"_a = 0.0);
    cls.def("setFluxMag0", (void (Calib::*)(std::pair<double, double>)) & Calib::setFluxMag0,
            "fluxMag0AndSigma"_a);
    cls.def("getFluxMag0", &Calib::getFluxMag0);
    cls.def("getFlux", (double (Calib::*)(double const) const) & Calib::getFlux, "mag"_a);
    cls.def("getFlux",
            (std::pair<double, double> (Calib::*)(double const, double const) const) & Calib::getFlux,
            "mag"_a, "magErr"_a);
    cls.def("getFlux", (ndarray::Array<double, 1> (Calib::*)(ndarray::Array<double const, 1> const &) const) &
                               Calib::getFlux,
            "mag"_a);
    cls.def("getFlux",
            (std::pair<ndarray::Array<double, 1>, ndarray::Array<double, 1>> (Calib::*)(
                    ndarray::Array<double const, 1> const &, ndarray::Array<double const, 1> const &) const) &
                    Calib::getFlux,
            "mag"_a, "magErr"_a);
    cls.def("getMagnitude", (double (Calib::*)(double const) const) & Calib::getMagnitude, "flux"_a);
    cls.def("getMagnitude",
            (std::pair<double, double> (Calib::*)(double const, double const) const) & Calib::getMagnitude,
            "flux"_a, "fluxErr"_a);
    cls.def("getMagnitude",
            (ndarray::Array<double, 1> (Calib::*)(ndarray::Array<double const, 1> const &) const) &
                    Calib::getMagnitude,
            "flux"_a);
    cls.def("getMagnitude",
            (std::pair<ndarray::Array<double, 1>, ndarray::Array<double, 1>> (Calib::*)(
                    ndarray::Array<double const, 1> const &, ndarray::Array<double const, 1> const &) const) &
                    Calib::getMagnitude,
            "flux"_a, "fluxMag"_a);
    cls.def_static("setThrowOnNegativeFlux", Calib::setThrowOnNegativeFlux, "raiseException"_a);
    cls.def_static("getThrowOnNegativeFlux", Calib::getThrowOnNegativeFlux);
    cls.def("isPersistable", &Calib::isPersistable);

    return mod.ptr();
}
}
}
}
}  // namespace lsst::afw::image::<anonymous>
