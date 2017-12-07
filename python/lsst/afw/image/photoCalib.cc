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

#include "numpy/arrayobject.h"
#include "ndarray/pybind11.h"

#include "lsst/daf/base/PropertySet.h"
#include "lsst/afw/math/BoundedField.h"
#include "lsst/afw/table/io/Persistable.h"
#include "lsst/afw/table/io/python.h"  // for addPersistableMethods
#include "lsst/afw/image/PhotoCalib.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst {
namespace afw {
namespace image {
namespace {

void declareMeasurement(py::module &mod) {
    py::class_<Measurement, std::shared_ptr<Measurement>> cls(mod, "Measurement");

    cls.def(py::init<double, double>(), "value"_a, "err"_a);
    cls.def_readonly("value", &Measurement::value);
    cls.def_readonly("err", &Measurement::err);
}

PYBIND11_PLUGIN(photoCalib) {
    py::module mod("photoCalib");
    py::module::import("lsst.afw.table.io");

    if (_import_array() < 0) {
        PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
        return nullptr;
    };

    declareMeasurement(mod);

    py::class_<PhotoCalib, std::shared_ptr<PhotoCalib>> cls(mod, "PhotoCalib");

    /* Constructors */
    cls.def(py::init<>());
    cls.def(py::init<double, double, afw::geom::Box2I>(), "instFluxMag0"_a, "instFluxMag0Err"_a = 0.0,
            "bbox"_a = afw::geom::Box2I());
    cls.def(py::init<std::shared_ptr<afw::math::BoundedField>, double>(), "calibration"_a,
            "instFluxMag0Err"_a = 0.0);
    cls.def(py::init<double, double, std::shared_ptr<afw::math::BoundedField>, bool>(), "instFluxMag0"_a,
            "instFluxMag0Err"_a, "calibration"_a, "isConstant"_a);

    table::io::python::addPersistableMethods<PhotoCalib>(cls);

    /* Members - maggies */
    cls.def("instFluxToMaggies",
            (double (PhotoCalib::*)(double, afw::geom::Point<double, 2> const &) const) &
                    PhotoCalib::instFluxToMaggies,
            "instFlux"_a, "point"_a);
    cls.def("instFluxToMaggies", (double (PhotoCalib::*)(double) const) & PhotoCalib::instFluxToMaggies,
            "instFlux"_a);

    cls.def("instFluxToMaggies",
            (Measurement(PhotoCalib::*)(double, double, afw::geom::Point<double, 2> const &) const) &
                    PhotoCalib::instFluxToMaggies,
            "instFlux"_a, "instFluxErr"_a, "point"_a);
    cls.def("instFluxToMaggies",
            (Measurement(PhotoCalib::*)(double, double) const) & PhotoCalib::instFluxToMaggies, "instFlux"_a,
            "instFluxErr"_a);

    cls.def("instFluxToMaggies",
            (Measurement(PhotoCalib::*)(afw::table::SourceRecord const &, std::string const &) const) &
                    PhotoCalib::instFluxToMaggies,
            "sourceRecord"_a, "instFluxField"_a);

    cls.def("instFluxToMaggies",
            (ndarray::Array<double, 2, 2>(PhotoCalib::*)(afw::table::SourceCatalog const &,
                                                         std::string const &) const) &
                    PhotoCalib::instFluxToMaggies,
            "sourceCatalog"_a, "instFluxField"_a);

    cls.def("instFluxToMaggies",
            (void (PhotoCalib::*)(afw::table::SourceCatalog &, std::string const &, std::string const &)
                     const) &
                    PhotoCalib::instFluxToMaggies,
            "sourceCatalog"_a, "instFluxField"_a, "outField"_a);

    /* Members - magnitudes */
    cls.def("instFluxToMagnitude",
            (double (PhotoCalib::*)(double, afw::geom::Point<double, 2> const &) const) &
                    PhotoCalib::instFluxToMagnitude,
            "instFlux"_a, "point"_a);
    cls.def("instFluxToMagnitude", (double (PhotoCalib::*)(double) const) & PhotoCalib::instFluxToMagnitude,
            "instFlux"_a);

    cls.def("instFluxToMagnitude",
            (Measurement(PhotoCalib::*)(double, double, afw::geom::Point<double, 2> const &) const) &
                    PhotoCalib::instFluxToMagnitude,
            "instFlux"_a, "instFluxErr"_a, "point"_a);
    cls.def("instFluxToMagnitude",
            (Measurement(PhotoCalib::*)(double, double) const) & PhotoCalib::instFluxToMagnitude,
            "instFlux"_a, "instFluxErr"_a);

    cls.def("instFluxToMagnitude",
            (Measurement(PhotoCalib::*)(afw::table::SourceRecord const &, std::string const &) const) &
                    PhotoCalib::instFluxToMagnitude,
            "sourceRecord"_a, "instFluxField"_a);

    cls.def("instFluxToMagnitude",
            (ndarray::Array<double, 2, 2>(PhotoCalib::*)(afw::table::SourceCatalog const &,
                                                         std::string const &) const) &
                    PhotoCalib::instFluxToMagnitude,
            "sourceCatalog"_a, "instFluxField"_a);

    cls.def("instFluxToMagnitude",
            (void (PhotoCalib::*)(afw::table::SourceCatalog &, std::string const &, std::string const &)
                     const) &
                    PhotoCalib::instFluxToMagnitude,
            "sourceCatalog"_a, "instFluxField"_a, "outField"_a);

    /* utilities */
    cls.def("getCalibrationMean", &PhotoCalib::getCalibrationMean);
    cls.def("getCalibrationErr", &PhotoCalib::getCalibrationErr);
    cls.def("getInstFluxMag0", &PhotoCalib::getInstFluxMag0);

    cls.def("magnitudeToInstFlux", (double (PhotoCalib::*)(double) const) & PhotoCalib::magnitudeToInstFlux,
            "magnitude"_a);

    cls.def("computeScaledCalibration", &PhotoCalib::computeScaledCalibration);
    cls.def("computeScalingTo", &PhotoCalib::computeScalingTo);

    /* Operators */
    cls.def("__eq__", &PhotoCalib::operator==, py::is_operator());
    cls.def("__ne__", &PhotoCalib::operator!=, py::is_operator());

    cls.def("__str__", [](PhotoCalib const &self) {
        std::ostringstream os;
        os << self;
        return os.str();
    });
    cls.def("__repr__", [](PhotoCalib const &self) {
        std::ostringstream os;
        os << "PhotoCalib(" << self << ")";
        return os.str();
    });

    return mod.ptr();
}
}  // namespace
}  // namespace image
}  // namespace afw
}  // namespace lsst
