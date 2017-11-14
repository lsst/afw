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

#include "lsst/afw/fits.h"
#include "lsst/afw/image/TransmissionCurve.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst {
namespace afw {
namespace image {
namespace {

using PyTransmissionCurve =
    py::class_<TransmissionCurve, std::shared_ptr<TransmissionCurve>, table::io::Persistable>;

using PySampleDef = py::class_<TransmissionCurve::SampleDef>;

PyTransmissionCurve declare(py::module & mod) {
    PyTransmissionCurve cls(mod, "TransmissionCurve");
    // safe to put full definition for SampleDef here, since it only uses PODs
    // and ndarray.
    PySampleDef clsSampleDef(cls, "SampleDef");
    clsSampleDef.def(py::init<double,double,int>(), "min"_a, "max"_a, "size"_a);
    clsSampleDef.def_readwrite("min", &TransmissionCurve::SampleDef::min);
    clsSampleDef.def_readwrite("max", &TransmissionCurve::SampleDef::max);
    clsSampleDef.def_readwrite("size", &TransmissionCurve::SampleDef::size);
    clsSampleDef.def("getSpacing", &TransmissionCurve::SampleDef::getSpacing);
    clsSampleDef.def("makeArray", &TransmissionCurve::SampleDef::makeArray);
    return cls;
}

void define(PyTransmissionCurve & cls) {
    // From PersistableFacade: we don't declare the CRTP base class to Python because its only
    // purpose is downcasting the result, and pybind11 does that automatically.
    // We should be able to remove these calls after DM-????
    cls.def_static("readFits",
                   (std::shared_ptr<TransmissionCurve>(*)(std::string const &, int))
                   &TransmissionCurve::readFits,
                   "fileName"_a, "hdu"_a = INT_MIN);
    cls.def_static("readFits",
                   (std::shared_ptr<TransmissionCurve>(*)(fits::MemFileManager &, int))
                   &TransmissionCurve::readFits,
                   "manager"_a, "hdu"_a = INT_MIN);
    cls.def_static("makeIdentity", &TransmissionCurve::makeIdentity);
    cls.def_static("makeConstant", &TransmissionCurve::makeConstant,
                   "throughput"_a, "wavelengths"_a,
                   "throughputAtMin"_a=0.0, "throughputAtMax"_a=0.0);
    cls.def_static("makeRadial", &TransmissionCurve::makeRadial,
                   "throughput"_a, "wavelengths"_a, "radii"_a,
                   "throughputAtMin"_a=0.0, "throughputAtMax"_a=0.0);
    cls.def("__mul__", &TransmissionCurve::multiply, py::is_operator());
    cls.def("transform", &TransmissionCurve::transform);
    cls.def("getNaturalSampling", &TransmissionCurve::getNaturalSampling);
    cls.def("getThroughputAtBounds", &TransmissionCurve::getThroughputAtBounds);
    cls.def(
        "sampleAt",
        (void (TransmissionCurve::*)(
            geom::Point2D const &,
            ndarray::Array<double const,1,1> const &,
            ndarray::Array<double,1,1> const &
        ) const) &TransmissionCurve::sampleAt,
        "position"_a, "wavelengths"_a, "out"_a
    );
    cls.def(
        "sampleAt",
        (void (TransmissionCurve::*)(
            geom::Point2D const &,
            TransmissionCurve::SampleDef const &,
            ndarray::Array<double,1,1> const &
        ) const) &TransmissionCurve::sampleAt,
        "position"_a, "wavelengths"_a, "out"_a
    );
    cls.def(
        "sampleAt",
        (ndarray::Array<double,1,1> (TransmissionCurve::*)(
            geom::Point2D const &,
            ndarray::Array<double const,1,1> const &
        ) const) &TransmissionCurve::sampleAt,
        "position"_a, "wavelengths"_a
    );
    cls.def(
        "sampleAt",
        (ndarray::Array<double,1,1> (TransmissionCurve::*)(
            geom::Point2D const &,
            TransmissionCurve::SampleDef const &
        ) const) &TransmissionCurve::sampleAt,
        "position"_a, "wavelengths"_a
    );
}

PYBIND11_PLUGIN(transmissionCurve) {
    py::module mod("transmissionCurve");

    if (_import_array() < 0) {
        PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
        return nullptr;
    };

    // Ordering to avoid circular dependencies:
    // first import dependencies that provide base classes
    py::module::import("lsst.afw.table.io");
    // then declare classes
    auto cls = declare(mod);
    // then import dependencies used in method signatures
    py::module::import("lsst.afw.geom");
    // and now we can safely define methods and other attributes
    define(cls);

    return mod.ptr();
}

}}}}  // namespace lsst::afw::image::<anonymous>

