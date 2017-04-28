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

#include "numpy/arrayobject.h"
#include "ndarray/pybind11.h"

#include "lsst/afw/image/TanWcs.h"
#include "lsst/afw/image/Wcs.h"
#include "lsst/afw/table/io/Persistable.h"

#include "lsst/afw/table/io/python.h"

namespace py = pybind11;
using namespace py::literals;

namespace lsst {
namespace afw {
namespace image {

using PyTanWcs = py::class_<TanWcs, std::shared_ptr<TanWcs>, table::io::PersistableFacade<TanWcs>, Wcs>;

PYBIND11_PLUGIN(tanWcs) {
    py::module mod("tanWcs");

    py::module::import("lsst.afw.image.wcs");

    if (_import_array() < 0) {
        PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
        return nullptr;
    }

    table::io::python::declarePersistableFacade<TanWcs>(mod, "TanWcs");

    PyTanWcs cls(mod, "TanWcs");

    cls.def(py::init<geom::Point2D const &, geom::Point2D const &, Eigen::Matrix2d const &, double,
                     std::string const &, std::string const &, std::string const &>(),
            "crval"_a, "crpix"_a, "cd"_a, "equinox"_a = 2000, "raDecSys"_a = "FK5", "crunits1"_a = "deg",
            "crunits2"_a = "deg");
    cls.def(py::init<geom::Point2D const &, geom::Point2D const &, Eigen::Matrix2d const &,
                     Eigen::MatrixXd const &, Eigen::MatrixXd const &, Eigen::MatrixXd const &,
                     Eigen::MatrixXd const &, double, std::string const &, std::string const &,
                     std::string const &>(),
            "crval"_a, "crpix"_a, "cd"_a, "sipA"_a, "sipB"_a, "sipAp"_a, "sipBp"_a, "equinox"_a = 2000,
            "raDecSys"_a = "FK5", "cunits1"_a = "deg", "cunits2"_a = "deg");

    /* Operators */
    cls.def("__eq__", &TanWcs::operator==, py::is_operator());
    cls.def("__ne__", &TanWcs::operator!=, py::is_operator());

    /* Members */
    cls.def_static("decodeSipHeader", TanWcs::decodeSipHeader);
    cls.def("clone", &TanWcs::clone);
    cls.def("pixelScale", &TanWcs::pixelScale);
    cls.def("distortPixel", &TanWcs::distortPixel);
    cls.def("undistortPixel", &TanWcs::undistortPixel);
    cls.def("hasDistortion", &TanWcs::hasDistortion);
    cls.def("getFitsMetadata", &TanWcs::getFitsMetadata);
    cls.def("setDistortionMatrices", &TanWcs::setDistortionMatrices);
    cls.def("getSipA", &TanWcs::getSipA, py::return_value_policy::copy);
    cls.def("getSipB", &TanWcs::getSipB, py::return_value_policy::copy);
    cls.def("getSipAp", &TanWcs::getSipAp, py::return_value_policy::copy);
    cls.def("getSipBp", &TanWcs::getSipBp, py::return_value_policy::copy);
    cls.def("isPersistable", &TanWcs::isPersistable);

    return mod.ptr();
}
}
}
}  // lsst::afw::image