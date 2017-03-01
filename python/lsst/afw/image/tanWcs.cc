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
//#include <pybind11/stl.h>

#include "numpy/arrayobject.h"
#include "ndarray/pybind11.h"
#include "ndarray/converter.h"

#include "lsst/afw/image/TanWcs.h"
#include "lsst/afw/image/Wcs.h"
#include "lsst/afw/table/io/Persistable.h"

#include "lsst/afw/table/io/python.h"

namespace py = pybind11;

using namespace py::literals;

using lsst::afw::table::io::PersistableFacade;

namespace lsst { namespace afw { namespace image {

PYBIND11_PLUGIN(_tanWcs) {
    py::module mod("_tanWcs", "Python wrapper for afw _tanWcs library");

    if (_import_array() < 0) {
            PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
            return nullptr;
        }

    lsst::afw::table::io::python::declarePersistableFacade<TanWcs>(mod, "TanWcs");

    py::class_<TanWcs, std::shared_ptr<TanWcs>, PersistableFacade<TanWcs>, lsst::afw::image::Wcs> clsTanWcs(mod, "TanWcs");

    clsTanWcs.def(py::init<geom::Point2D const &,
                           geom::Point2D const &,
                           Eigen::Matrix2d const &,
                           double,
                           std::string const &,
                           std::string const &,
                           std::string const &>(),
                           "crval"_a,
                           "crpix"_a,
                           "cd"_a,
                           "equinox"_a=2000,
                           "raDecSys"_a="FK5",
                           "crunits1"_a="deg",
                           "crunits2"_a="deg");
    clsTanWcs.def(py::init<geom::Point2D const &,
                           geom::Point2D const &,
                           Eigen::Matrix2d const &,
                           Eigen::MatrixXd const &,
                           Eigen::MatrixXd const &,
                           Eigen::MatrixXd const &,
                           Eigen::MatrixXd const &,
                           double,
                           std::string const &,
                           std::string const &,
                           std::string const &>(),
                           "crval"_a,
                           "crpix"_a,
                           "cd"_a,
                           "sipA"_a,
                           "sipB"_a,
                           "sipAp"_a,
                           "sipBp"_a,
                           "equinox"_a=2000,
                           "raDecSys"_a="FK5",
                           "cunits1"_a="deg",
                           "cunits2"_a="deg");

    /* Operators */
    clsTanWcs.def(py::self == py::self);
    clsTanWcs.def(py::self != py::self);

    /* Members */
    clsTanWcs.def_static("decodeSipHeader", TanWcs::decodeSipHeader);
    clsTanWcs.def("clone", &TanWcs::clone);
    clsTanWcs.def("pixelScale", &TanWcs::pixelScale);
    clsTanWcs.def("distortPixel", &TanWcs::distortPixel);
    clsTanWcs.def("undistortPixel", &TanWcs::undistortPixel);
    clsTanWcs.def("hasDistortion", &TanWcs::hasDistortion);
    clsTanWcs.def("getFitsMetadata", &TanWcs::getFitsMetadata);
    clsTanWcs.def("setDistortionMatrices", &TanWcs::setDistortionMatrices);
    clsTanWcs.def("isPersistable", &TanWcs::isPersistable);

    return mod.ptr();
}
}}} // lsst::afw::image