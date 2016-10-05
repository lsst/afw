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

#include "lsst/afw/image/TanWcs.h"
#include "lsst/afw/image/Wcs.h"
#include "lsst/afw/table/io/Persistable.h"

#include "lsst/afw/table/io/pybind11.h"

namespace py = pybind11;

using namespace lsst::afw::image;

using lsst::afw::table::io::PersistableFacade;

PYBIND11_PLUGIN(_tanWcs) {
    py::module mod("_tanWcs", "Python wrapper for afw _tanWcs library");

    lsst::afw::table::io::declarePersistableFacade<TanWcs>(mod, "TanWcs");

    py::class_<TanWcs, std::shared_ptr<TanWcs>, PersistableFacade<TanWcs>, lsst::afw::image::Wcs> clsTanWcs(mod, "TanWcs");

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