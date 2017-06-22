/*
 * LSST Data Management System
 * Copyright 2017 LSST Corporation.
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the LSST License Statement and
 * the GNU General Public License along with this program. If not,
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */
#include "pybind11/pybind11.h"

#include <memory>

#include "astshim.h"
#include "numpy/arrayobject.h"
#include "ndarray/pybind11.h"

#include "lsst/daf/base.h"
#include "lsst/afw/geom/Angle.h"
#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/detail/frameSetUtils.h"

namespace py = pybind11;
using namespace py::literals;

namespace lsst {
namespace afw {
namespace geom {
namespace detail {
namespace {

PYBIND11_PLUGIN(frameSetUtils) {
    py::module mod("frameSetUtils");

    py::module::import("lsst.daf.base");
    py::module::import("lsst.afw.geom.angle");
    py::module::import("lsst.afw.geom.coordinates");
    py::module::import("lsst.afw.geom.spherePoint");

    // Need to import numpy for ndarray and eigen conversions
    if (_import_array() < 0) {
        PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
        return nullptr;
    }

    mod.def("getSkyFrame", getSkyFrame, "frameSet"_a, "index"_a, "copy"_a);
    mod.def("makeTanWcsMetadata", makeTanWcsMetadata, "crpix"_a, "crval"_a, "cdMatrix"_a);
    mod.def("readFitsWcs", readFitsWcs, "metadata"_a, "strip"_a = true);
    mod.def("readLsstSkyWcs", readLsstSkyWcs, "metadata"_a, "strip"_a = true);

    return mod.ptr();
}

}  // namespace
}  // namespace detail
}  // namespace geom
}  // namespace afw
}  // namespace lsst
