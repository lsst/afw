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
#include "ndarray/eigen.h"

#include "lsst/daf/base.h"
#include "lsst/afw/geom/Angle.h"
#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/detail/wcsUtils.h"

namespace py = pybind11;
using namespace py::literals;

namespace lsst {
namespace afw {
namespace geom {
namespace detail {
namespace {

PYBIND11_PLUGIN(wcsUtils) {
    py::module mod("wcsUtils");

    // Need to import numpy for ndarray and eigen conversions
    if (_import_array() < 0) {
        PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
        return nullptr;
    }

    mod.def("createTrivialWcsAsPropertySet", createTrivialWcsAsPropertySet, "wcsName"_a, "xy0"_a);
    mod.def("deleteBasicWcsMetadata", deleteBasicWcsMetadata, "metadata"_a, "wcsName"_a);
    mod.def("getCdMatrixFromMetadata", getCdMatrixFromMetadata, "metadata"_a);
    mod.def("getImageXY0FromMetadata", getImageXY0FromMetadata, "metadata"_a, "wcsName"_a, "strip"_a = false);
    mod.def("getSipMatrixFromMetadata", getSipMatrixFromMetadata, "metadata"_a, "name"_a);
    mod.def("hasSipMatrix", hasSipMatrix, "metadata"_a, "name"_a);
    mod.def("makeSipMatrixMetadata", makeSipMatrixMetadata, "matrix"_a, "name"_a);
    mod.def("makeTanSipMetadata",
            (std::shared_ptr<daf::base::PropertyList>(*)(Point2D const&, coord::IcrsCoord const&,
                                                         Eigen::Matrix2d const&, Eigen::MatrixXd const&,
                                                         Eigen::MatrixXd const&))makeTanSipMetadata,
            "crpix"_a, "crval"_a, "cdMatrix"_a, "sipA"_a, "sipB"_a);
    mod.def("makeTanSipMetadata",
            (std::shared_ptr<daf::base::PropertyList>(*)(Point2D const&, coord::IcrsCoord const&,
                                                         Eigen::Matrix2d const&, Eigen::MatrixXd const&,
                                                         Eigen::MatrixXd const&, Eigen::MatrixXd const&,
                                                         Eigen::MatrixXd const&))makeTanSipMetadata,
            "crpix"_a, "crval"_a, "cdMatrix"_a, "sipA"_a, "sipB"_a, "sipAp"_a, "sipBp"_a);
    return mod.ptr();
}

}  // namespace
}  // namespace detail
}  // namespace geom
}  // namespace afw
}  // namespace lsst
