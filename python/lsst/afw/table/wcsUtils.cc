/*
 * LSST Data Management System
 * Copyright 2018  AURA/LSST.
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

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "lsst/afw/table/fwd.h"
#include "lsst/afw/table/Simple.h"
#include "lsst/afw/table/Source.h"
#include "lsst/afw/table/wcsUtils.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst {
namespace afw {
namespace table {
namespace {

template <typename ReferenceCollection>
void declareUpdateRefCentroids(py::module &mod) {
    mod.def("updateRefCentroids", updateRefCentroids<ReferenceCollection>, "wcs"_a, "refList"_a);
}

template <typename SourceCollection>
void declareUpdateSourceCoords(py::module &mod) {
    mod.def("updateSourceCoords", updateSourceCoords<SourceCollection>, "wcs"_a, "sourceList"_a);
}

PYBIND11_PLUGIN(wcsUtils) {
    py::module mod("wcsUtils");

    py::module::import("lsst.afw.table.aggregates");
    py::module::import("lsst.afw.table.simple");
    py::module::import("lsst.afw.table.source");

    declareUpdateRefCentroids<std::vector<std::shared_ptr<lsst::afw::table::SimpleRecord>>>(mod);
    declareUpdateRefCentroids<lsst::afw::table::SimpleCatalog>(mod);

    declareUpdateSourceCoords<std::vector<std::shared_ptr<lsst::afw::table::SourceRecord>>>(mod);
    declareUpdateSourceCoords<lsst::afw::table::SourceCatalog>(mod);

    return mod.ptr();
}

}  // namespace
}  // namespace table
}  // namespace afw
}  // namespace lsst
