/*
 * This file is part of afw.
 *
 * Developed for the LSST Data Management System.
 * This product includes software developed by the LSST Project
 * (https://www.lsst.org).
 * See the COPYRIGHT file at the top-level directory of this distribution
 * for details of code ownership.
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
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "lsst/utils/python.h"

#include "lsst/afw/detection/FootprintMerge.h"

namespace py = pybind11;
using namespace py::literals;

namespace lsst {
namespace afw {
namespace detection {

void wrapFootprintMerge(utils::python::WrapperCollection &wrappers) {
    wrappers.addSignatureDependency("lsst.afw.table");

    wrappers.wrapType(
            py::class_<FootprintMergeList>(wrappers.module, "FootprintMergeList"), [](auto &mod, auto &cls) {
                cls.def(py::init<afw::table::Schema &, std::vector<std::string> const &,
                                 afw::table::Schema const &>(),
                        "sourceSchema"_a, "filterList"_a, "initialPeakSchema"_a);
                cls.def(py::init<afw::table::Schema &, std::vector<std::string> const &>(), "sourceSchema"_a,
                        "filterList"_a);

                cls.def("getPeakSchema", &FootprintMergeList::getPeakSchema);
                cls.def("addCatalog", &FootprintMergeList::addCatalog, "sourceTable"_a, "inputCat"_a,
                        "filter"_a, "minNewPeakDist"_a = -1., "doMerge"_a = true, "maxSamePeakDist"_a = -1.);
                cls.def("clearCatalog", &FootprintMergeList::clearCatalog);
                cls.def("getFinalSources", &FootprintMergeList::getFinalSources, "outputCat"_a);
            });
}

}  // namespace detection
}  // namespace afw
}  // namespace lsst
