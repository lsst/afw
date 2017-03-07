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
//#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include "lsst/afw/detection/FootprintMerge.h"

namespace py = pybind11;
using namespace py::literals;

namespace lsst {
namespace afw {
namespace detection {

PYBIND11_PLUGIN(_footprintMerge) {
    py::module mod("_footprintMerge", "Python wrapper for afw _footprintMerge library");

    py::class_<FootprintMergeList> clsFootprintMergeList(mod, "FootprintMergeList");

    /* Constructors */
    clsFootprintMergeList.def(py::init<afw::table::Schema &, std::vector<std::string> const &, afw::table::Schema const &>(),
            "sourceSchema"_a, "filterList"_a, "initialPeakSchema"_a);
    clsFootprintMergeList.def(py::init<afw::table::Schema &, std::vector<std::string> const &>(),
           "sourceSchema"_a, "filterList"_a); 

    /* Members */
    clsFootprintMergeList.def("getPeakSchema", &FootprintMergeList::getPeakSchema);
    clsFootprintMergeList.def("addCatalog", &FootprintMergeList::addCatalog,
            "sourceTable"_a, "inputCat"_a, "filter"_a, "minNewPeakDist"_a=-1., "doMerge"_a=true, "maxSamePeakDist"_a=-1.);
    clsFootprintMergeList.def("clearCatalog", &FootprintMergeList::clearCatalog);
    clsFootprintMergeList.def("getFinalSources", &FootprintMergeList::getFinalSources,
            "outputCat"_a, "doNorm"_a=true);

    return mod.ptr();
}
}}} // lsst::afw::detection