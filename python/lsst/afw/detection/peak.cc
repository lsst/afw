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
//#include <pybind11/stl.h>

#include "lsst/afw/table/BaseRecord.h"
#include "lsst/afw/detection/Peak.h"
#include "lsst/afw/table/pybind11/catalog.h"

namespace py = pybind11;
using namespace py::literals;

namespace lsst {
namespace afw {
namespace detection {

PYBIND11_PLUGIN(_peak) {
    py::module mod("_peak", "Python wrapper for afw _peak library");

    py::class_<PeakRecord, std::shared_ptr<PeakRecord>, lsst::afw::table::BaseRecord> clsPeakRecord(mod, "PeakRecord");

    clsPeakRecord.def("getTable", &PeakRecord::getTable);
    clsPeakRecord.def("getId", &PeakRecord::getId);
    clsPeakRecord.def("setId", &PeakRecord::setId);
    clsPeakRecord.def("getIx", &PeakRecord::getIx);
    clsPeakRecord.def("getIy", &PeakRecord::getIy);
    clsPeakRecord.def("setIx", &PeakRecord::setIx);
    clsPeakRecord.def("setIy", &PeakRecord::setIy);
    clsPeakRecord.def("getI", &PeakRecord::getI);
    clsPeakRecord.def("getCentroid", (afw::geom::Point2I (PeakRecord::*)(bool) const) &PeakRecord::getCentroid);
    clsPeakRecord.def("getCentroid", (afw::geom::Point2D (PeakRecord::*)() const) &PeakRecord::getCentroid);
    clsPeakRecord.def("getFx", &PeakRecord::getFx);
    clsPeakRecord.def("getFy", &PeakRecord::getFy);
    clsPeakRecord.def("setFx", &PeakRecord::setFx);
    clsPeakRecord.def("setFy", &PeakRecord::setFy);
    clsPeakRecord.def("getF", &PeakRecord::getF);
    clsPeakRecord.def("getPeakValue", &PeakRecord::getPeakValue);
    clsPeakRecord.def("setPeakValue", &PeakRecord::setPeakValue);

    py::class_<table::CatalogT<PeakRecord>, std::shared_ptr<table::CatalogT<PeakRecord>>> clsPeakRecordCatalog(mod, "PeakRecordCatalog");
    declareCatalog(clsPeakRecordCatalog);
    /* Module level */

    /* Member types and enums */

    /* Constructors */

    /* Operators */

    /* Members */

    return mod.ptr();
}
}}} // lsst::afw::detection