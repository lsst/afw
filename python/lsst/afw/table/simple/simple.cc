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

#include "pybind11/pybind11.h"
#include "ndarray/pybind11.h"

#include "lsst/afw/table/BaseRecord.h"
#include "lsst/afw/table/BaseTable.h"
#include "lsst/afw/table/Simple.h"
#include "lsst/afw/table/python/catalog.h"
#include "lsst/afw/table/python/columnView.h"
#include "lsst/afw/table/python/sortedCatalog.h"

namespace py = pybind11;
using namespace py::literals;

namespace lsst {
namespace afw {
namespace table {
namespace {

using PySimpleTable = py::class_<SimpleTable, std::shared_ptr<SimpleTable>, BaseTable>;
using PySimpleRecord = py::class_<SimpleRecord, std::shared_ptr<SimpleRecord>, BaseRecord>;

PySimpleRecord declareSimpleRecord(py::module &mod) {
    PySimpleRecord cls(mod, "SimpleRecord");
    cls.def("getId", &SimpleRecord::getId);
    cls.def("setId", &SimpleRecord::setId);
    cls.def("getTable", &SimpleRecord::getTable);
    cls.def_property_readonly("table", &SimpleRecord::getTable);
    cls.def("getCoord", &SimpleRecord::getCoord);
    cls.def("setCoord", &SimpleRecord::setCoord);
    cls.def("getRa", &SimpleRecord::getRa);
    cls.def("setRa", &SimpleRecord::setRa);
    cls.def("getDec", &SimpleRecord::getDec);
    cls.def("setDec", &SimpleRecord::setDec);
    return cls;
}

PySimpleTable declareSimpleTable(py::module &mod) {
    PySimpleTable cls(mod, "SimpleTable");
    cls.def_static("make",
                   (std::shared_ptr<SimpleTable>(*)(Schema const &, std::shared_ptr<IdFactory> const &)) &
                           SimpleTable::make);
    cls.def_static("make", (std::shared_ptr<SimpleTable>(*)(Schema const &)) & SimpleTable::make);
    cls.def_static("makeMinimalSchema", &SimpleTable::makeMinimalSchema);
    cls.def_static("checkSchema", &SimpleTable::checkSchema, "schema"_a);
    cls.def_static("getIdKey", &SimpleTable::getIdKey);
    cls.def_static("getCoordKey", &SimpleTable::getCoordKey);

    cls.def("getIdFactory", (std::shared_ptr<IdFactory> (SimpleTable::*)()) & SimpleTable::getIdFactory);
    cls.def("setIdFactory", &SimpleTable::setIdFactory, "idFactory"_a);
    cls.def("clone", &SimpleTable::clone);
    cls.def("makeRecord", &SimpleTable::makeRecord);
    cls.def("copyRecord",
            (std::shared_ptr<SimpleRecord> (SimpleTable::*)(BaseRecord const &)) & SimpleTable::copyRecord,
            "other"_a);
    cls.def("copyRecord",
            (std::shared_ptr<SimpleRecord> (SimpleTable::*)(BaseRecord const &, SchemaMapper const &)) &
                    SimpleTable::copyRecord,
            "other"_a, "mapper"_a);
    return cls;
}

}  // namespace lsst::afw::table::<anonymous>

PYBIND11_MODULE(simple, mod) {
    py::module::import("lsst.afw.coord");  // needed, but why? Perhaps can be removed after RFC-460
    py::module::import("lsst.afw.table.idFactory");

    auto clsSimpleRecord = declareSimpleRecord(mod);
    auto clsSimpleTable = declareSimpleTable(mod);
    auto clsSimpleColumnView = table::python::declareColumnView<SimpleRecord>(mod, "Simple");
    auto clsSimpleCatalog = table::python::declareSortedCatalog<SimpleRecord>(mod, "Simple");

    clsSimpleRecord.attr("Table") = clsSimpleTable;
    clsSimpleRecord.attr("ColumnView") = clsSimpleColumnView;
    clsSimpleRecord.attr("Catalog") = clsSimpleCatalog;
    clsSimpleTable.attr("Record") = clsSimpleRecord;
    clsSimpleTable.attr("ColumnView") = clsSimpleColumnView;
    clsSimpleTable.attr("Catalog") = clsSimpleCatalog;
    clsSimpleCatalog.attr("Record") = clsSimpleRecord;
    clsSimpleCatalog.attr("Table") = clsSimpleTable;
    clsSimpleCatalog.attr("ColumnView") = clsSimpleColumnView;
}
}
}
}  // namespace lsst::afw::table
