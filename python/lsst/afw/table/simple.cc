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

#include <memory>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "lsst/afw/coord/Coord.h"
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
using PySimpleColumnView = py::class_<ColumnViewT<SimpleRecord>,
                                      std::shared_ptr<ColumnViewT<SimpleRecord>>,
                                      BaseColumnView>;
using PyBaseSimpleCatalog = py::class_<CatalogT<SimpleRecord>, std::shared_ptr<CatalogT<SimpleRecord>>>;
using PySimpleCatalog = py::class_<SimpleCatalog, std::shared_ptr<SimpleCatalog>, CatalogT<SimpleRecord>>;

void declareSimpleRecord(PySimpleRecord & cls) {
    table::python::addCastFrom<BaseRecord>(cls);

    cls.def("getId", &SimpleRecord::getId);
    cls.def("setId", &SimpleRecord::setId);
    cls.def("getTable", &SimpleRecord::getTable);
    cls.def_property_readonly("table", &SimpleRecord::getTable);
    cls.def("getCoord", &SimpleRecord::getCoord);
    cls.def("setCoord", (void (SimpleRecord::*)(IcrsCoord const &)) &SimpleRecord::setCoord);
    cls.def("setCoord", (void (SimpleRecord::*)(Coord const &)) &SimpleRecord::setCoord);
    cls.def("getRa", &SimpleRecord::getRa);
    cls.def("setRa", &SimpleRecord::setRa);
    cls.def("getDec", &SimpleRecord::getDec);
    cls.def("setDec", &SimpleRecord::setDec);
}

void declareSimpleTable(PySimpleTable & cls) {
    table::python::addCastFrom<BaseTable>(cls);

    cls.def_static("make",
                   (std::shared_ptr<SimpleTable> (*)(Schema const &, std::shared_ptr<IdFactory> const &))
                        &SimpleTable::make);
    cls.def_static("make", (std::shared_ptr<SimpleTable> (*)(Schema const &)) &SimpleTable::make);
    cls.def_static("makeMinimalSchema", &SimpleTable::makeMinimalSchema);
    cls.def_static("checkSchema", &SimpleTable::checkSchema, "schema"_a);
    cls.def_static("getIdKey", &SimpleTable::getIdKey);
    cls.def_static("getCoordKey", &SimpleTable::getCoordKey);

    cls.def("getIdFactory", (std::shared_ptr<IdFactory> (SimpleTable::*)()) &SimpleTable::getIdFactory);
    cls.def("setIdFactory", &SimpleTable::setIdFactory, "idFactory"_a);
    cls.def("clone", &SimpleTable::clone);
    cls.def("makeRecord", &SimpleTable::makeRecord);
    cls.def("copyRecord",
            (std::shared_ptr<SimpleRecord> (SimpleTable::*)(BaseRecord const &)) &SimpleTable::copyRecord,
            "other"_a);
    cls.def("copyRecord",
            (std::shared_ptr<SimpleRecord> (SimpleTable::*)(BaseRecord const &, SchemaMapper const &))
                &SimpleTable::copyRecord,
            "other"_a, "mapper"_a);

}

}  // namespace lsst::afw::table::<anonymous>

PYBIND11_PLUGIN(_simple) {
    py::module mod("_simple", "Python wrapper for afw _simple library");

    /* Module level */
    PySimpleTable clsSimpleTable(mod, "SimpleTable");
    PySimpleRecord clsSimpleRecord(mod, "SimpleRecord");
    PySimpleColumnView clsSimpleColumnView(mod, "SimpleColumnView");
    PyBaseSimpleCatalog clsBaseSimpleCatalog(mod, "_BaseSimpleCatalog");
    PySimpleCatalog clsSimpleCatalog(mod, "SimpleCatalog", py::dynamic_attr());

    /* Members */
    declareSimpleRecord(clsSimpleRecord);
    declareSimpleTable(clsSimpleTable);
    table::python::declareColumnView(clsSimpleColumnView);
    table::python::declareCatalog(clsBaseSimpleCatalog);
    table::python::declareSortedCatalog(clsSimpleCatalog);

    clsSimpleRecord.attr("Table") = clsSimpleTable;
    clsSimpleRecord.attr("ColumnView") = clsSimpleColumnView;
    clsSimpleRecord.attr("Catalog") = clsSimpleCatalog;
    clsSimpleTable.attr("Record") = clsSimpleRecord;
    clsSimpleTable.attr("ColumnView") = clsSimpleColumnView;
    clsSimpleTable.attr("Catalog") = clsSimpleCatalog;
    clsSimpleCatalog.attr("Record") = clsSimpleRecord;
    clsSimpleCatalog.attr("Table") = clsSimpleTable;
    clsSimpleCatalog.attr("ColumnView") = clsSimpleColumnView;

    return mod.ptr();
}

}}}  // namespace lsst::afw::table
