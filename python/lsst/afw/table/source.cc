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

#include "lsst/afw/table/Simple.h"
#include "lsst/afw/table/Schema.h"
#include "lsst/afw/table/Source.h"

namespace py = pybind11;

using namespace lsst::afw::table;

template <typename RecordT>
void declareSourceColumnView(py::module & mod){
    py::class_<SourceColumnViewT<RecordT>, std::shared_ptr<SourceColumnViewT<RecordT>>> cls(mod, "SourceColumnViewT");
    //cls.def_static("make", &SourceColumnViewT<RecordT>::make);
};

PYBIND11_PLUGIN(_source) {
    py::module mod("_source", "Python wrapper for afw _source library");

    /* Module level */
    py::class_<SourceTable, std::shared_ptr<SourceTable>, SimpleTable> clsSourceTable(mod, "SourceTable");
    py::class_<SourceRecord, std::shared_ptr<SourceRecord>, SimpleRecord> clsSourceRecord(mod, "SourceRecord");

    /* Member types and enums */

    /* Constructors */

    /* Operators */

    /* Members */
    clsSourceTable.def_static("make", (PTR(SourceTable) (*)(Schema const &, PTR(IdFactory) const &))
        &SourceTable::make);
    clsSourceTable.def_static("make", (PTR(SourceTable) (*)(Schema const &)) &SourceTable::make);
    clsSourceTable.def_static("makeMinimalSchema", &SourceTable::makeMinimalSchema);
    clsSourceTable.def("copyRecord", (PTR(SourceRecord) (SourceTable::*)(BaseRecord const &)) & SourceTable::copyRecord);
    clsSourceTable.def("copyRecord", (PTR(SourceRecord) (SourceTable::*)(BaseRecord const &, SchemaMapper const &)) & SourceTable::copyRecord);
    clsSourceTable.def("makeRecord", &SourceTable::makeRecord);
    
    declareSourceColumnView<SourceRecord>(mod);

    return mod.ptr();
}