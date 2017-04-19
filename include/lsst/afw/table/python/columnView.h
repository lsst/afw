/*
 * LSST Data Management System
 * Copyright 2008-2017  AURA/LSST.
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
#ifndef AFW_TABLE_PYBIND11_COLUMNVIEW_H_INCLUDED
#define AFW_TABLE_PYBIND11_COLUMNVIEW_H_INCLUDED

#include "pybind11/pybind11.h"

#include "lsst/afw/table/BaseColumnView.h"

namespace lsst {
namespace afw {
namespace table {
namespace python {

template <typename Record>
using PyColumnView =
        pybind11::class_<ColumnViewT<Record>, std::shared_ptr<ColumnViewT<Record>>, BaseColumnView>;

/**
Declare member and static functions for a given instantiation of lsst::afw::table::ColumnViewT<RecordT>.

@tparam Record  Record type, e.g. BaseRecord or SimpleRecord.

@param[in] mod    Module object class will be added to.
@param[in] name   Name prefix of the record type, e.g. "Base" or "Simple".
@param[in] isBase Whether this instantiation is only being used as a base class (used to set the class name).
*/
template <typename Record>
PyColumnView<Record> declareColumnView(pybind11::module& mod, std::string const& name, bool isBase = false) {
    std::string fullName;
    if (isBase) {
        fullName = "_" + name + "ColumnViewBase";
    } else {
        fullName = name + "ColumnView";
    }
    PyColumnView<Record> cls(mod, fullName.c_str());
    cls.def("getTable", &ColumnViewT<Record>::getTable);
    cls.def_property_readonly("table", &ColumnViewT<Record>::getTable);
    return cls;
};
}
}
}
}  // namespace lsst::afw::table::python
#endif
