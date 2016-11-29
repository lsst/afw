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

#include "lsst/afw/table/misc.h"
#include "lsst/afw/table/Key.h"
#include "lsst/afw/table/Field.h"
#include "lsst/afw/table/detail/SchemaImpl.h"

namespace py = pybind11;

using namespace lsst::afw::table;
using namespace detail;

template <typename T>
void declareSchemaItem(py::module & mod, const std::string suffix) {
    py::class_<SchemaItem<T>> clsSchemaItem(mod, ("SchemaItem_"+suffix).c_str());
    clsSchemaItem.def_readwrite("key", &SchemaItem<T>::key);
    clsSchemaItem.def_readwrite("field", &SchemaItem<T>::field);
};

PYBIND11_PLUGIN(_schemaImpl) {
    py::module mod("_schemaImpl", "Python wrapper for afw _schemaImpl library");

    /* Module level */
    py::class_<SchemaImpl, std::shared_ptr<SchemaImpl>> clsSchemaImpl(mod, "SchemaImpl");

    /* Member types and enums */

    /* Constructors */

    /* Operators */

    /* Members */
    declareSchemaItem<std::uint16_t>(mod, "U");
    declareSchemaItem<std::int32_t>(mod, "I");
    declareSchemaItem<std::int64_t>(mod, "L");
    declareSchemaItem<float>(mod, "F");
    declareSchemaItem<double>(mod, "D");
    declareSchemaItem<std::string>(mod, "String");
    declareSchemaItem<lsst::afw::geom::Angle>(mod, "Angle");
    declareSchemaItem<lsst::afw::table::Flag>(mod, "Flag");
    declareSchemaItem<lsst::afw::table::Array<std::uint16_t>>(mod, "ArrayU");
    declareSchemaItem<lsst::afw::table::Array<int>>(mod, "ArrayI");
    declareSchemaItem<lsst::afw::table::Array<float>>(mod, "ArrayF");
    declareSchemaItem<lsst::afw::table::Array<double>>(mod, "ArrayD");

    return mod.ptr();
}