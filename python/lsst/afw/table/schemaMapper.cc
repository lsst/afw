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

#include "lsst/afw/table/Field.h"
#include "lsst/afw/table/Key.h"
#include "lsst/afw/table/Schema.h"
#include "lsst/afw/table/detail/SchemaImpl.h"
#include "lsst/afw/table/SchemaMapper.h"

namespace py = pybind11;
using namespace pybind11::literals;

using namespace lsst::afw::table;

template <typename T>
void declareSchemaMapperOverloads(py::class_<SchemaMapper> & clsSchemaMapper, const std::string & suffix){
    clsSchemaMapper.def("addOutputField", (Key<T> (SchemaMapper::*)(Field<T> const &, bool))
        &SchemaMapper::addOutputField, "newField"_a, "doReplace"_a=false);
    clsSchemaMapper.def("addMapping",
        (Key<T> (SchemaMapper::*)(Key<T> const &, bool)) &SchemaMapper::addMapping,
        "inputKey"_a, "doReplace"_a=false);
    clsSchemaMapper.def("addMapping",
                        (Key<T> (SchemaMapper::*)(Key<T> const &, Field<T> const &, bool))
                            &SchemaMapper::addMapping,
                        "inputKey"_a, "outputField"_a, "doReplace"_a=false);
    clsSchemaMapper.def("addMapping", 
                        (Key<T> (SchemaMapper::*)(Key<T> const &, std::string const &, bool))
                            &SchemaMapper::addMapping,
                        "inputKey"_a, "outputName"_a, "doReplace"_a=true);
    clsSchemaMapper.def("getMapping", (Key<T> (SchemaMapper::*)(Key<T> const &) const)
        &SchemaMapper::getMapping);
};

PYBIND11_PLUGIN(_schemaMapper) {
    py::module mod("_schemaMapper", "Python wrapper for afw _schemaMapper library");

    /* Module level */
    py::class_<SchemaMapper> clsSchemaMapper(mod, "SchemaMapper");

    /* Member types and enums */

    /* Constructors */
    clsSchemaMapper.def(py::init<>());
    clsSchemaMapper.def(py::init<Schema const &, Schema const &>());
    clsSchemaMapper.def(py::init<Schema const &, bool>(), "input"_a, "shareAliasMap"_a=false);

    /* Operators */

    /* Members */
    clsSchemaMapper.def("getInputSchema", &SchemaMapper::getInputSchema);
    clsSchemaMapper.def("getOutputSchema", &SchemaMapper::getOutputSchema);
    clsSchemaMapper.def("editOutputSchema", &SchemaMapper::editOutputSchema);
    clsSchemaMapper.def("addMinimalSchema", &SchemaMapper::addMinimalSchema,
                        "minimal"_a, "doMap"_a=true);
    clsSchemaMapper.def_static("removeMinimalSchema", &SchemaMapper::removeMinimalSchema);
    clsSchemaMapper.def_static("join", &SchemaMapper::join,
                               "inputs"_a, "prefixes"_a=std::vector<std::string>());
    declareSchemaMapperOverloads<std::uint16_t>(clsSchemaMapper, "U");
    declareSchemaMapperOverloads<std::int32_t>(clsSchemaMapper, "I");
    declareSchemaMapperOverloads<std::int64_t>(clsSchemaMapper, "L");
    declareSchemaMapperOverloads<float>(clsSchemaMapper, "F");
    declareSchemaMapperOverloads<double>(clsSchemaMapper, "D");
    declareSchemaMapperOverloads<std::string>(clsSchemaMapper, "String");
    declareSchemaMapperOverloads<lsst::afw::geom::Angle>(clsSchemaMapper, "Angle");
    declareSchemaMapperOverloads<lsst::afw::table::Flag>(clsSchemaMapper, "Flag");
    declareSchemaMapperOverloads<lsst::afw::table::Array<std::uint16_t>>(clsSchemaMapper, "ArrayU");
    declareSchemaMapperOverloads<lsst::afw::table::Array<int>>(clsSchemaMapper, "ArrayI");
    declareSchemaMapperOverloads<lsst::afw::table::Array<float>>(clsSchemaMapper, "ArrayF");
    declareSchemaMapperOverloads<lsst::afw::table::Array<double>>(clsSchemaMapper, "ArrayD");

    return mod.ptr();
}