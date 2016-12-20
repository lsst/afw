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
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "lsst/afw/table/misc.h"
#include "lsst/afw/table/Key.h"
#include "lsst/afw/table/Field.h"
#include "lsst/afw/table/FieldBase.h"
#include "lsst/afw/table/Flag.h"
#include "lsst/afw/table/detail/SchemaImpl.h"
#include "lsst/afw/table/Schema.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst {
namespace afw {
namespace table {

template <typename T>
void declareSchemaOverloads(py::class_<Schema> & clsSchema, std::string const & suffix) {
    clsSchema.def(("_addField_"+suffix).c_str(),
                  (Key<T> (Schema::*)(Field<T> const &, bool)) &Schema::addField,
                  "field"_a, "doReplace"_a=false);
    clsSchema.def(("_addField_"+suffix).c_str(),
                  (Key<T> (Schema::*)(std::string const &,
                                    std::string const &,
                                    std::string const &,
                                    FieldBase<T> const &,
                                    bool)) &Schema::addField,
                  "name"_a, "doc"_a, "units"_a="", "base"_a=FieldBase<T>(), "doReplace"_a=false);
    clsSchema.def(("_addField_"+suffix).c_str(),
                  (Key<T> (Schema::*)(std::string const &,
                                      std::string const &,
                                      FieldBase<T> const &,
                                      bool)) &Schema::addField,
                  "name"_a, "doc"_a, "base"_a, "doReplace"_a=false);
    clsSchema.def(("_find_"+suffix).c_str(),
                  (SchemaItem<T> (Schema::*)(std::string const &) const) &Schema::find);
    clsSchema.def(("_find_"+suffix).c_str(),
                  (SchemaItem<T> (Schema::*)(Key<T> const &) const) &Schema::find);
    clsSchema.def("contains", (int (Schema::*)(SchemaItem<T> const &, int) const) &Schema::contains,
                  "item"_a, "flags"_a=Schema::ComparisonFlags::EQUAL_KEYS);
};
template <>
void declareSchemaOverloads<std::string>(py::class_<Schema> & clsSchema, std::string const & suffix) {
    clsSchema.def(("_addField_"+suffix).c_str(),
                  (Key<std::string> (Schema::*)(Field<std::string> const &, bool)) &Schema::addField,
                  "field"_a, "doReplace"_a=false);
    clsSchema.def(("_addField_"+suffix).c_str(),
                  (Key<std::string> (Schema::*)(std::string const &,
                                    std::string const &,
                                    std::string const &,
                                    FieldBase<std::string> const &,
                                    bool)) &Schema::addField,
                  "name"_a, "doc"_a, "units"_a="", "base"_a=FieldBase<std::string>(1), "doReplace"_a=false);
    clsSchema.def(("_addField_"+suffix).c_str(),
                  (Key<std::string> (Schema::*)(std::string const &,
                                      std::string const &,
                                      FieldBase<std::string> const &,
                                      bool)) &Schema::addField,
                  "name"_a, "doc"_a, "base"_a, "doReplace"_a=false);
    clsSchema.def(("_find_"+suffix).c_str(),
                   (SchemaItem<std::string> (Schema::*)(std::string const &) const) &Schema::find);
    clsSchema.def(("_find_"+suffix).c_str(),
                  (SchemaItem<std::string> (Schema::*)(Key<std::string> const &) const) &Schema::find);
    clsSchema.def("contains",
                  (int (Schema::*)(SchemaItem<std::string> const &, int) const) &Schema::contains,
                  "item"_a, "flags"_a=Schema::ComparisonFlags::EQUAL_KEYS);
};

template <typename T>
void declareSubSchemaOverloads(py::class_<SubSchema> & clsSubSchema, std::string const & suffix) {
    clsSubSchema.def(("_find_"+suffix).c_str(),
                     (SchemaItem<T> (SubSchema::*)(std::string const &) const) &SubSchema::find);
    clsSubSchema.def(("_asKey_"+suffix).c_str(), [](SubSchema & self)->Key<T> {
        return self;
    });
    clsSubSchema.def(("_asField_"+suffix).c_str(), [](SubSchema & self)->Field<T> {
        return self;
    });
};

PYBIND11_PLUGIN(_schema) {
    py::module mod("_schema", "Python wrapper for afw _schema library");

    /* Module level */
    py::class_<Schema> clsSchema(mod, "Schema");
    py::class_<SubSchema> clsSubSchema(mod, "SubSchema");

    /* Member types and enums */
    py::enum_<Schema::ComparisonFlags>(clsSchema, "ComparisonFlags")
        .value("EQUAL_KEYS", Schema::ComparisonFlags::EQUAL_KEYS)
        .value("EQUAL_NAMES", Schema::ComparisonFlags::EQUAL_NAMES)
        .value("EQUAL_DOCS", Schema::ComparisonFlags::EQUAL_DOCS)
        .value("EQUAL_UNITS", Schema::ComparisonFlags::EQUAL_UNITS)
        .value("EQUAL_FIELDS", Schema::ComparisonFlags::EQUAL_FIELDS)
        .value("EQUAL_ALIASES", Schema::ComparisonFlags::EQUAL_ALIASES)
        .value("IDENTICAL", Schema::ComparisonFlags::IDENTICAL)
        .export_values();

    /* Constructors */
    clsSchema.def(py::init<>());
    clsSchema.def(py::init<Schema const &>());

    /* Operators */
    clsSchema.def("__getitem__", [](Schema & self, std::string const & name) {
        return self[name];
    });
    clsSchema.def("__eq__",
                 [](Schema const & self, Schema const & other) { return self == other; },
                 py::is_operator());
    clsSchema.def("__ne__",
                 [](Schema const & self, Schema const & other) { return self != other; },
                 py::is_operator());

    /* Members */
    clsSchema.def("getRecordSize", &Schema::getRecordSize);
    clsSchema.def("getFieldCount", &Schema::getFieldCount);
    clsSchema.def("getFlagFieldCount", &Schema::getFlagFieldCount);
    clsSchema.def("getNonFlagFieldCount", &Schema::getNonFlagFieldCount);
    clsSchema.def("getNames", &Schema::getNames, "topOnly"_a=false);
    clsSchema.def("getAliasMap", &Schema::getAliasMap);
    clsSchema.def("setAliasMap", &Schema::setAliasMap, "aliases"_a);
    clsSchema.def("disconnectAliases", &Schema::disconnectAliases);
    clsSchema.def("_forEach", [](Schema & self, py::object & obj) {
        self.forEach(obj);
    });
    clsSchema.def("compare", &Schema::compare, "other"_a, "flags"_a=Schema::ComparisonFlags::EQUAL_KEYS);
    clsSchema.def("contains", (int (Schema::*)(Schema const &, int) const) &Schema::contains,
                  "other"_a, "flags"_a=Schema::ComparisonFlags::EQUAL_KEYS);

    clsSchema.def_static("readFits",
                         (Schema (*)(std::string const &, int)) &Schema::readFits,
                         "filename"_a, "hdu"_a=0);
    // clsSchema.def_static("readFits",
    //                      (Schema (*)(fits::MemFileManager &, int)) &Schema::readFits,
    //                      "manager"_a, "hdu"_a=0);
    // clsSchema.def_static("readFits",
    //                      (Schema (*)(fits::Fits &)) &Schema::readFits,
    //                      "fitsfile"_a);

    clsSchema.def("join",
                  (std::string (Schema::*)(std::string const &, std::string const &) const) &Schema::join,
                  "a"_a, "b"_a);
    clsSchema.def("join",
                  (std::string (Schema::*)(std::string const &,
                                           std::string const &,
                                           std::string const &) const) &Schema::join,
                  "a"_a, "b"_a, "c"_a);
    clsSchema.def("join",
                  (std::string (Schema::*)(std::string const &,
                                           std::string const &,
                                           std::string const &,
                                           std::string const &) const) &Schema::join,
                  "a"_a, "b"_a, "c"_a, "d"_a);

    declareSchemaOverloads<std::uint16_t>(clsSchema, "U");
    declareSchemaOverloads<std::int32_t>(clsSchema, "I");
    declareSchemaOverloads<std::int64_t>(clsSchema, "L");
    declareSchemaOverloads<float>(clsSchema, "F");
    declareSchemaOverloads<double>(clsSchema, "D");
    declareSchemaOverloads<std::string>(clsSchema, "String");
    declareSchemaOverloads<lsst::afw::geom::Angle>(clsSchema, "Angle");
    declareSchemaOverloads<lsst::afw::table::Flag>(clsSchema, "Flag");
    declareSchemaOverloads<lsst::afw::table::Array<std::uint16_t>>(clsSchema, "ArrayU");
    declareSchemaOverloads<lsst::afw::table::Array<int>>(clsSchema, "ArrayI");
    declareSchemaOverloads<lsst::afw::table::Array<float>>(clsSchema, "ArrayF");
    declareSchemaOverloads<lsst::afw::table::Array<double>>(clsSchema, "ArrayD");
    
    clsSubSchema.def("getNames", &SubSchema::getNames, "topOnly"_a=false);
    clsSubSchema.def("getPrefix", &SubSchema::getPrefix);
    
    declareSubSchemaOverloads<std::uint16_t>(clsSubSchema, "U");
    declareSubSchemaOverloads<std::int32_t>(clsSubSchema, "I");
    declareSubSchemaOverloads<std::int64_t>(clsSubSchema, "L");
    declareSubSchemaOverloads<float>(clsSubSchema, "F");
    declareSubSchemaOverloads<double>(clsSubSchema, "D");
    declareSubSchemaOverloads<std::string>(clsSubSchema, "String");
    declareSubSchemaOverloads<lsst::afw::geom::Angle>(clsSubSchema, "Angle");
    declareSubSchemaOverloads<lsst::afw::table::Flag>(clsSubSchema, "Flag");
    declareSubSchemaOverloads<lsst::afw::table::Array<std::uint16_t>>(clsSubSchema, "ArrayU");
    declareSubSchemaOverloads<lsst::afw::table::Array<int>>(clsSubSchema, "ArrayI");
    declareSubSchemaOverloads<lsst::afw::table::Array<float>>(clsSubSchema, "ArrayF");
    declareSubSchemaOverloads<lsst::afw::table::Array<double>>(clsSubSchema, "ArrayD");

    return mod.ptr();
}

}}}  // namespace lsst::afw::table
