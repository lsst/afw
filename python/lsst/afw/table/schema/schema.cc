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
#include "pybind11/stl.h"

#include <sstream>

#include "ndarray/pybind11.h"

#include "lsst/utils/python.h"

#include "lsst/afw/fits.h"
#include "lsst/afw/table/Schema.h"
#include "lsst/afw/table/BaseRecord.h"
#include "lsst/afw/table/SchemaMapper.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst {
namespace afw {
namespace table {

namespace {

using PySchema = py::class_<Schema>;

using PySubSchema = py::class_<SubSchema>;

template <typename T>
using PyFieldBase = py::class_<FieldBase<T>>;

template <typename T>
using PyKeyBase = py::class_<KeyBase<T>>;

template <typename T>
using PyField = py::class_<Field<T>, FieldBase<T>>;

template <typename T>
using PyKey = py::class_<Key<T>, KeyBase<T>, FieldBase<T>>;

template <typename T>
using PySchemaItem = py::class_<SchemaItem<T>>;

// Specializations for FieldBase

template <typename T>
void declareFieldBaseSpecializations(PyFieldBase<T> &cls) {
    cls.def(py::init<>());
}

template <typename T>
void declareFieldBaseSpecializations(PyFieldBase<Array<T>> &cls) {
    cls.def(py::init<int>(), "size"_a = 0);
    cls.def("getSize", &FieldBase<Array<T>>::getSize);
    cls.def("isVariableLength", &FieldBase<Array<T>>::isVariableLength);
}

void declareFieldBaseSpecializations(PyFieldBase<std::string> &cls) {
    cls.def(py::init<int>(), "size"_a = -1);
    cls.def("getSize", &FieldBase<std::string>::getSize);
}

// Specializations for KeyBase

template <typename T>
void declareKeyBaseSpecializations(PyKeyBase<T> &) {}

template <typename T>
void declareKeyBaseSpecializations(PyKeyBase<Array<T>> &cls) {
    cls.def("__getitem__", [](Key<Array<T>> const &self, py::object const &index) -> py::object {
        if (py::isinstance<py::slice>(index)) {
            py::slice slice(index);
            py::size_t start = 0, stop = 0, step = 0, length = 0;
            bool valid = slice.compute(self.getSize(), &start, &stop, &step, &length);
            if (!valid) throw py::error_already_set();
            if (step != 1) {
                throw LSST_EXCEPT(pex::exceptions::InvalidParameterError,
                                  "Step for array Key indexing must be 1.");
            }
            return py::cast(self.slice(start, stop));
        } else {
            return py::cast(self[py::cast<int>(index)]);
        }
    });
    cls.def("slice", &KeyBase<Array<T>>::slice);
}

// Specializations for Key

template <typename T>
void declareKeyAccessors(PyKey<T> &cls) {
    cls.def("get", [](Key<T> const &self, BaseRecord &record) { return record.get(self); });
    cls.def("set", [](Key<T> const &self, BaseRecord &record, typename Key<T>::Value const &value) {
        record.set(self, value);
    });
}

template <typename U>
void declareKeyAccessors(PyKey<Array<U>> &cls) {
    auto getter = [](Key<Array<U>> const &self, BaseRecord &record) -> ndarray::Array<U, 1, 1> {
        return record[self];
    };
    auto setter = [](Key<Array<U>> const &self, BaseRecord &record, py::object const &value) {
        if (self.getSize() == 0) {
            // Variable-length array field: do a shallow copy, which requires a non-const
            // contiguous array.
            record.set(self, py::cast<ndarray::Array<U, 1, 1>>(value));
        } else {
            // Fixed-length array field: do a deep copy, which can work with a const
            // noncontiguous array.  But we need to check the size first, since the
            // penalty for getting that wrong is assert->abort.
            auto v = py::cast<ndarray::Array<U const, 1, 0>>(value);
            ndarray::ArrayRef<U, 1, 1> ref = record[self];
            if (v.size() != ref.size()) {
                throw LSST_EXCEPT(
                        pex::exceptions::LengthError,
                        (boost::format("Array sizes do not agree: %s != %s") % v.size() % ref.size()).str());
            }
            ref = v;
        }
        return;
    };
    cls.def("get", getter);
    cls.def("set", setter);
}

template <typename T>
void declareKeySpecializations(PyKey<T> &cls) {
    declareKeyAccessors(cls);
    cls.def_property_readonly("subfields", [](py::object const &) { return py::none(); });
    cls.def_property_readonly("subkeys", [](py::object const &) { return py::none(); });
}

void declareKeySpecializations(PyKey<Flag> &cls) {
    declareKeyAccessors(cls);
    cls.def_property_readonly("subfields", [](py::object const &) { return py::none(); });
    cls.def_property_readonly("subkeys", [](py::object const &) { return py::none(); });
    cls.def("getBit", &Key<Flag>::getBit);
}

template <typename U>
void declareKeySpecializations(PyKey<Array<U>> &cls) {
    declareKeyAccessors(cls);
    cls.def_property_readonly("subfields", [](Key<Array<U>> const &self) -> py::object {
        py::list result;
        for (int i = 0; i < self.getSize(); ++i) {
            result.append(py::cast(i));
        }
        return py::tuple(result);
    });
    cls.def_property_readonly("subkeys", [](Key<Array<U>> const &self) -> py::object {
        py::list result;
        for (int i = 0; i < self.getSize(); ++i) {
            result.append(py::cast(self[i]));
        }
        return py::tuple(result);
    });
}

// Wrap all helper classes (FieldBase, KeyBase, Key, Field, SchemaItem) declarefor a Schema field type.
template <typename T>
void declareSchemaType(py::module &mod) {
    std::string suffix = FieldBase<T>::getTypeString();
    py::str pySuffix(suffix);

    py::object astropyUnit = py::module::import("astropy.units").attr("Unit");

    // FieldBase
    PyFieldBase<T> clsFieldBase(mod, ("FieldBase" + suffix).c_str());
    clsFieldBase.def_static("getTypeString", &FieldBase<T>::getTypeString);
    declareFieldBaseSpecializations(clsFieldBase);

    // KeyBase
    PyKeyBase<T> clsKeyBase(mod, ("KeyBase" + suffix).c_str());
    clsKeyBase.def_readonly_static("HAS_NAMED_SUBFIELDS", &KeyBase<T>::HAS_NAMED_SUBFIELDS);
    declareKeyBaseSpecializations(clsKeyBase);

    // Field
    PyField<T> clsField(mod, ("Field" + suffix).c_str());
    mod.attr("_Field")[pySuffix] = clsField;
    clsField.def(py::init(
                 [astropyUnit](  // capture by value to refcount in Python instead of dangle in C++
                         std::string const &name, std::string const &doc,
                         py::str const &units, py::object const &size, py::str const &parse_strict) {
                     astropyUnit(units, "parse_strict"_a = parse_strict);
                     std::string u = py::cast<std::string>(units);
                     if (size == py::none()) {
                         return new Field<T>(name, doc, u);
                     } else {
                         int s = py::cast<int>(size);
                         return new Field<T>(name, doc, u, s);
                     }
                 }),
                 "name"_a, "doc"_a = "", "units"_a = "", "size"_a = py::none(), "parse_strict"_a = "raise");
    clsField.def("_addTo", [](Field<T> const &self, Schema &schema, bool doReplace) -> Key<T> {
        return schema.addField(self, doReplace);
    });
    clsField.def("getName", &Field<T>::getName);
    clsField.def("getDoc", &Field<T>::getDoc);
    clsField.def("getUnits", &Field<T>::getUnits);
    clsField.def("copyRenamed", &Field<T>::copyRenamed);
    utils::python::addOutputOp(clsField, "__str__");
    utils::python::addOutputOp(clsField, "__repr__");

    // Key
    PyKey<T> clsKey(mod, ("Key" + suffix).c_str());
    mod.attr("_Key")[pySuffix] = clsKey;
    clsKey.def(py::init<>());
    clsKey.def("__eq__", [](Key<T> const &self, Key<T> const &other) -> bool { return self == other; },
               py::is_operator());
    clsKey.def("__ne__", [](Key<T> const &self, Key<T> const &other) -> bool { return self != other; },
               py::is_operator());
    clsKey.def("isValid", &Key<T>::isValid);
    clsKey.def("getOffset", &Key<T>::getOffset);
    utils::python::addOutputOp(clsKey, "__str__");
    utils::python::addOutputOp(clsKey, "__repr__");
    // The Key methods below actually wrap templated methods on Schema and
    // SchemaMapper.  Rather than doing many-type overload resolution by
    // wrapping those methods directly, we use the visitor pattern by having
    // the wrappers for those methods delegate back to these non-templated
    // methods on the templated Key classes.
    clsKey.def("_findIn", [](Key<T> const &self, Schema const &schema) { return schema.find(self); });
    clsKey.def("_addMappingTo", [](Key<T> const &self, SchemaMapper &mapper, Field<T> const &field,
                                   bool doReplace) { return mapper.addMapping(self, field, doReplace); });
    clsKey.def("_addMappingTo", [](Key<T> const &self, SchemaMapper &mapper, std::string const &name,
                                   bool doReplace) { return mapper.addMapping(self, name, doReplace); });
    clsKey.def("_addMappingTo", [](Key<T> const &self, SchemaMapper &mapper, py::object const &,
                                   bool doReplace) { return mapper.addMapping(self, doReplace); });
    declareKeySpecializations(clsKey);

    // SchemaItem
    PySchemaItem<T> clsSchemaItem(mod, ("SchemaItem" + suffix).c_str());
    mod.attr("_SchemaItem")[pySuffix] = clsSchemaItem;
    clsSchemaItem.def_readonly("key", &SchemaItem<T>::key);
    clsSchemaItem.def_readonly("field", &SchemaItem<T>::field);
    clsSchemaItem.def("getKey", [](SchemaItem<T> const &self) { return self.key; });
    clsSchemaItem.def("getField", [](SchemaItem<T> const &self) { return self.field; });
    clsSchemaItem.def("__getitem__", [](py::object const &self, int index) -> py::object {
        if (index == 0) {
            return self.attr("key");
        } else if (index == 1) {
            return self.attr("field");
        }
        // Have to raise IndexError not some LSST exception to get the
        // right behavior when unpacking.
        throw py::index_error("Index to SchemaItem must be 0 or 1.");
    });
    clsSchemaItem.def("__len__", [](py::object const &self) -> int { return 2; });
    clsSchemaItem.def("__str__", [](py::object const &self) -> py::str { return py::str(py::tuple(self)); });
    clsSchemaItem.def("__repr__", [](py::object const &self) -> py::str {
        return py::str("SchemaItem(key={0.key}, field={0.field})").format(self);
    });
}

// Helper class for Schema::find(name, func) that converts the result to Python.
// In C++14, this should be converted to a universal lambda.
struct MakePythonSchemaItem {
    template <typename T>
    void operator()(SchemaItem<T> const &item) {
        result = py::cast(item);
    }

    py::object result;
};

void declareSchema(py::module &mod) {
    py::module::import("lsst.afw.table.aliasMap");

    PySchema cls(mod, "Schema");
    // wrap ComparisonFlags values as ints since we use them as bitflags,
    // not true enums
    cls.attr("EQUAL_KEYS") = py::cast(int(Schema::EQUAL_KEYS));
    cls.attr("EQUAL_NAMES") = py::cast(int(Schema::EQUAL_NAMES));
    cls.attr("EQUAL_DOCS") = py::cast(int(Schema::EQUAL_DOCS));
    cls.attr("EQUAL_UNITS") = py::cast(int(Schema::EQUAL_UNITS));
    cls.attr("EQUAL_FIELDS") = py::cast(int(Schema::EQUAL_FIELDS));
    cls.attr("EQUAL_ALIASES") = py::cast(int(Schema::EQUAL_ALIASES));
    cls.attr("IDENTICAL") = py::cast(int(Schema::IDENTICAL));

    cls.attr("VERSION") = py::cast(int(Schema::VERSION));

    cls.def(py::init<>());
    cls.def(py::init<Schema const &>());
    cls.def("__getitem__", [](Schema &self, std::string const &name) { return self[name]; });
    cls.def("__eq__", [](Schema const &self, Schema const &other) { return self == other; },
            py::is_operator());
    cls.def("__ne__", [](Schema const &self, Schema const &other) { return self != other; },
            py::is_operator());
    cls.def("getCitizen", &Schema::getCitizen, py::return_value_policy::reference_internal);
    cls.def("getRecordSize", &Schema::getRecordSize);
    cls.def("getFieldCount", &Schema::getFieldCount);
    cls.def("getFlagFieldCount", &Schema::getFlagFieldCount);
    cls.def("getNonFlagFieldCount", &Schema::getNonFlagFieldCount);
    cls.def("find", [](py::object const &self, py::object const &key) -> py::object {
        try {
            if (py::isinstance<py::str>(key) || py::isinstance<py::bytes>(key)) {
                Schema const &s = py::cast<Schema const &>(self);
                std::string name = py::cast<std::string>(key);
                MakePythonSchemaItem func;
                s.findAndApply(name, func);
                return func.result;
            }
            return key.attr("_findIn")(self);
        } catch (pex::exceptions::NotFoundError &err) {
            // Avoid API change by re-throwing as KeyError.
            PyErr_SetString(PyExc_KeyError, err.what());
            throw py::error_already_set();
        }
    });
    cls.def("getNames", &Schema::getNames, "topOnly"_a = false);
    cls.def("getAliasMap", &Schema::getAliasMap);
    cls.def("setAliasMap", &Schema::setAliasMap, "aliases"_a);
    cls.def("disconnectAliases", &Schema::disconnectAliases);
    cls.def("forEach", [](Schema &self, py::object &obj) { self.forEach(obj); });
    cls.def("compare", &Schema::compare, "other"_a, "flags"_a = int(Schema::EQUAL_KEYS));
    cls.def("contains", (int (Schema::*)(Schema const &, int) const) & Schema::contains, "other"_a,
            "flags"_a = int(Schema::EQUAL_KEYS));
    cls.def("__contains__", [](py::object const &self, py::object const &key) {
        try {
            self.attr("find")(key);
        } catch (py::error_already_set &err) {
            err.clear();
            return false;
        }
        return true;
    });
    cls.def_static("readFits", (Schema(*)(std::string const &, int)) & Schema::readFits, "filename"_a,
                   "hdu"_a = fits::DEFAULT_HDU);
    cls.def_static("readFits", (Schema(*)(fits::MemFileManager &, int)) & Schema::readFits, "manager"_a,
                   "hdu"_a = fits::DEFAULT_HDU);

    cls.def("join", (std::string (Schema::*)(std::string const &, std::string const &) const) & Schema::join,
            "a"_a, "b"_a);
    cls.def("join",
            (std::string (Schema::*)(std::string const &, std::string const &, std::string const &) const) &
                    Schema::join,
            "a"_a, "b"_a, "c"_a);
    cls.def("join", (std::string (Schema::*)(std::string const &, std::string const &, std::string const &,
                                             std::string const &) const) &
                            Schema::join,
            "a"_a, "b"_a, "c"_a, "d"_a);
    utils::python::addOutputOp(cls, "__str__");
    utils::python::addOutputOp(cls, "__repr__");
}

void declareSubSchema(py::module &mod) {
    PySubSchema cls(mod, "SubSchema");
    cls.def("getNames", &SubSchema::getNames, "topOnly"_a = false);
    cls.def("getPrefix", &SubSchema::getPrefix);
    cls.def("asKey", [](SubSchema const &self) -> py::object {
        MakePythonSchemaItem func;
        self.apply(func);
        return func.result.attr("key");
    });
    cls.def("asField", [](SubSchema const &self) -> py::object {
        MakePythonSchemaItem func;
        self.apply(func);
        return func.result.attr("field");
    });
    cls.def("find", [](SubSchema const &self, std::string const &name) -> py::object {
        MakePythonSchemaItem func;
        self.findAndApply(name, func);
        return func.result;
    });
    cls.def("__getitem__", [](SubSchema &self, std::string const &name) { return self[name]; });
}

PYBIND11_PLUGIN(schema) {
    py::module mod("schema");

    // We'll add instantiations of Field, Key, and SchemaItem to these private
    // dicts, and then in schemaContinued.py we'll add them to a TemplateMeta
    // ABC.
    mod.attr("_Field") = py::dict();
    mod.attr("_Key") = py::dict();
    mod.attr("_SchemaItem") = py::dict();

    declareSchemaType<std::uint8_t>(mod);
    declareSchemaType<std::uint16_t>(mod);
    declareSchemaType<std::int32_t>(mod);
    declareSchemaType<std::int64_t>(mod);
    declareSchemaType<float>(mod);
    declareSchemaType<double>(mod);
    declareSchemaType<std::string>(mod);
    declareSchemaType<lsst::geom::Angle>(mod);
    declareSchemaType<Array<std::uint8_t>>(mod);
    declareSchemaType<Array<std::uint16_t>>(mod);
    declareSchemaType<Array<int>>(mod);
    declareSchemaType<Array<float>>(mod);
    declareSchemaType<Array<double>>(mod);
    declareSchemaType<Flag>(mod);

    declareSchema(mod);
    declareSubSchema(mod);

    return mod.ptr();
}
}
}
}
}  // namespace lsst::afw::table::<anonymous>
