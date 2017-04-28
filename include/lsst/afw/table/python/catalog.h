/*
 * LSST Data Management System
 * Copyright 2016  AURA/LSST.
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
#ifndef AFW_TABLE_PYTHON_CATALOG_H_INCLUDED
#define AFW_TABLE_PYTHON_CATALOG_H_INCLUDED

#include "pybind11/pybind11.h"

#include "lsst/utils/python.h"
#include "lsst/afw/table/BaseColumnView.h"
#include "lsst/afw/table/Catalog.h"

namespace lsst {
namespace afw {
namespace table {
namespace python {

template <typename Record>
using PyCatalog = pybind11::class_<CatalogT<Record>, std::shared_ptr<CatalogT<Record>>>;

/**
Declare field-type-specific overloaded catalog member functions for one field type

@tparam T  Field type.
@tparam Record  Record type, e.g. BaseRecord or SimpleRecord.

@param[in] cls  Catalog pybind11 class.
*/

template <typename T, typename Record>
void declareCatalogOverloads(PyCatalog<Record> &cls) {
    namespace py = pybind11;
    using namespace pybind11::literals;

    typedef CatalogT<Record> Catalog;
    typedef typename Field<T>::Value Value;

    cls.def("isSorted", (bool (Catalog::*)(Key<T> const &) const) & Catalog::isSorted);
    cls.def("sort", (void (Catalog::*)(Key<T> const &)) & Catalog::sort);
    cls.def("find", [](Catalog &self, Value const &value, Key<T> const &key) -> std::shared_ptr<Record> {
        auto iter = self.find(value, key);
        if (iter == self.end()) {
            return nullptr;
        };
        return iter;
    });
    cls.def("upper_bound", [](Catalog &self, Value const &value, Key<T> const &key) -> std::ptrdiff_t {
        return self.upper_bound(value, key) - self.begin();
    });
    cls.def("lower_bound", [](Catalog &self, Value const &value, Key<T> const &key) -> std::ptrdiff_t {
        return self.lower_bound(value, key) - self.begin();
    });
    cls.def("equal_range", [](Catalog &self, Value const &value, Key<T> const &key) {
        auto p = self.equal_range(value, key);
        return py::slice(p.first - self.begin(), p.second - self.begin(), 1);
    });
    cls.def("between", [](Catalog &self, Value const &lower, Value const &upper, Key<T> const &key) {
        std::ptrdiff_t a = self.lower_bound(lower, key) - self.begin();
        std::ptrdiff_t b = self.upper_bound(upper, key) - self.begin();
        return py::slice(a, b, 1);
    });
}

/**
Wrap an instantiation of lsst::afw::table::CatalogT<Record>.

In addition to calling this method you must call addCatalogMethods on the
class object in Python.

@tparam Record  Record type, e.g. BaseRecord or SimpleRecord.

@param[in] mod    Module object class will be added to.
@param[in] name   Name prefix of the record type, e.g. "Base" or "Simple".
@param[in] isBase Whether this instantiation is only being used as a base class (used to set the class name).
*/
template <typename Record>
PyCatalog<Record> declareCatalog(pybind11::module &mod, std::string const &name, bool isBase = false) {
    namespace py = pybind11;
    using namespace pybind11::literals;

    using Catalog = CatalogT<Record>;
    using Table = typename Record::Table;

    std::string fullName;
    if (isBase) {
        fullName = "_" + name + "CatalogBase";
    } else {
        fullName = name + "Catalog";
    }

    // We need py::dynamic_attr() below to support our Python-side caching of the associated ColumnView.
    PyCatalog<Record> cls(mod, fullName.c_str(), py::dynamic_attr());

    /* Constructors */
    cls.def(py::init<Schema const &>(), "schema"_a);
    cls.def(py::init<std::shared_ptr<Table> const &>(), "table"_a);
    cls.def(py::init<Catalog const &>(), "other"_a);

    /* Static Methods */
    cls.def_static("readFits", (Catalog(*)(std::string const &, int, int)) & Catalog::readFits, "filename"_a,
                   "hdu"_a = INT_MIN, "flags"_a = 0);
    cls.def_static("readFits", (Catalog(*)(fits::MemFileManager &, int, int)) & Catalog::readFits,
                   "manager"_a, "hdu"_a = INT_MIN, "flags"_a = 0);
    // readFits taking Fits objects not wrapped, because Fits objects are not wrapped.

    /* Methods */
    cls.def("getTable", &Catalog::getTable);
    cls.def_property_readonly("table", &Catalog::getTable);
    cls.def("getSchema", &Catalog::getSchema);
    cls.def_property_readonly("schema", &Catalog::getSchema);
    cls.def("capacity", &Catalog::capacity);
    cls.def("__len__", &Catalog::size);

    // Use private names for the following so the public Python method
    // can manage the _column cache
    cls.def("_getColumnView", &Catalog::getColumnView);
    cls.def("_addNew", &Catalog::addNew);
    cls.def("_extend", [](Catalog &self, Catalog const &other, bool deep) {
        self.insert(self.end(), other.begin(), other.end(), deep);
    });
    cls.def("_extend", [](Catalog &self, Catalog const &other, SchemaMapper const &mapper) {
        self.insert(mapper, self.end(), other.begin(), other.end());
    });
    cls.def("_append", [](Catalog &self, std::shared_ptr<Record> const &rec) { self.push_back(rec); });
    cls.def("_delitem_", [](Catalog &self, std::ptrdiff_t i) {
        self.erase(self.begin() + utils::python::cppIndex(self.size(), i));
    });
    cls.def("_delslice_", [](Catalog &self, py::slice const &s) {
        Py_ssize_t start = 0, stop = 0, step = 0, length = 0;
        if (PySlice_GetIndicesEx(
// The interface to this function changed in Python 3.2
#if PY_MAJOR_VERSION < 3 || (PY_MINOR_VERSION == 3 && PY_MINOR_VERSION < 2)
                    (PySliceObject *)
#endif
                            s.ptr(),
                    self.size(), &start, &stop, &step, &length) != 0) {
            throw py::error_already_set();
        }
        if (step != 1) {
            throw py::index_error("Slice step must not exactly 1");
        }
        self.erase(self.begin() + start, self.begin() + stop);
    });
    cls.def("_clear", &Catalog::clear);

    cls.def("set", &Catalog::set);
    cls.def("_getitem_",
            [](Catalog &self, int i) { return self.get(utils::python::cppIndex(self.size(), i)); });
    cls.def("isContiguous", &Catalog::isContiguous);
    cls.def("writeFits",
            (void (Catalog::*)(std::string const &, std::string const &, int) const) & Catalog::writeFits,
            "filename"_a, "mode"_a = "w", "flags"_a = 0);
    cls.def("writeFits",
            (void (Catalog::*)(fits::MemFileManager &, std::string const &, int) const) & Catalog::writeFits,
            "manager"_a, "mode"_a = "w", "flags"_a = 0);
    cls.def("reserve", &Catalog::reserve);
    cls.def("subset", (Catalog (Catalog::*)(ndarray::Array<bool const, 1> const &) const) & Catalog::subset);
    cls.def("subset",
            (Catalog (Catalog::*)(std::ptrdiff_t, std::ptrdiff_t, std::ptrdiff_t) const) & Catalog::subset);

    declareCatalogOverloads<std::int32_t>(cls);
    declareCatalogOverloads<std::int64_t>(cls);
    declareCatalogOverloads<float>(cls);
    declareCatalogOverloads<double>(cls);
    declareCatalogOverloads<lsst::afw::geom::Angle>(cls);

    return cls;
};
}
}
}
}  // lsst::afw::table::python

#endif  // !LSST_AFW_TABLE_PYTHON_CATALOG_H_INCLUDED
