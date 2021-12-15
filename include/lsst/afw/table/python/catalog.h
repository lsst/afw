/*
 * This file is part of afw.
 *
 * Developed for the LSST Data Management System.
 * This product includes software developed by the LSST Project
 * (https://www.lsst.org).
 * See the COPYRIGHT file at the top-level directory of this distribution
 * for details of code ownership.
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
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */
#ifndef AFW_TABLE_PYTHON_CATALOG_H_INCLUDED
#define AFW_TABLE_PYTHON_CATALOG_H_INCLUDED

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/numpy.h"
#include "ndarray/pybind11.h"

#include "lsst/utils/python.h"
#include "lsst/afw/table/BaseColumnView.h"
#include "lsst/afw/table/Catalog.h"
#include "lsst/afw/table/python/columnView.h"

namespace lsst {
namespace afw {
namespace table {
namespace python {

template <typename Record>
using PyCatalog = pybind11::class_<CatalogT<Record>, std::shared_ptr<CatalogT<Record>>>;

/*
 * A helper class for returning numpy arrays of Catalog columns.
 *
 * This builds on ColumnViewGetter (in python/columnView.h) by providing copy
 * (as opposed to view) arrays when the catalog is noncontiguous.
 */
template <typename Record>
class ColumnGetter {
public:

    explicit ColumnGetter(CatalogT<Record> & catalog, bool force_copy=false, bool ensure_writeable=false) :
        _catalog(catalog),
        _columns(_catalog.getColumnView()),
        _force_copy(force_copy),
        _ensure_writeable(ensure_writeable)
    {}

    template <typename T>
    pybind11::array operator()(Key<T> const & key) const {
        if (_columns && !_force_copy) {
            return ColumnViewGetter(_columns.value())(key);
        } else {
            return _maybe_read_only(_catalog.copyColumn(key));
        }
    }

    template <typename T>
    pybind11::array operator()(Key<Array<T>> const & key) const {
        if (key.isVariableLength()) {
            return _make_object_array(key);
        } else if (_columns && !_force_copy) {
            return ColumnViewGetter(_columns.value())(key);
        } else {
            return _maybe_read_only(pybind11::cast(array));
        }
    }

    pybind11::array operator()(Key<std::string> const & key) const {
        if (key.isVariableLength()) {
            return _make_object_array(key);
        } else if (_columns && !_force_copy) {
            return ColumnViewGetter(_columns.value())(key);
        } else {
            return _maybe_read_only(
                ColumnViewGetter::make_str_array(
                    [this, key](std::size_t n) -> pybind11::str {
                        return pybind11::str(this->_catalog[n].getElement(key), key.getElementCount());
                    },
                    _catalog.size(),
                    key.getElementCount()
                )
            );
        }
    }

    pybind11::array operator()(Key<Angle> const & key) const {
        if (_columns && !_force_copy) {
            return ColumnViewGetter(_columns.value())(key);
        } else {
            ndarray::Array<Angle, 1, 1> angles = _catalog.copyColumn(key);
            ndarray::Array<double, 1, 1> radians = ndarray::allocate(angles.getShape());
            std::transform(
                angles.begin(),
                angles.end(),
                radians.begin(),
                [](Angle const & angle) { return angle.asRadians(); }
            );
            return _maybe_read_only(radians);
        }
    }

    pybind11::array operator()(Key<Flag> const & key) const {
        if (_columns && !_force_copy) {
            return ColumnViewGetter(_columns.value())(key);
        } else {
            return _maybe_read_only(array);
        }
    }

    template <typename T>
    pybind11::array operator()(SchemaItem<T> const & item) const {
        return this->operator()(item.key);
    }

    pybind11::array operator()(std::string const & name) const {
        return _catalog.getSchema().findAndApply(name, *this);
    }

private:

    pybind11::array _maybe_read_only(pybind11::array array) const {
        if (!_ensure_writeable) {
            array.attr("flags")["WRITEABLE"] = false;
        }
        return array;
    }

    template <typename T, int N, int C>
    pybind11::array _maybe_read_only(ndarray::Array<T, N, C> const & array) const {
        return _maybe_read_only(pybind11::cast(array));
    }

    template <typename T>
    pybind11::object _make_object_element(BaseRecord & record, Key<Array<T>> const & key) const {
        ndarray::Array<T, 1, 1> array = record[key];
        if (_force_copy) {
            array = ndarray::copy(array);
        }
        return pybind11::cast(array);
    }

    template <typename T>
    pybind11::object _make_object_element(BaseRecord & record, Key<std::string> const & key) const {
        return pybind11::str(record.getElement(key), key.getElementCount());
    }

    template <typename T>
    pybind11::array _make_object_array(Key<T> const & key) const {
        pybind11::list result_as_list;;
        auto record_iter = _catalog.begin();
        auto const record_end = _catalog.end();
        for (pybind11::ssize_t n = 0; record_iter != record_end; ++n, ++record_iter) {
            result_as_list[n] = _make_object_element(*record_iter, key);
        }
        return _maybe_read_only(pybind11::array(result_as_list));
    }

    CatalogT<Record> & _catalog;
    std::optional<BaseColumnView> _columns;
    bool _force_copy;
    bool _ensure_writeable;
};


template <typename Record>
void _setFlagColumnToArray(
    CatalogT<Record> & catalog,
    Key<Flag> const & key,
    ndarray::Array<bool const, 1> const & array
) {
    if (array.size() != catalog.size()) {
        throw LSST_EXCEPT(
            pex::exceptions::LengthError,
            (boost::format("Catalog has %d rows, while array has %d elements.")
             % catalog.size() % array.size()).str()
        );
    }
    auto catIter = catalog.begin();
    auto arrayIter = array.begin();
    for (; catIter != catalog.end(); ++catIter, ++arrayIter) {
        catIter->set(key, *arrayIter);
    }
}

template <typename Record>
void _setFlagColumnToScalar(
    CatalogT<Record> & catalog,
    Key<Flag> const & key,
    bool value
) {
    for (auto catIter = catalog.begin(); catIter != catalog.end(); ++catIter) {
        catIter->set(key, value);
    }
}

/*
 * A local helper class for declaring Catalog's methods that are overloaded
 * on column types.
 *
 * _CatalogOverloadHelper is designed to be invoked by
 * TypeList::for_each_nullptr, which calls operator() with a null pointer cast
 * to each of the types in the list.  This implementation of operator() then
 * dispatches to _declare methods for different kinds of operations.  Various
 * overloads of those _declare methods handle the fact that different column
 * types support different kinds of operations; the compiler will always pick
 * the most specific overload, and this lets us provide a generic templated
 * implementation and then overload specific types to do nothing.
 */
template <typename Record>
class _CatalogOverloadHelper {
public:

    _CatalogOverloadHelper(PyCatalog<Record> & cls) : _cls(cls) {}

    template <typename T>
    void operator()(T const * tag) {
        _declare_comparison_overloads(tag);
        _cls.def(
            "_getitem_",
            [](CatalogT<Record> &self, Key<T> const &key) { return ColumnGetter<Record>(self)(key); }
        );
        _cls.def(
            "get_column",
            [](CatalogT<Record> &self, Key<T> const &key, bool force_copy, bool ensure_writeable) {
                return ColumnGetter<Record>(self, force_copy, ensure_writeable)(key);
            },
            pybind11::arg("key"),
            pybind11::arg("force_copy") = false,
            pybind11::arg("ensure_writeable") = false
        );
    }

private:

    template <typename T>
    void _declare_comparison_overloads(T const *) {
        namespace py = pybind11;
        using namespace pybind11::literals;

        using Catalog = CatalogT<Record>;
        using Value = typename Field<T>::Value;

        _cls.def("isSorted", (bool (Catalog::*)(Key<T> const &) const) & Catalog::isSorted);
        _cls.def("sort", (void (Catalog::*)(Key<T> const &)) & Catalog::sort);
        _cls.def("find", [](Catalog &self, Value const &value, Key<T> const &key) -> std::shared_ptr<Record> {
            auto iter = self.find(value, key);
            if (iter == self.end()) {
                return nullptr;
            };
            return iter;
        });
        _cls.def("upper_bound", [](Catalog &self, Value const &value, Key<T> const &key) -> std::ptrdiff_t {
            return self.upper_bound(value, key) - self.begin();
        });
        _cls.def("lower_bound", [](Catalog &self, Value const &value, Key<T> const &key) -> std::ptrdiff_t {
            return self.lower_bound(value, key) - self.begin();
        });
        _cls.def("equal_range", [](Catalog &self, Value const &value, Key<T> const &key) {
            auto p = self.equal_range(value, key);
            return py::slice(p.first - self.begin(), p.second - self.begin(), 1);
        });
        _cls.def("between", [](Catalog &self, Value const &lower, Value const &upper, Key<T> const &key) {
            std::ptrdiff_t a = self.lower_bound(lower, key) - self.begin();
            std::ptrdiff_t b = self.upper_bound(upper, key) - self.begin();
            return py::slice(a, b, 1);
        });
    }

    template <typename T>
    void _declare_comparison_overloads(Array<T> const *) {
        // Array columns cannot be compared, so we do not define comparison
        // overloads for these.
    }

    PyCatalog<Record> & _cls;
};

/**
 * Wrap an instantiation of lsst::afw::table::CatalogT<Record>.
 *
 * In addition to calling this method you must call addCatalogMethods on the
 * class object in Python.
 *
 * @tparam Record  Record type, e.g. BaseRecord or SimpleRecord.
 *
 * @param[in] wrappers Package manager class will be added to.
 * @param[in] name   Name prefix of the record type, e.g. "Base" or "Simple".
 * @param[in] isBase Whether this instantiation is only being used as a base class
 *                   (used to set the class name).
 */
template <typename Record>
PyCatalog<Record> declareCatalog(utils::python::WrapperCollection &wrappers, std::string const &name,
                                 bool isBase = false) {
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

    // We need py::dynamic_attr() in the class definition to support our Python-side caching
    // of the associated ColumnView.
    return wrappers.wrapType(
            PyCatalog<Record>(wrappers.module, fullName.c_str(), py::dynamic_attr()),
            [](auto &mod, auto &cls) {
                /* Constructors */
                cls.def(py::init<Schema const &>(), "schema"_a);
                cls.def(py::init<std::shared_ptr<Table> const &>(), "table"_a);
                cls.def(py::init<Catalog const &>(), "other"_a);

                /* Static Methods */
                cls.def_static("readFits", (Catalog(*)(std::string const &, int, int)) & Catalog::readFits,
                               "filename"_a, "hdu"_a = fits::DEFAULT_HDU, "flags"_a = 0);
                cls.def_static("readFits", (Catalog(*)(fits::MemFileManager &, int, int)) & Catalog::readFits,
                               "manager"_a, "hdu"_a = fits::DEFAULT_HDU, "flags"_a = 0);
                // readFits taking Fits objects not wrapped, because Fits objects are not wrapped.

                /* Methods */
                cls.def("getTable", &Catalog::getTable);
                cls.def_property_readonly("table", &Catalog::getTable);
                cls.def("getSchema", &Catalog::getSchema);
                cls.def_property_readonly("schema", &Catalog::getSchema);
                cls.def("capacity", &Catalog::capacity);
                cls.def("__len__", &Catalog::size);
                cls.def("resize", &Catalog::resize);

                // Trying to get a column view of a noncontiguous catalog is
                // an error in Python for backwards-compatibility reasons
                // (it's safer to change the C++ because it's less likely to
                // have been used outside these wrappers, and if it was, it'd
                // be an obvious compile failure).
                auto py_get_columns = [](Catalog & self) {
                    auto columns = self.getColumnView();
                    if (!columns) {
                        throw LSST_EXCEPT(
                            pex::exceptions::RuntimeError,
                            "Record data is not contiguous in memory."
                        );
                    }
                    return columns.value();
                };
                cls.def("getColumnView", py_get_columns);
                cls.def_property_readonly("columns", py_get_columns);
                cls.def("addNew", &Catalog::addNew);
                cls.def("_extend", [](Catalog &self, Catalog const &other, bool deep) {
                    self.insert(self.end(), other.begin(), other.end(), deep);
                });
                cls.def("_extend", [](Catalog &self, Catalog const &other, SchemaMapper const &mapper) {
                    self.insert(mapper, self.end(), other.begin(), other.end());
                });
                cls.def("append",
                        [](Catalog &self, std::shared_ptr<Record> const &rec) { self.push_back(rec); });
                cls.def("__delitem__", [](Catalog &self, std::ptrdiff_t i) {
                    self.erase(self.begin() + utils::python::cppIndex(self.size(), i));
                });
                cls.def("__delitem__", [](Catalog &self, py::slice const &s) {
                    Py_ssize_t start = 0, stop = 0, step = 0, length = 0;
                    if (PySlice_GetIndicesEx(s.ptr(), self.size(), &start, &stop, &step, &length) != 0) {
                        throw py::error_already_set();
                    }
                    if (step != 1) {
                        throw py::index_error("Slice step must be exactly 1");
                    }
                    self.erase(self.begin() + start, self.begin() + stop);
                });
                cls.def("clear", &Catalog::clear);

                cls.def("set", &Catalog::set);
                cls.def("isContiguous", &Catalog::isContiguous);
                cls.def("writeFits",
                        (void (Catalog::*)(std::string const &, std::string const &, int) const) &
                                Catalog::writeFits,
                        "filename"_a, "mode"_a = "w", "flags"_a = 0);
                cls.def("writeFits",
                        (void (Catalog::*)(fits::MemFileManager &, std::string const &, int) const) &
                                Catalog::writeFits,
                        "manager"_a, "mode"_a = "w", "flags"_a = 0);
                cls.def("reserve", &Catalog::reserve);
                cls.def("subset",
                        (Catalog(Catalog::*)(ndarray::Array<bool const, 1> const &) const) & Catalog::subset);
                cls.def("subset",
                        (Catalog(Catalog::*)(std::ptrdiff_t, std::ptrdiff_t, std::ptrdiff_t) const) &
                                Catalog::subset);

                // Overloads in pybind11 are tried in the order they are
                // defined, so we define those that we think will see the
                // most usage.  That starts with single-row indexing and
                // column access by string field name, followed by column
                // access by Key.
                cls.def(
                    "_getitem_",
                    [](CatalogT<Record> &self, std::ptrdiff_t index) {
                        return self.get(utils::python::cppIndex(self.size(), index));
                    }
                );
                cls.def(
                    "_getitem_",
                    [](CatalogT<Record> &self, std::string const & name) {
                        return ColumnGetter<Record>(self)(name);
                    }
                );
                cls.def(
                    "get_column",
                    [](
                        CatalogT<Record> &self,
                        std::string const & name,
                        bool force_copy,
                        bool ensure_writeable
                    ) {
                        return ColumnGetter<Record>(self, force_copy, ensure_writeable)(name);
                    },
                    pybind11::arg("name"),
                    pybind11::arg("force_copy") = false,
                    pybind11::arg("ensure_writeable") = false
                );
                FieldTypes::for_each_nullptr(_CatalogOverloadHelper(cls));

                cls.def(
                    "_set_flag",
                    [](Catalog &self, Key<Flag> const & key, ndarray::Array<bool const, 1> const & array) {
                        _setFlagColumnToArray(self, key, array);
                    }
                );
                cls.def(
                    "_set_flag",
                    [](Catalog &self, Key<Flag> const & key, bool value) {
                        _setFlagColumnToScalar(self, key, value);
                    }
                );

            });
}

}  // namespace python
}  // namespace table
}  // namespace afw
}  // namespace lsst

#endif  // !LSST_AFW_TABLE_PYTHON_CATALOG_H_INCLUDED
