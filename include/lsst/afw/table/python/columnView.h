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
#ifndef AFW_TABLE_PYBIND11_COLUMNVIEW_H_INCLUDED
#define AFW_TABLE_PYBIND11_COLUMNVIEW_H_INCLUDED

#include "pybind11/pybind11.h"

#include "lsst/utils/python.h"

#include "lsst/afw/table/BaseColumnView.h"

namespace lsst {
namespace afw {
namespace table {
namespace python {

/*
 * A helper class for returning numpy arrays from ColumnView.
 *
 * This class works out a few differences between Python and C++:
 *
 * - In C++, we treat view arrays (most types from ColumnView) objects
 *   completely differently from copy arrays in terms of the classes used to
 *   access columns.  In Python, users strongly expect __getitem__ on both
 *   Catalog and ColumnView to take care of both, preferring views and falling
 *   back to copies when needed.
 *
 * - In C++, it thus makes sense for both types to be non-const, because it's
 *   clear from how they are obtained that modifying them will do different
 *   things.  In Python, that clarity is lost, so we mark copied returns as
 *   const to prevent users from thinking that by modifying the column they are
 *   modifying the catalog; instead they just can't modify copies at all.
 *
 * - In C++, we can return arrays of Angle directly, but that isn't a valid
 *   dtype in Python - we have to explicitly return an array of double radians
 *   (which we can still make a view).
 *
 * - In C++, we can't return an array for Flag columns, which are packed bits
 *   rather than (full-byte) bools, so we turn an ndarray lazy expression
 *   objects instead.  We copy those into regular bool arrays for Python.
 *
 * This class can be invoked directly by passing its function-call operator a
 * Key or string column name.  It can also be invoked by passing a SchemaItem;
 * this lets it be used as the argument to Schema::findAndApply, which is
 * actually how column-name lookups work.
 */
class ColumnViewGetter {
public:

    explicit ColumnViewGetter(BaseColumnView const & columns) : _columns(columns) {}

    template <typename T>
    pybind11::array operator()(Key<T> const & key) const {
        ndarray::Array<T, 1, 0> array = _columns[key];
        return pybind11::cast(array);
    }

    template <typename T>
    pybind11::array operator()(Key<Array<T>> const & key) const {
        ndarray::Array<T, 2, 1> array = _columns[key];
        return pybind11::cast(array);
    }

    pybind11::array operator()(Key<std::string> const & key) const {
        ndarray::Array<char, 2, 1> utf8_array = _columns.get_utf8_bytes(key);
        return make_str_array(
            [utf8_array](std::size_t n) -> pybind11::str {
                return pybind11::str(utf8_array[n].getData(), utf8_array.getSize<1>());
            },
            utf8_array.getSize<0>(),
            utf8_array.getSize<1>()
        );
    }

    pybind11::array operator()(Key<Angle> const & key) const {
        ndarray::Array<double, 1, 0> radians = _columns.radians(key);
        return pybind11::cast(radians);
    }

    pybind11::array operator()(Key<Flag> const & key) const {
        ndarray::Array<bool const, 1, 1> array = ndarray::copy(_columns[key]);
        return pybind11::cast(array);
    }

    template <typename T>
    pybind11::array operator()(SchemaItem<T> const & item) const {
        return this->operator()(item.key);
    }

    pybind11::array operator()(std::string const & name) const {
        return _columns.getSchema().findAndApply(name, *this);
    }

    template <typename F>
    static pybind11::array make_str_array(
        F get_pystr,
        pybind11::ssize_t n_rows,
        pybind11::ssize_t n_bytes_per_row
    ) {
        // Add 1 to bytes per row for null terminator.
        ndarray::Vector<std::size_t, 2> shape = ndarray::makeVector(n_rows, n_bytes_per_row + 1);
        ndarray::Array<Py_UCS4, 2, 2> utf32_array = ndarray::allocate(shape);
        auto utf32_iter = utf32_array.begin();
        auto const utf32_end = utf32_array.end();
        for (pybind11::ssize_t n = 0; utf32_iter != utf32_end; ++utf32_iter, ++n) {
            if (!PyUnicode_AsUCS4(get_pystr(n).ptr(), utf32_iter->getData(), utf32_iter->size(), true)) {
                throw pybind11::error_already_set();
            }
        }
        pybind11::array utf32_2d = pybind11::cast(ndarray::Array<Py_UCS4 const, 2, 2>(utf32_array));
        pybind11::dtype dtype("U" + std::to_string(shape[1]));
        return utf32_2d.attr("view")(dtype);
    }

private:
    BaseColumnView const & _columns;
};


class StringBytesColumnViewGetter {
public:

    explicit StringBytesColumnViewGetter(BaseColumnView const & columns) : _columns(columns) {}

    template <typename T>
    pybind11::array operator()(Key<T> const & key) const {
        PyErr_SetString(PyExc_TypeError, "Invalid key type for character bytes view.");
        throw pybind11::error_already_set();
    }

    pybind11::array operator()(Key<std::string> const & key) const {
        ndarray::Array<char, 2, 1> utf8_bytes = _columns.get_utf8_bytes(key);
        return pybind11::cast(utf8_bytes);
    }

    template <typename T>
    pybind11::array operator()(SchemaItem<T> const & item) const {
        return this->operator()(item.key);
    }

    pybind11::array operator()(std::string const & name) const {
        return _columns.getSchema().findAndApply(name, *this);
    }

private:
    BaseColumnView const & _columns;
};


template <typename Record>
using PyColumnView =
        pybind11::class_<ColumnViewT<Record>, std::shared_ptr<ColumnViewT<Record>>, BaseColumnView>;

/**
 * Declare member and static functions for a given instantiation of lsst::afw::table::ColumnViewT<RecordT>.
 *
 * @tparam Record  Record type, e.g. BaseRecord or SimpleRecord.
 *
 * @param[in] wrappers Package manager class will be added to.
 * @param[in] name     Name prefix of the record type, e.g. "Base" or "Simple".
 * @param[in] isBase   Whether this instantiation is only being used as a base class
 *                     (used to set the class name).
 */
template <typename Record>
PyColumnView<Record> declareColumnView(utils::python::WrapperCollection& wrappers, std::string const& name,
                                       bool isBase = false) {
    std::string fullName;
    if (isBase) {
        fullName = "_" + name + "ColumnViewBase";
    } else {
        fullName = name + "ColumnView";
    }
    return wrappers.wrapType(PyColumnView<Record>(wrappers.module, fullName.c_str()),
                             [](auto& mod, auto& cls) {
                                 cls.def("getTable", &ColumnViewT<Record>::getTable);
                                 cls.def_property_readonly("table", &ColumnViewT<Record>::getTable);
                             });
};
}  // namespace python
}  // namespace table
}  // namespace afw
}  // namespace lsst
#endif
