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

#include "numpy/arrayobject.h"
#include "ndarray/pybind11.h"
#include "ndarray/converter.h"

#include "lsst/afw/table/Source.h"
#include "lsst/afw/table/Key.h"
#include "lsst/afw/table/BaseColumnView.h"

namespace ndarray { namespace detail {
template <> struct NumpyTraits<lsst::afw::geom::Angle> : public NumpyTraits<double> {};
}}

namespace py = pybind11;
using namespace py::literals;

namespace lsst {
namespace afw {
namespace table {

namespace {

template <typename T>
void declareBaseColumnViewOverloads(py::class_<BaseColumnView> clsBaseColumnView) {
    clsBaseColumnView.def("_basicget", [](BaseColumnView & self, Key<T> const & key)
                          ->typename ndarray::Array<T,1> const {
        return self[key];
    });
};
template <typename U>
void declareBaseColumnViewArrayOverloads(py::class_<BaseColumnView> clsBaseColumnView) {
    clsBaseColumnView.def("_basicget", [](BaseColumnView & self, Key<lsst::afw::table::Array<U>> const & key)
                          ->typename ndarray::Array<U,2,1> const {
        return self[key];
    });
};
void declareBaseColumnViewFlagOverloads(py::class_<BaseColumnView> clsBaseColumnView) {
    clsBaseColumnView.def("_basicget", [](BaseColumnView & self, Key<Flag> const & key)
                          ->ndarray::Array<bool const,1,1> const {
        return ndarray::copy(self[key]);
    });
};

template <typename RecordT>
void declareColumnViewT(py::module & mod) {
    py::class_<ColumnViewT<RecordT>, BaseColumnView> cls(mod, "ColumnViewT");

    cls.def("getTable", &ColumnViewT<RecordT>::getTable);
    cls.def_property_readonly("table", &ColumnViewT<RecordT>::getTable);
};

} // namespace lsst::afw::table::<anonymous>


PYBIND11_PLUGIN(_baseColumnView) {
    py::module mod("_baseColumnView", "Python wrapper for afw _baseColumnView library");
    typedef Field<Flag>::Element argument_type;
    typedef bool result_type;
    
    if (_import_array() < 0) {
            PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
            return nullptr;
        }

    /* Module level */
    py::class_<BaseColumnView> clsBaseColumnView(mod, "BaseColumnView");
    py::class_<BitsColumn> clsBitsColumn(mod, "BitsColumn");
    py::class_<detail::FlagExtractor> clsFlagExtractor(mod, "FlagExtractor");

    /* Member types and enums */

    /* Constructors */
    clsFlagExtractor.def(py::init<Key<Flag> const &>());

    /* Operators */

    /* Members */
    clsBaseColumnView.def("getTable", &BaseColumnView::getTable);
    clsBaseColumnView.def_property_readonly("table", &BaseColumnView::getTable);
    clsBaseColumnView.def("getSchema", &BaseColumnView::getSchema);
    clsBaseColumnView.def_property_readonly("schema", &BaseColumnView::getSchema);
    // _getBits supports a Python version of getBits that accepts None and field names as keys
    clsBaseColumnView.def("_getBits", &BaseColumnView::getBits);
    clsBaseColumnView.def("getAllBits", &BaseColumnView::getAllBits);
    //clsBaseColumnView.def("__getitem__", &BaseColumnView::operator[]);

    clsBitsColumn.def("getArray", &BitsColumn::getArray);
    clsBitsColumn.def_property_readonly("array", &BitsColumn::getArray);
    clsBitsColumn.def("getBit",
                      (BitsColumn::IntT (BitsColumn::*)(Key<Flag> const &) const) &BitsColumn::getBit,
                      "key"_a);
    clsBitsColumn.def("getBit",
                      (BitsColumn::IntT (BitsColumn::*)(std::string const &) const) &BitsColumn::getBit,
                      "name"_a);
    clsBitsColumn.def("getMask",
                      (BitsColumn::IntT (BitsColumn::*)(Key<Flag> const &) const) &BitsColumn::getMask,
                      "key"_a);
    clsBitsColumn.def("getMask",
                      (BitsColumn::IntT (BitsColumn::*)(std::string const &) const) &BitsColumn::getMask,
                      "name"_a);

    clsFlagExtractor.def("__call__", [](detail::FlagExtractor & self, argument_type element) {
        return self(element);
    });
    
    declareBaseColumnViewOverloads<std::uint16_t>(clsBaseColumnView);
    declareBaseColumnViewOverloads<std::int32_t>(clsBaseColumnView);
    declareBaseColumnViewOverloads<std::int64_t>(clsBaseColumnView);
    declareBaseColumnViewOverloads<float>(clsBaseColumnView);
    declareBaseColumnViewOverloads<double>(clsBaseColumnView);
    declareBaseColumnViewFlagOverloads(clsBaseColumnView);
    // std::string columns will need to be declared in a slightly different way
    // but none of the tests so far have required using a string column, so this functinality
    // can be added when it is needed.
    //declareBaseColumnViewOverloads<std::string>(clsBaseColumnView);
    declareBaseColumnViewOverloads<lsst::afw::geom::Angle>(clsBaseColumnView);
    declareBaseColumnViewArrayOverloads<std::uint16_t>(clsBaseColumnView);
    declareBaseColumnViewArrayOverloads<int>(clsBaseColumnView);
    declareBaseColumnViewArrayOverloads<float>(clsBaseColumnView);
    declareBaseColumnViewArrayOverloads<double>(clsBaseColumnView);
    
    declareColumnViewT<SourceRecord>(mod);

    return mod.ptr();
}

}}}  // namespace lsst::afw::table
