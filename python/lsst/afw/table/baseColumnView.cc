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

using namespace lsst::afw::table;

template <typename T>
void declareBaseColumnViewOverloads(py::class_<BaseColumnView> clsBaseColumnView){
    clsBaseColumnView.def("__getitem__", [](BaseColumnView & self, Key<T> const & key)->typename ndarray::Array<T,1> const{
        return self[key];
    });
};
template <typename U>
void declareBaseColumnViewArrayOverloads(py::class_<BaseColumnView> clsBaseColumnView){
    clsBaseColumnView.def("__getitem__", [](BaseColumnView & self, Key<lsst::afw::table::Array<U>> const & key)->typename ndarray::Array<U,2,1> const{
        return self[key];
    });
};
void declareBaseColumnViewFlagOverloads(py::class_<BaseColumnView> clsBaseColumnView){
    clsBaseColumnView.def("__getitem__", [](BaseColumnView & self, Key<Flag> const & key)->ndarray::Array<bool const,1,1> const {
        return ndarray::copy(self[key]);
    });
};

template <typename RecordT>
void declareColumnViewT(py::module & mod){
    py::class_<ColumnViewT<RecordT>, BaseColumnView> cls(mod, "ColumnViewT");
};

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
    py::class_<detail::FlagExtractor> clsFlagExtractor(mod, "FlagExtractor");

    /* Member types and enums */

    /* Constructors */
    clsFlagExtractor.def(py::init<Key<Flag> const &>());

    /* Operators */

    /* Members */
    clsBaseColumnView.def("getTable", &BaseColumnView::getTable);
    clsBaseColumnView.def("getSchema", &BaseColumnView::getSchema);
    clsFlagExtractor.def("__call__", [](detail::FlagExtractor & self, argument_type element){
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