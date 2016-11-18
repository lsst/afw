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

#include "lsst/afw/table/Key.h"
#include "lsst/afw/table/BaseRecord.h"
#include "lsst/afw/table/FunctorKey.h"
#include "lsst/afw/table/arrays.h"

namespace py = pybind11;

using namespace lsst::afw::table;

template <typename T>
void declareArrayKey(py::module & mod, const std::string suffix){
    py::class_<ArrayKey<T>,
               std::shared_ptr<ArrayKey<T>>,
               FunctorKey< ndarray::Array<T const,1,1> >> clsArrayKey(mod, ("Array"+suffix+"Key").c_str());
    
    clsArrayKey.def(py::init<>());
    clsArrayKey.def(py::init<Key<Array<T>> const &>());
    clsArrayKey.def(py::init<std::vector< Key<T> > const &>());
    clsArrayKey.def(py::init<SubSchema const &>());
    
    clsArrayKey.def_static("addFields", (ArrayKey<T> (*)(
        Schema &,
        std::string const &,
        std::string const &,
        std::string const &,
        std::vector<T> const &
    )) &ArrayKey<T>::addFields);
    clsArrayKey.def_static("addFields", (ArrayKey<T> (*)(
        Schema &,
        std::string const &,
        std::string const &,
        std::string const &,
        int size
    )) &ArrayKey<T>::addFields);
    clsArrayKey.def("get", &ArrayKey<T>::get);
    clsArrayKey.def("set", &ArrayKey<T>::set);
    clsArrayKey.def("isValid", &ArrayKey<T>::isValid);
    clsArrayKey.def("__eq__", [](ArrayKey<T> & self, ArrayKey<T> & other){
        return self==other;
    });
    clsArrayKey.def("__ne__", [](ArrayKey<T> & self, ArrayKey<T> & other){
        return self!=other;
    });
    clsArrayKey.def("_get_", [](ArrayKey<T> & self, int i){
        return self[i];
    });
    clsArrayKey.def("getSize", &ArrayKey<T>::getSize);
    clsArrayKey.def("slice", &ArrayKey<T>::slice);
};

PYBIND11_PLUGIN(_arrays) {
    py::module mod("_arrays", "Python wrapper for afw _arrays library");
    
    if (_import_array() < 0) {
            PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
            return nullptr;
    };

    /* Module level */
    declareArrayKey<float>(mod, "F");
    declareArrayKey<double>(mod, "D");

    /* Member types and enums */

    /* Constructors */

    /* Operators */

    /* Members */

    return mod.ptr();
}