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

#include "lsst/afw/coord/Coord.h"
#include "lsst/afw/table/arrays.h"
#include "lsst/afw/table/FunctorKey.h"

namespace py = pybind11;

using namespace lsst::afw::table;

template <typename T>
void declareFunctorKeys(py::module & mod, const std::string suffix){
    py::class_<OutputFunctorKey<T>, std::shared_ptr<OutputFunctorKey<T>>>
        clsOutputFunctorKey(mod, ("OutputFunctorKey"+suffix).c_str());
    py::class_<InputFunctorKey<T>, std::shared_ptr<InputFunctorKey<T>>>
        clsInputFunctorKey(mod, ("InputFunctorKey"+suffix).c_str());
    py::class_<FunctorKey<T>, std::shared_ptr<FunctorKey<T>>, OutputFunctorKey<T>, InputFunctorKey<T>>
        clsFunctorKey(mod, ("FunctorKey"+suffix).c_str());
    
};

PYBIND11_PLUGIN(_functorKey) {
    py::module mod("_functorKey", "Python wrapper for afw _functorKey library");

    if (_import_array() < 0) {
            PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
            return nullptr;
    };

    /* Module level */
    declareFunctorKeys<lsst::afw::coord::IcrsCoord>(mod, "Coord");
    declareFunctorKeys<ndarray::Array<float const,1,1>>(mod, "ArrayF");
    declareFunctorKeys<ndarray::Array<double const,1,1>>(mod, "ArrayD");

    /* Member types and enums */

    /* Constructors */

    /* Operators */

    /* Members */

    return mod.ptr();
}