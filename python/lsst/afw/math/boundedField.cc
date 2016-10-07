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

#include "lsst/afw/geom/Point.h"
#include "lsst/afw/math/BoundedField.h"

namespace py = pybind11;
PYBIND11_DECLARE_HOLDER_TYPE(MyType, std::shared_ptr<MyType>);

using namespace lsst::afw::math;

PYBIND11_PLUGIN(_boundedField) {
    py::module mod("_boundedField", "Python wrapper for afw _boundedField library");
    
    if (_import_array() < 0) {
            PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
            return nullptr;
    };
    
    /* Bounded Field */
    py::class_<BoundedField, std::shared_ptr<BoundedField>> clsBoundedField(mod, "BoundedField");
    clsBoundedField.def("evaluate", (double (BoundedField::*)(double, double) const)
        &BoundedField::evaluate);
    clsBoundedField.def("evaluate", (ndarray::Array<double,1,1> 
        (BoundedField::*)(ndarray::Array<double const,1> const &,
                          ndarray::Array<double const,1> const &) const) &BoundedField::evaluate);
    clsBoundedField.def("getBBox", &BoundedField::getBBox);

    /* Module level */

    /* Member types and enums */

    /* Constructors */

    /* Operators */

    /* Members */

    return mod.ptr();
}