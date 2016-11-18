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

#include "lsst/afw/table/Key.h"
#include "lsst/afw/table/misc.h"
#include "lsst/afw/table/KeyBase.h"

namespace py = pybind11;

using namespace lsst::afw::table;

template <typename T>
void declareKeyBase(py::module &mod, const std::string & suffix){
    py::class_<KeyBase<T>> clsKeyBase(mod, ("KeyBase_"+suffix).c_str());
};

template <typename U>
void declareKeyBaseArray(py::module &mod, const std::string & suffix){
    /*
    KeyBase has slightly different methods if the type is an Array.
    Currently it does not appear that those methods need to be wrapped but in case this changes
    in the future we use a different declare function for Array types.
    */
    py::class_<KeyBase<lsst::afw::table::Array<U>>> clsKeyBase(mod, ("KeyBase_"+suffix).c_str());
    clsKeyBase.def("_getitem_", [](const KeyBase<lsst::afw::table::Array<U>> &self, int i) {
        return self[i];
    });
    clsKeyBase.def("slice", &KeyBase<lsst::afw::table::Array<U>>::slice);
};

PYBIND11_PLUGIN(_keyBase) {
    py::module mod("_keyBase", "Python wrapper for afw _keyBase library");

    /* Module level */

    /* Member types and enums */

    /* Constructors */

    /* Operators */

    /* Members */
    declareKeyBase<std::uint16_t>(mod, "U");
    declareKeyBase<std::int32_t>(mod, "I");
    declareKeyBase<std::int64_t>(mod, "L");
    declareKeyBase<float>(mod, "F");
    declareKeyBase<double>(mod, "D");
    declareKeyBase<std::string>(mod, "String");
    declareKeyBase<lsst::afw::geom::Angle>(mod, "Angle");
    declareKeyBaseArray<std::uint16_t>(mod, "ArrayU");
    declareKeyBaseArray<int>(mod, "ArrayI");
    declareKeyBaseArray<float>(mod, "ArrayF");
    declareKeyBaseArray<double>(mod, "ArrayD");

    return mod.ptr();
}