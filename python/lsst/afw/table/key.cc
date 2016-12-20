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

#include "lsst/afw/table/FieldBase.h"
#include "lsst/afw/table/Key.h"

namespace py = pybind11;

namespace lsst {
namespace afw {
namespace table {

template <typename T>
void declareKey(py::module & mod, std::string const & suffix) {
    py::class_<Key<T>, KeyBase<T>, FieldBase<T>> clsKey(mod, ("Key_"+suffix).c_str());
    clsKey.def(py::init<>());
    clsKey.def("_eq_impl", [](const Key<T> & self, Key<T> const & other)-> bool {
        return self == other;
    });
    clsKey.def("isValid", &Key<T>::isValid);
    clsKey.def("getOffset", &Key<T>::getOffset);
};

PYBIND11_PLUGIN(_key) {
    py::module mod("_key", "Python wrapper for afw _key library");
    
    declareKey<std::uint16_t>(mod, "U");
    declareKey<std::int32_t>(mod, "I");
    declareKey<std::int64_t>(mod, "L");
    declareKey<float>(mod, "F");
    declareKey<double>(mod, "D");
    declareKey<std::string>(mod, "String");
    declareKey<lsst::afw::geom::Angle>(mod, "Angle");
    declareKey<lsst::afw::table::Array<std::uint16_t>>(mod, "ArrayU");
    declareKey<lsst::afw::table::Array<int>>(mod, "ArrayI");
    declareKey<lsst::afw::table::Array<float>>(mod, "ArrayF");
    declareKey<lsst::afw::table::Array<double>>(mod, "ArrayD");

    /* Module level */

    /* Member types and enums */

    /* Constructors */

    /* Operators */

    /* Members */

    return mod.ptr();
}

}}}  // namespace lsst::afw::table
