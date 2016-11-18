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

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst {
namespace afw {
namespace table {

/// Declare a FieldBase<T>
template <typename T>
py::class_<FieldBase<T>> declareFieldBase(py::module & mod, const std::string & suffix) {
    py::class_<FieldBase<T>> clsFieldBase(mod, ("FieldBase_"+suffix).c_str());

    clsFieldBase.def(py::init<>());
    clsFieldBase.def(py::init<int>(), "size"_a=0);

    clsFieldBase.def_static("getTypeString", &FieldBase<T>::getTypeString);

//    clsField.def("getElementCount", &FieldBase<std::string>::getElementCount);

    return clsFieldBase;
};

/// Declare a FieldBase<lsst::afw::table::Array<U>>
template <typename U>
void declareFieldBaseArray(py::module & mod, std::string const & suffix) {
    typedef lsst::afw::table::Array<U> Array;
    auto clsFieldBase = declareFieldBase<Array>(mod, suffix);

    clsFieldBase.def(py::init<int>(), "size"_a=0);

    clsFieldBase.def("getSize", &FieldBase<Array>::getSize);
};

/// Declare a FieldBase<std::string>
void declareFieldBaseString(py::module & mod) {
    auto clsFieldBase = declareFieldBase<std::string>(mod, "String"); 

    clsFieldBase.def(py::init<int>(), "size"_a=-1);

    clsFieldBase.def("getSize", &FieldBase<std::string>::getSize);
};

PYBIND11_PLUGIN(_fieldBase) {
    py::module mod("_fieldBase", "Python wrapper for afw _fieldBase library");

    /* Module level */
    declareFieldBaseString(mod);
    declareFieldBase<std::uint16_t>(mod, "U");
    declareFieldBase<std::int32_t>(mod, "I");
    declareFieldBase<std::int64_t>(mod, "L");
    declareFieldBase<float>(mod, "F");
    declareFieldBase<double>(mod, "D");
    declareFieldBase<lsst::afw::geom::Angle>(mod, "Angle");
    declareFieldBaseArray<std::uint16_t>(mod, "ArrayU");
    declareFieldBaseArray<int>(mod, "ArrayI");
    declareFieldBaseArray<float>(mod, "ArrayF");
    declareFieldBaseArray<double>(mod, "ArrayD");

    /* Member types and enums */

    /* Constructors */

    /* Operators */

    /* Members */

    return mod.ptr();
}

}}}  // namespace lsst::afw::table
