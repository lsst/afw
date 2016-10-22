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

#include "lsst/afw/table/misc.h"
#include "lsst/afw/table/FieldBase.h"
#include "lsst/afw/table/Field.h"

namespace py = pybind11;
using namespace pybind11::literals;

using namespace lsst::afw::table;

template <typename T>
void declareField(py::module & mod, const std::string suffix){
    py::class_<Field<T>, FieldBase<T>> clsField(mod, ("Field_"+suffix).c_str());
    /* Constructors */
    clsField.def(py::init<std::string const &,
                      std::string const &,
                      std::string const &,
                      FieldBase<T> const &>(),
             "name"_a, "doc"_a, "units"_a="", "size"_a=FieldBase<T>());
    clsField.def(py::init<std::string const &,
                      std::string const &,
                      FieldBase<T> const &>(),
             "name"_a, "doc"_a, "size"_a=FieldBase<T>());
    clsField.def(py::init<std::string const &, std::string const &, std::string const &, int>());
    clsField.def(py::init<std::string const &, std::string const &, int>());
    clsField.def("getName", &Field<T>::getName);
    clsField.def("getDoc", &Field<T>::getDoc);
    clsField.def("getUnits", &Field<T>::getUnits);
    clsField.def("copyRenamed", &Field<T>::copyRenamed);
};
template <>
void declareField<std::string>(py::module & mod, const std::string suffix){
    py::class_<Field<std::string>, FieldBase<std::string>> clsField(mod, ("Field_"+suffix).c_str());
    /* Constructors */
    clsField.def(py::init<std::string const &,
                      std::string const &,
                      std::string const &,
                      FieldBase<std::string> const &>(),
             "name"_a, "doc"_a, "units"_a="", "size"_a=FieldBase<std::string>(1));
    clsField.def(py::init<std::string const &,
                      std::string const &,
                      FieldBase<std::string> const &>(),
             "name"_a, "doc"_a, "size"_a=FieldBase<std::string>(1));
    clsField.def(py::init<std::string const &, std::string const &, std::string const &, int>());
    clsField.def(py::init<std::string const &, std::string const &, int>());
    clsField.def("getName", &Field<std::string>::getName);
    clsField.def("getDoc", &Field<std::string>::getDoc);
    clsField.def("getUnits", &Field<std::string>::getUnits);
};


PYBIND11_PLUGIN(_field) {
    py::module mod("_field", "Python wrapper for afw _field library");

    /* Module level */

    /* Member types and enums */

    /* Constructors */

    /* Operators */

    /* Members */
    declareField<std::uint16_t>(mod, "U");
    declareField<std::int32_t>(mod, "I");
    declareField<std::int64_t>(mod, "L");
    declareField<float>(mod, "F");
    declareField<double>(mod, "D");
    declareField<std::string>(mod, "String");
    declareField<lsst::afw::table::Flag>(mod, "Flag");
    declareField<lsst::afw::geom::Angle>(mod, "Angle");
    declareField<lsst::afw::table::Array<std::uint16_t>>(mod, "ArrayU");
    declareField<lsst::afw::table::Array<int>>(mod, "ArrayI");
    declareField<lsst::afw::table::Array<float>>(mod, "ArrayF");
    declareField<lsst::afw::table::Array<double>>(mod, "ArrayD");

    return mod.ptr();
}