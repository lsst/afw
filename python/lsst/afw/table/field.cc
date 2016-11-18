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

namespace lsst {
namespace afw {
namespace table {

/**
Declare a Field<T>

@param[in] mod  Pybind11 module
@param[in] suffix  The python name for the object = "Field_" + suffix
@param[in] defSize  Default size, as a FieldBase<T>; most types can use the default FieldBase<T>(),
                    but if T = std::string then specify FieldBase<std::string>(1)
*/
template <typename T>
void declareField(py::module & mod, std::string const & suffix,
                                  FieldBase<T> const & defSize=FieldBase<T>()) {
    py::class_<Field<T>, FieldBase<T>> clsField(mod, ("Field_"+suffix).c_str());
    /* Constructors */
    clsField.def(py::init<std::string const &,
                      std::string const &,
                      std::string const &,
                      FieldBase<T> const &>(),
             "name"_a, "doc"_a, "units"_a="", "size"_a=defSize);
    clsField.def(py::init<std::string const &,
                      std::string const &,
                      FieldBase<T> const &>(),
             "name"_a, "doc"_a, "size"_a=defSize);
    clsField.def(py::init<std::string const &, std::string const &, std::string const &, int>());
    clsField.def(py::init<std::string const &, std::string const &, int>());

    clsField.def("getName", &Field<T>::getName);
    clsField.def("getDoc", &Field<T>::getDoc);
    clsField.def("getUnits", &Field<T>::getUnits);
    clsField.def("copyRenamed", &Field<T>::copyRenamed);
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
    declareField<std::string>(mod, "String", FieldBase<std::string>(1));
    declareField<lsst::afw::table::Flag>(mod, "Flag");
    declareField<lsst::afw::geom::Angle>(mod, "Angle");
    declareField<lsst::afw::table::Array<std::uint16_t>>(mod, "ArrayU");
    declareField<lsst::afw::table::Array<int>>(mod, "ArrayI");
    declareField<lsst::afw::table::Array<float>>(mod, "ArrayF");
    declareField<lsst::afw::table::Array<double>>(mod, "ArrayD");

    return mod.ptr();
}

}}}  // namespace lsst::afw::table
