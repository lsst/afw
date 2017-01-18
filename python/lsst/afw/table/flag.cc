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
//#include <pybind11/stl.h>

#include "lsst/afw/table/misc.h"
#include "lsst/afw/table/Flag.h"

namespace py = pybind11;

namespace lsst {
namespace afw {
namespace table {

PYBIND11_PLUGIN(_flag) {
    py::module mod("_flag", "Python wrapper for afw _flag library");

    /* Module level */
    py::class_<FieldBase<Flag>> clsFieldBase(mod, "FieldBase_Flag");
    py::class_<KeyBase<Flag>> clsKeyBase(mod, "KeyBase_Flag");
    py::class_<Key<Flag>, KeyBase<Flag>, FieldBase<Flag>> clsKey(mod, "Key_Flag");
    clsKey.def("_eq_impl", [](const Key<Flag> & self, Key<Flag> const & other)-> bool {
        return self == other;
    });

    /* Member types and enums */

    /* Constructors */
    clsFieldBase.def(py::init<>());
    clsFieldBase.def(py::init<int>());
    clsKey.def(py::init<>());

    /* Operators */

    /* Members */
    clsFieldBase.def_static("getTypeString", &FieldBase<Flag>::getTypeString);
    clsKey.def("getBit", &Key<Flag>::getBit);
    clsKey.def("isValid", &Key<Flag>::isValid);

    return mod.ptr();
}

}}}  // namespace lsst::afw::table
