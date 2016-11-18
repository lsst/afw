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

#include "lsst/afw/table/AliasMap.h"

namespace py = pybind11;
using namespace pybind11::literals;

using namespace lsst::afw::table;

PYBIND11_PLUGIN(_aliasMap) {
    py::module mod("_aliasMap", "Python wrapper for afw _aliasMap library");

    /* Module level */
    py::class_<AliasMap, std::shared_ptr<AliasMap>> clsAliasMap(mod, "AliasMap");

    /* Member types and enums */

    /* Constructors */
    clsAliasMap.def(py::init<>());
    clsAliasMap.def(py::init<AliasMap const &>());

    /* Operators */

    /* Members */
    //clsAliasMap.def("begin", &AliasMap::begin);
    //clsAliasMap.def("end", &AliasMap::end);
    clsAliasMap.def("__len__", &AliasMap::size);
    clsAliasMap.def("empty", &AliasMap::empty);
    clsAliasMap.def("apply", &AliasMap::apply, "name"_a);
    clsAliasMap.def("get", &AliasMap::get, "alias"_a);
    clsAliasMap.def("__getitem__", &AliasMap::get, "alias"_a);
    clsAliasMap.def("set", &AliasMap::set, "alias"_a, "target"_a);
    clsAliasMap.def("__setitem__", &AliasMap::set);
    clsAliasMap.def("erase", &AliasMap::erase, "alias"_a);
    clsAliasMap.def("__delitem__", &AliasMap::erase, "alias"_a);
    clsAliasMap.def("__eq__", [](AliasMap & self, AliasMap & other){ return self == other; });
    clsAliasMap.def("__ne__", [](AliasMap & self, AliasMap & other){ return self != other; });
    clsAliasMap.def("contains", &AliasMap::contains, "other"_a);
    clsAliasMap.def("items", [](AliasMap & self){
        return py::make_iterator(self.begin(), self.end());
    }, py::keep_alive<0,1>());

    return mod.ptr();
}