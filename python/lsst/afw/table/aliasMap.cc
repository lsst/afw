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
    //clsAliasMap.def("empty", &AliasMap::empty);
    //clsAliasMap.def("apply", &AliasMap::apply);
    clsAliasMap.def("get", &AliasMap::get);
    clsAliasMap.def("set", &AliasMap::set);
    clsAliasMap.def("erase", &AliasMap::erase);
    clsAliasMap.def("contains", &AliasMap::contains);
    clsAliasMap.def("__iter__", [](AliasMap & self){
        return py::make_iterator(self.begin(), self.end());
    }, py::keep_alive<0,1>());

    return mod.ptr();
}