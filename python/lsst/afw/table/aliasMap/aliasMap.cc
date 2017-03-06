/*
 * LSST Data Management System
 * Copyright 2008-2017  AURA/LSST.
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

#include "pybind11/pybind11.h"

#include "lsst/afw/table/AliasMap.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst {
namespace afw {
namespace table {
namespace {

using PyAliasMap = py::class_<AliasMap, std::shared_ptr<AliasMap>>;

PYBIND11_PLUGIN(aliasMap) {
    py::module mod("aliasMap");

    PyAliasMap cls(mod, "AliasMap");

    cls.def(py::init<>());
    cls.def(py::init<AliasMap const &>());

    cls.def("__len__", &AliasMap::size);
    cls.def("empty", &AliasMap::empty);
    cls.def("apply", &AliasMap::apply, "name"_a);
    cls.def("get", &AliasMap::get, "alias"_a);
    cls.def("__getitem__", &AliasMap::get, "alias"_a);
    cls.def("set", &AliasMap::set, "alias"_a, "target"_a);
    cls.def("__setitem__", &AliasMap::set);
    cls.def("erase", &AliasMap::erase, "alias"_a);
    cls.def("__delitem__", &AliasMap::erase, "alias"_a);
    cls.def("__eq__", [](AliasMap & self, AliasMap & other) { return self == other; });
    cls.def("__ne__", [](AliasMap & self, AliasMap & other) { return self != other; });
    cls.def("contains", &AliasMap::contains, "other"_a);
    cls.def("__contains__", &AliasMap::contains);
    cls.def(
        "items",
        [](AliasMap & self) {
            return py::make_iterator(self.begin(), self.end());
        },
        py::keep_alive<0,1>()
    );

    return mod.ptr();
}

}}}}  // namespace lsst::afw::table::<anonymous>
