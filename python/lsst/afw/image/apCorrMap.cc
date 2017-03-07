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

#include <string>
#include <vector>

#include <pybind11/pybind11.h>
//#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include "lsst/afw/table/io/python.h"

#include "lsst/afw/image/ApCorrMap.h"
#include "lsst/afw/table/io/Persistable.h"

namespace py = pybind11;

using namespace lsst::afw::image;

PYBIND11_PLUGIN(_apCorrMap) {
    py::module mod("_apCorrMap", "Python wrapper for afw _apCorrMap library");

    /* Module level */
    lsst::afw::table::io::python::declarePersistableFacade<ApCorrMap>(mod, "ApCorrMap");
    py::class_<ApCorrMap,
               std::shared_ptr<ApCorrMap>,
               lsst::afw::table::io::PersistableFacade<ApCorrMap>,
               lsst::afw::table::io::Persistable>
        clsApCorrMap(mod, "ApCorrMap");

    /* Member types and enums */

    /* Constructors */
    clsApCorrMap.def(py::init<>());

    /* Operators */
    // TODO: pybind11 ApCorrMap's C++ operators should return ApCorrMap &
    clsApCorrMap.def("__imul__", [](ApCorrMap & self, double const scale) {
        self *= scale;
        return self;
    });
    clsApCorrMap.def("__idiv__", [](ApCorrMap & self, double const scale) {
        self /= scale;
        return self;
    });
    clsApCorrMap.def("__itruediv__", [](ApCorrMap & self, double const scale) {
        self /= scale;
        return self;
    });

    /* Members */
    clsApCorrMap.def("get", &ApCorrMap::get);
    clsApCorrMap.def("set", &ApCorrMap::set);

    // I couldn't figure out how to get Swig to expose the C++ iterators using std_map.i and (e.g.)
    // code in utils.i; it kept wrapping the iterators as opaque objects I couldn't deference, probably
    // due to some weirdness with the typedefs.  I don't want to sink time into debugging that.
    // So I just wrote this function to return a list of names, and I'll base the Python iterators on
    // that. -- Russell Owen
    // TODO: pybind11 replace this with pybind11's support for iterators/slicing/etc.
    clsApCorrMap.def("keys", [](ApCorrMap const & self) {
        // Can't create a vector of const element type, not sure why
        using KeyType = std::remove_const<ApCorrMap::Iterator::value_type::first_type>::type;
        auto r = std::vector<KeyType>();
        r.reserve(self.size());
        for (ApCorrMap::Iterator i = self.begin(); i != self.end(); ++i) {
            r.push_back(i->first);
        }
        return r;
    });
    // values, items, and __iter__ defined in apCorrMap.py

    clsApCorrMap.def("__len__", &ApCorrMap::size);
    clsApCorrMap.def("__getitem__", &ApCorrMap::operator[]);
    clsApCorrMap.def("__setitem__", &ApCorrMap::set);
    clsApCorrMap.def("__contains__", [](ApCorrMap const & self, std::string name) {
        // Test for empty pointer, which is not the same as null pointer
        return self.get(name).use_count() > 0;
    });

    return mod.ptr();
}