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
#include <memory>

#include <pybind11/pybind11.h>
//#include <pybind11/stl.h>

#include "lsst/afw/table/io/Persistable.h"
#include "lsst/afw/table/io/python.h"
#include "lsst/afw/table/Schema.h"
#include "lsst/afw/table/Exposure.h"
#include "lsst/afw/image/CoaddInputs.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst {
namespace afw {
namespace image {

PYBIND11_PLUGIN(_coaddInputs) {
    py::module mod("_coaddInputs", "Python wrapper for afw _coaddInputs library");

    /* Module level */

    table::io::python::declarePersistableFacade<CoaddInputs>(mod, "CoaddInputs");

    py::class_<CoaddInputs, std::shared_ptr<CoaddInputs>,
               table::io::PersistableFacade<CoaddInputs>,
               table::io::Persistable> cls(mod, "CoaddInputs");

    /* Member types and enums */
    cls.def_readwrite("visits", &CoaddInputs::visits);
    cls.def_readwrite("ccds", &CoaddInputs::ccds);

    /* Constructors */
    cls.def(py::init<>());
    cls.def(py::init<table::Schema const &, table::Schema const &>(),
            "visitSchema"_a, "ccdSchema"_a);
    cls.def(py::init<table::ExposureCatalog const &, table::ExposureCatalog const &>(),
            "visits"_a, "ccds"_a);

    /* Operators */

    /* Members */
    cls.def("isPersistable", &CoaddInputs::isPersistable);

    return mod.ptr();
}

}}}  // namespace lsst::afw::image
