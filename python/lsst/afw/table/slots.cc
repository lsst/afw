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

#include "pybind11/pybind11.h"
#include "pybind11/eigen.h"

#include "lsst/afw/table/slots.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst {
namespace afw {
namespace table {
namespace {

// tiny classes only used in afw::table: no need for shared_ptr.
using PySlotDefinition = py::class_<SlotDefinition>;

void declareSlotDefinition(py::module &mod) {
    PySlotDefinition cls(mod, "SlotDefinition");
    cls.def("getName", &SlotDefinition::getName);
    cls.def("getAlias", &SlotDefinition::getAlias);
}

/*
Declare standard methods for subclasses of SlotDefinition (but not SlotDefinition itself).
*/
template <typename Class>
void declareSlotDefinitionSubclass(py::module &mod, std::string const &name) {
    py::class_<Class, SlotDefinition> cls(mod, name.c_str());
    cls.def(py::init<std::string const &>(), "name"_a);
    cls.def("isValid", &Class::isValid);
    cls.def("getMeasKey", &Class::getMeasKey);
    cls.def("getErrKey", &Class::getErrKey);
    cls.def("getFlagKey", &Class::getFlagKey);
    cls.def("setKeys", &Class::setKeys, "alias"_a, "schema"_a);
}

PYBIND11_PLUGIN(slots) {
    py::module mod("slots");
    py::module::import("lsst.afw.table.aggregates");

    declareSlotDefinition(mod);
    declareSlotDefinitionSubclass<FluxSlotDefinition>(mod, "Flux");
    declareSlotDefinitionSubclass<CentroidSlotDefinition>(mod, "Centroid");
    declareSlotDefinitionSubclass<ShapeSlotDefinition>(mod, "Shape");

    return mod.ptr();
}
}
}
}
}  // namespace lsst::afw::table::<anonymous>
