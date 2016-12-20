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
//#include <pybind11/stl.h>

#include "lsst/afw/table/slots.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst {
namespace afw {
namespace table {

namespace {
/**
Declare standard methods for subclasses of SlotDefinition (but not SlotDefinition itself).
*/
template<typename SlotDefSubclass>
void declareSlotDefinitions(py::class_<SlotDefSubclass, SlotDefinition> & cls) {
    cls.def("isValid", &SlotDefSubclass::isValid);
    cls.def("getMeasKey", &SlotDefSubclass::getMeasKey);
    cls.def("getErrKey", &SlotDefSubclass::getErrKey);
    cls.def("getFlagKey", &SlotDefSubclass::getFlagKey);
    cls.def("setKeys", &SlotDefSubclass::setKeys, "alias"_a, "schema"_a);
}

}  // anonymous namespace

PYBIND11_PLUGIN(_slots) {
    py::module mod("_slots", "Python wrapper for afw _slots library");

    /* Module level */
    py::class_<SlotDefinition> clsSlotDefinition(mod, "SlotDefinition");
    py::class_<FluxSlotDefinition, SlotDefinition> clsFluxSlotDefinition(mod, "FluxSlotDefinition");
    py::class_<CentroidSlotDefinition, SlotDefinition>
        clsCentroidSlotDefinition(mod, "CentroidSlotDefinition");
    py::class_<ShapeSlotDefinition, SlotDefinition> clsShapeSlotDefinition(mod, "ShapeSlotDefinition");

    /* Member types and enums */

    /* Constructors */
    clsSlotDefinition.def(py::init<std::string const &>(), "name"_a);

    /* Operators */

    /* Members */
    clsSlotDefinition.def("getName", &SlotDefinition::getName);
    clsSlotDefinition.def("getAlias", &SlotDefinition::getAlias);

    declareSlotDefinitions(clsFluxSlotDefinition);
    declareSlotDefinitions(clsCentroidSlotDefinition);
    declareSlotDefinitions(clsShapeSlotDefinition);

    return mod.ptr();
}

}}}  // namespace lsst::afw::table
