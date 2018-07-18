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

#include "lsst/afw/detection/FootprintCtrl.h"

namespace py = pybind11;

using namespace py::literals;

namespace lsst {
namespace afw {
namespace detection {

PYBIND11_MODULE(footprintCtrl, mod) {
    py::class_<FootprintControl> clsFootprintControl(mod, "FootprintControl");

    /* Constructors */
    clsFootprintControl.def(py::init<>());
    clsFootprintControl.def(py::init<bool, bool>(), "circular"_a, "isotropic"_a = false);
    clsFootprintControl.def(py::init<bool, bool, bool, bool>(), "left"_a, "right"_a, "up"_a, "down"_a);

    /* Members */
    clsFootprintControl.def("growCircular", &FootprintControl::growCircular);
    clsFootprintControl.def("growIsotropic", &FootprintControl::growIsotropic);
    clsFootprintControl.def("growLeft", &FootprintControl::growLeft);
    clsFootprintControl.def("growRight", &FootprintControl::growRight);
    clsFootprintControl.def("growUp", &FootprintControl::growUp);
    clsFootprintControl.def("growDown", &FootprintControl::growDown);

    clsFootprintControl.def("isCircular", &FootprintControl::isCircular);
    clsFootprintControl.def("isIsotropic", &FootprintControl::isIsotropic);
    clsFootprintControl.def("isLeft", &FootprintControl::isLeft);
    clsFootprintControl.def("isRight", &FootprintControl::isRight);
    clsFootprintControl.def("isUp", &FootprintControl::isUp);
    clsFootprintControl.def("isDown", &FootprintControl::isDown);

    py::class_<HeavyFootprintCtrl> clsHeavyFootprintCtrl(mod, "HeavyFootprintCtrl");

    py::enum_<HeavyFootprintCtrl::ModifySource>(clsHeavyFootprintCtrl, "ModifySource")
            .value("NONE", HeavyFootprintCtrl::ModifySource::NONE)
            .value("SET", HeavyFootprintCtrl::ModifySource::SET)
            .export_values();

    clsHeavyFootprintCtrl.def(py::init<HeavyFootprintCtrl::ModifySource>(),
                              "modifySource"_a = HeavyFootprintCtrl::ModifySource::NONE);

    clsHeavyFootprintCtrl.def("getModifySource", &HeavyFootprintCtrl::getModifySource);
    clsHeavyFootprintCtrl.def("setModifySource", &HeavyFootprintCtrl::setModifySource);
    clsHeavyFootprintCtrl.def("getImageVal", &HeavyFootprintCtrl::getImageVal);
    clsHeavyFootprintCtrl.def("setImageVal", &HeavyFootprintCtrl::setImageVal);
    clsHeavyFootprintCtrl.def("getMaskVal", &HeavyFootprintCtrl::getMaskVal);
    clsHeavyFootprintCtrl.def("setMaskVal", &HeavyFootprintCtrl::setMaskVal);
    clsHeavyFootprintCtrl.def("getVarianceVal", &HeavyFootprintCtrl::getVarianceVal);
    clsHeavyFootprintCtrl.def("setVarianceVal", &HeavyFootprintCtrl::setVarianceVal);

    /* Module level */

    /* Member types and enums */
}
}
}
}  // lsst::afw::detection
