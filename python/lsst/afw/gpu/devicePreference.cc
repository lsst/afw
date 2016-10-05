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

#include "lsst/afw/gpu/DevicePreference.h"

namespace py = pybind11;

using namespace lsst::afw::gpu;

PYBIND11_PLUGIN(_devicePreference) {
    py::module mod("_devicePreference", "Python wrapper for afw _devicePreference library");

    py::enum_<DevicePreference>(mod, "DevicePreference")
        .value("AUTO", DevicePreference::AUTO)
        .value("AUTO_WITH_CPU_FALLBACK", DevicePreference::AUTO_WITH_CPU_FALLBACK)
        .value("USE_CPU", DevicePreference::USE_CPU)
        .value("USE_GPU", DevicePreference::USE_GPU)
        .export_values();

    mod.attr("DEFAULT_DEVICE_PREFERENCE") = py::cast(AUTO);

    /* Module level */

    /* Member types and enums */

    /* Constructors */

    /* Operators */

    /* Members */

    return mod.ptr();
}