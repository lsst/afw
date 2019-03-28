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
#include "pybind11/stl.h"

#include "ndarray/pybind11.h"

#include "lsst/afw/table/BaseRecord.h"
#include "lsst/afw/cameraGeom/Amplifier.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst {
namespace afw {
namespace cameraGeom {
namespace {

using PyAmplifier = py::class_<Amplifier, std::shared_ptr<Amplifier>>;

PYBIND11_MODULE(amplifier, mod) {
    py::enum_<ReadoutCorner>(mod, "ReadoutCorner")
        .value("LL", ReadoutCorner::LL)
        .value("LR", ReadoutCorner::LR)
        .value("UR", ReadoutCorner::UR)
        .value("UL", ReadoutCorner::UL);

    PyAmplifier cls(mod, "Amplifier");
    cls.def_static("getRecordSchema", &Amplifier::getRecordSchema);
    cls.def_static("fromRecord", &Amplifier::fromRecord);
    cls.def(py::init());
    cls.def("toRecord", &Amplifier::toRecord);
    cls.def("getName", &Amplifier::getName);
    cls.def("setName", &Amplifier::setName, "name"_a);
    cls.def("getBBox", &Amplifier::getBBox);
    cls.def("setBBox", &Amplifier::setBBox, "bbox"_a);
    cls.def("getGain", &Amplifier::getGain);
    cls.def("setGain", &Amplifier::setGain, "gain"_a);
    cls.def("getReadNoise", &Amplifier::getReadNoise);
    cls.def("setReadNoise", &Amplifier::setReadNoise, "readNoise"_a);
    cls.def("getSaturation", &Amplifier::getSaturation);
    cls.def("setSaturation", &Amplifier::setSaturation, "saturation"_a);
    cls.def("getSuspectLevel", &Amplifier::getSuspectLevel);
    cls.def("setSuspectLevel", &Amplifier::setSuspectLevel, "suspectLevel"_a);
    cls.def("getReadoutCorner", &Amplifier::getReadoutCorner);
    cls.def("setReadoutCorner", &Amplifier::setReadoutCorner, "corner"_a);
    cls.def("getLinearityCoeffs", &Amplifier::getLinearityCoeffs);
    cls.def("setLinearityCoeffs", &Amplifier::setLinearityCoeffs, "coeffs"_a);
    cls.def("getLinearityType", &Amplifier::getLinearityType);
    cls.def("setLinearityType", &Amplifier::setLinearityType, "type"_a);
    cls.def("getLinearityThreshold", &Amplifier::getLinearityThreshold);
    cls.def("setLinearityThreshold", &Amplifier::setLinearityThreshold, "threshold"_a);
    cls.def("getLinearityMaximum", &Amplifier::getLinearityMaximum);
    cls.def("setLinearityMaximum", &Amplifier::setLinearityMaximum, "maximum"_a);
    cls.def("getLinearityUnits", &Amplifier::getLinearityUnits);
    cls.def("setLinearityUnits", &Amplifier::setLinearityUnits, "units"_a);
    cls.def("getHasRawInfo", &Amplifier::getHasRawInfo);  // TODO: deprecate
    cls.def("getRawBBox", &Amplifier::getRawBBox);
    cls.def("setRawBBox", &Amplifier::setRawBBox, "bbox"_a);
    cls.def("getRawDataBBox", &Amplifier::getRawDataBBox);
    cls.def("setRawDataBBox", &Amplifier::setRawDataBBox, "bbox"_a);
    cls.def("getRawFlipX", &Amplifier::getRawFlipX);
    cls.def("setRawFlipX", &Amplifier::setRawFlipX, "rawFlipX"_a);
    cls.def("getRawFlipY", &Amplifier::getRawFlipY);
    cls.def("setRawFlipY", &Amplifier::setRawFlipY, "rawFlipY"_a);
    cls.def("getRawXYOffset", &Amplifier::getRawXYOffset);
    cls.def("setRawXYOffset", &Amplifier::setRawXYOffset, "offset"_a);
    cls.def("getRawHorizontalOverscanBBox", &Amplifier::getRawHorizontalOverscanBBox);
    cls.def("setRawHorizontalOverscanBBox", &Amplifier::setRawHorizontalOverscanBBox, "bbox"_a);
    cls.def("getRawVerticalOverscanBBox", &Amplifier::getRawVerticalOverscanBBox);
    cls.def("setRawVerticalOverscanBBox", &Amplifier::setRawVerticalOverscanBBox, "bbox"_a);
    cls.def("getRawPrescanBBox", &Amplifier::getRawPrescanBBox);
    cls.def("setRawPrescanBBox", &Amplifier::setRawPrescanBBox, "bbox"_a);
}

}  // namespace
}  // namespace cameraGeom
}  // namespace afw
}  // namespace lsst
