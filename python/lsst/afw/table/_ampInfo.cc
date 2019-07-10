/*
 * This file is part of afw.
 *
 * Developed for the LSST Data Management System.
 * This product includes software developed by the LSST Project
 * (https://www.lsst.org).
 * See the COPYRIGHT file at the top-level directory of this distribution
 * for details of code ownership.
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
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include <memory>

#include "lsst/utils/python.h"

#include "lsst/afw/table/Catalog.h"
#include "lsst/afw/table/AmpInfo.h"
#include "lsst/afw/table/python/catalog.h"
#include "lsst/afw/table/python/columnView.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst {
namespace afw {
namespace table {

using utils::python::WrapperCollection;

namespace {

using PyAmpInfoRecord = py::class_<AmpInfoRecord, std::shared_ptr<AmpInfoRecord>, BaseRecord>;
using PyAmpInfoTable = py::class_<AmpInfoTable, std::shared_ptr<AmpInfoTable>, BaseTable>;

static PyAmpInfoRecord declareAmpInfoRecord(WrapperCollection &wrappers) {
    return wrappers.wrapType(PyAmpInfoRecord(wrappers.module, "AmpInfoRecord"), [](auto &mod, auto &cls) {
        cls.def("getName", &AmpInfoRecord::getName);
        cls.def("setName", &AmpInfoRecord::setName, "name"_a, "Set name of amplifier location in camera");
        cls.def("getTable", &AmpInfoRecord::getTable);
        cls.def_property_readonly("table", &AmpInfoRecord::getTable);
        cls.def("getBBox", &AmpInfoRecord::getBBox);
        cls.def("setBBox", &AmpInfoRecord::setBBox, "bbox"_a);
        cls.def("getGain", &AmpInfoRecord::getGain);
        cls.def("setGain", &AmpInfoRecord::setGain, "gain"_a, "Set amplifier gain in electron/adu");
        cls.def("getReadNoise", &AmpInfoRecord::getReadNoise);
        cls.def("setReadNoise", &AmpInfoRecord::setReadNoise, "readNoise"_a, "Set read noise in electron");
        cls.def("getSaturation", &AmpInfoRecord::getSaturation);
        cls.def("setSaturation", &AmpInfoRecord::setSaturation, "saturation"_a,
                "Set level in ADU above which pixels are considered saturated; "
                "use `nan` if no such level applies");
        cls.def("getSuspectLevel", &AmpInfoRecord::getSuspectLevel);
        cls.def("setSuspectLevel", &AmpInfoRecord::setSuspectLevel, "suspectLevel"_a,
                "Set level in ADU above which pixels are considered suspicious; "
                "use `nan` if no such level applies");
        cls.def("getReadoutCorner", &AmpInfoRecord::getReadoutCorner);
        cls.def("setReadoutCorner", &AmpInfoRecord::setReadoutCorner, "corner"_a);
        cls.def("getLinearityCoeffs", &AmpInfoRecord::getLinearityCoeffs);
        cls.def("setLinearityCoeffs", &AmpInfoRecord::setLinearityCoeffs, "coeffs"_a);
        cls.def("getLinearityType", &AmpInfoRecord::getLinearityType);
        cls.def("setLinearityType", &AmpInfoRecord::setLinearityType, "type"_a);
        cls.def("getHasRawInfo", &AmpInfoRecord::getHasRawInfo);
        cls.def("setHasRawInfo", &AmpInfoRecord::setHasRawInfo, "hasRawInfo"_a);
        cls.def("getRawBBox", &AmpInfoRecord::getRawBBox);
        cls.def("setRawBBox", &AmpInfoRecord::setRawBBox, "bbox"_a);
        cls.def("getRawDataBBox", &AmpInfoRecord::getRawDataBBox);
        cls.def("setRawDataBBox", &AmpInfoRecord::setRawDataBBox, "bbox"_a);
        cls.def("getRawFlipX", &AmpInfoRecord::getRawFlipX);
        cls.def("setRawFlipX", &AmpInfoRecord::setRawFlipX, "rawFlipX"_a);
        cls.def("getRawFlipY", &AmpInfoRecord::getRawFlipY);
        cls.def("setRawFlipY", &AmpInfoRecord::setRawFlipY, "rawFlipY"_a);
        cls.def("getRawXYOffset", &AmpInfoRecord::getRawXYOffset);
        cls.def("setRawXYOffset", &AmpInfoRecord::setRawXYOffset, "offset"_a);
        cls.def("getRawHorizontalOverscanBBox", &AmpInfoRecord::getRawHorizontalOverscanBBox);
        cls.def("setRawHorizontalOverscanBBox", &AmpInfoRecord::setRawHorizontalOverscanBBox, "bbox"_a);
        cls.def("getRawVerticalOverscanBBox", &AmpInfoRecord::getRawVerticalOverscanBBox);
        cls.def("setRawVerticalOverscanBBox", &AmpInfoRecord::setRawVerticalOverscanBBox, "bbox"_a);
        cls.def("getRawPrescanBBox", &AmpInfoRecord::getRawPrescanBBox);
        cls.def("setRawPrescanBBox", &AmpInfoRecord::setRawPrescanBBox, "bbox"_a);
    });
}

static PyAmpInfoTable declareAmpInfoTable(WrapperCollection &wrappers) {
    return wrappers.wrapType(PyAmpInfoTable(wrappers.module, "AmpInfoTable"), [](auto &mod, auto &cls) {
        cls.def_static("make", &AmpInfoTable::make);
        cls.def_static("makeMinimalSchema", &AmpInfoTable::makeMinimalSchema);
        cls.def_static("checkSchema", &AmpInfoTable::checkSchema, "other"_a);
        cls.def_static("getNameKey", &AmpInfoTable::getNameKey);
        cls.def_static("getBBoxMinKey", &AmpInfoTable::getBBoxMinKey);
        cls.def_static("getBBoxExtentKey", &AmpInfoTable::getBBoxExtentKey);
        cls.def_static("getGainKey", &AmpInfoTable::getGainKey);
        cls.def_static("getReadNoiseKey", &AmpInfoTable::getReadNoiseKey);
        cls.def_static("getSaturationKey", &AmpInfoTable::getSaturationKey);
        cls.def_static("getSuspectLevelKey", &AmpInfoTable::getSuspectLevelKey);
        cls.def_static("getReadoutCornerKey", &AmpInfoTable::getReadoutCornerKey);
        cls.def_static("getLinearityCoeffsKey", &AmpInfoTable::getLinearityCoeffsKey);
        cls.def_static("getLinearityTypeKey", &AmpInfoTable::getLinearityTypeKey);
        cls.def_static("getHasRawInfoKey", &AmpInfoTable::getHasRawInfoKey);
        cls.def_static("getRawBBoxMinKey", &AmpInfoTable::getRawBBoxMinKey);
        cls.def_static("getRawBBoxExtentKey", &AmpInfoTable::getRawBBoxExtentKey);
        cls.def_static("getRawDataBBoxMinKey", &AmpInfoTable::getRawDataBBoxMinKey);
        cls.def_static("getRawDataBBoxExtentKey", &AmpInfoTable::getRawDataBBoxExtentKey);
        cls.def_static("getRawFlipXKey", &AmpInfoTable::getRawFlipXKey);
        cls.def_static("getRawFlipYKey", &AmpInfoTable::getRawFlipYKey);
        cls.def_static("getRawXYOffsetKey", &AmpInfoTable::getRawXYOffsetKey);
        cls.def_static("getRawHorizontalOverscanBBoxMinKey",
                       &AmpInfoTable::getRawHorizontalOverscanBBoxMinKey);
        cls.def_static("getRawHorizontalOverscanBBoxExtentKey",
                       &AmpInfoTable::getRawHorizontalOverscanBBoxExtentKey);
        cls.def_static("getRawVerticalOverscanBBoxMinKey", &AmpInfoTable::getRawVerticalOverscanBBoxMinKey);
        cls.def_static("getRawVerticalOverscanBBoxExtentKey",
                       &AmpInfoTable::getRawVerticalOverscanBBoxExtentKey);
        cls.def_static("getRawPrescanBBoxMinKey", &AmpInfoTable::getRawPrescanBBoxMinKey);
        cls.def_static("getRawPrescanBBoxExtentKey", &AmpInfoTable::getRawPrescanBBoxExtentKey);

        cls.def("clone", &AmpInfoTable::clone);
        cls.def("makeRecord", &AmpInfoTable::makeRecord);
        cls.def("copyRecord",
                (std::shared_ptr<AmpInfoRecord>(AmpInfoTable::*)(BaseRecord const &)) &
                        AmpInfoTable::copyRecord,
                "other"_a);
        cls.def("copyRecord",
                (std::shared_ptr<AmpInfoRecord>(AmpInfoTable::*)(BaseRecord const &, SchemaMapper const &)) &
                        AmpInfoTable::copyRecord,
                "other"_a, "mapper"_a);
    });
}

}  // namespace

void wrapAmpInfo(WrapperCollection &wrappers) {
    wrappers.wrapType(py::enum_<ReadoutCorner>(wrappers.module, "ReadoutCorner"), [](auto &mod, auto &enm) {
        enm.value("LL", ReadoutCorner::LL);
        enm.value("LR", ReadoutCorner::LR);
        enm.value("UR", ReadoutCorner::UR);
        enm.value("UL", ReadoutCorner::UL);
        enm.export_values();
    });

    auto clsAmpInfoRecord = declareAmpInfoRecord(wrappers);
    auto clsAmpInfoTable = declareAmpInfoTable(wrappers);
    auto clsAmpInfoColumnView = table::python::declareColumnView<AmpInfoRecord>(wrappers, "AmpInfo");
    auto clsAmpInfoCatalog = table::python::declareCatalog<AmpInfoRecord>(wrappers, "AmpInfo");

    clsAmpInfoRecord.attr("Table") = clsAmpInfoTable;
    clsAmpInfoRecord.attr("ColumnView") = clsAmpInfoColumnView;
    clsAmpInfoRecord.attr("Catalog") = clsAmpInfoCatalog;
    clsAmpInfoTable.attr("Record") = clsAmpInfoRecord;
    clsAmpInfoTable.attr("ColumnView") = clsAmpInfoColumnView;
    clsAmpInfoTable.attr("Catalog") = clsAmpInfoCatalog;
    clsAmpInfoCatalog.attr("Record") = clsAmpInfoRecord;
    clsAmpInfoCatalog.attr("Table") = clsAmpInfoTable;
    clsAmpInfoCatalog.attr("ColumnView") = clsAmpInfoColumnView;
}

}  // namespace table
}  // namespace afw
}  // namespace lsst
