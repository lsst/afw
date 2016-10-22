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

namespace py = pybind11;
using namespace pybind11::literals;

#include "lsst/afw/table/Catalog.h"
#include "lsst/afw/table/AmpInfo.h"
#include "lsst/afw/table/pybind11/catalog.h"

namespace lsst {
namespace afw {
namespace table {

PYBIND11_PLUGIN(_ampInfo) {
    py::module mod("_ampInfo", "Python wrapper for afw _ampInfo library");

    /* Module level */
    py::class_<AmpInfoTable, std::shared_ptr<AmpInfoTable>, BaseTable> clsAmpInfoTable(mod, "AmpInfoTable");
    py::class_<AmpInfoRecord, std::shared_ptr<AmpInfoRecord>, BaseRecord>
        clsAmpInfoRecord(mod, "AmpInfoRecord");
    py::class_<CatalogT<AmpInfoRecord>, std::shared_ptr<AmpInfoCatalog>>
        clsAmpInfoCatalog(mod, "AmpInfoCatalog");

    /* Member types and enums */
    py::enum_<ReadoutCorner>(mod, "ReadoutCorner")
        .value("LL", ReadoutCorner::LL)
        .value("LR", ReadoutCorner::LR)
        .value("UR", ReadoutCorner::UR)
        .value("UL", ReadoutCorner::UL)
        .export_values();

    /* Constructors */

    /* Operators */

    /* Members */
    clsAmpInfoRecord.def("getName", &AmpInfoRecord::getName);
    clsAmpInfoRecord.def("setName", &AmpInfoRecord::setName, "name"_a,
                         "Set name of amplifier location in camera");
    clsAmpInfoRecord.def("getBBox", &AmpInfoRecord::getBBox);
    clsAmpInfoRecord.def("setBBox", &AmpInfoRecord::setBBox, "bbox"_a);
    clsAmpInfoRecord.def("getGain", &AmpInfoRecord::getGain);
    clsAmpInfoRecord.def("setGain", &AmpInfoRecord::setGain, "gain"_a, "Set amplifier gain in e-/ADU");
    clsAmpInfoRecord.def("getReadNoise", &AmpInfoRecord::getReadNoise);
    clsAmpInfoRecord.def("setReadNoise", &AmpInfoRecord::setReadNoise, "readNoise"_a, "Set read noise in e-");
    clsAmpInfoRecord.def("getSaturation", &AmpInfoRecord::getSaturation);
    clsAmpInfoRecord.def("setSaturation", &AmpInfoRecord::setSaturation, "saturation"_a,
                         "Set level in ADU above which pixels are considered saturated; "
                          "use `nan` if no such level applies");
    clsAmpInfoRecord.def("getSuspectLevel", &AmpInfoRecord::getSuspectLevel);
    clsAmpInfoRecord.def("setSuspectLevel", &AmpInfoRecord::setSuspectLevel, "suspectLevel"_a,
                         "Set level in ADU above which pixels are considered suspicious; "
                          "use `nan` if no such level applies");
    clsAmpInfoRecord.def("getReadoutCorner", &AmpInfoRecord::getReadoutCorner);
    clsAmpInfoRecord.def("setReadoutCorner", &AmpInfoRecord::setReadoutCorner, "corner"_a);
    clsAmpInfoRecord.def("getLinearityCoeffs", &AmpInfoRecord::getLinearityCoeffs);
    clsAmpInfoRecord.def("setLinearityCoeffs", &AmpInfoRecord::setLinearityCoeffs, "coeffs"_a);
    clsAmpInfoRecord.def("getLinearityType", &AmpInfoRecord::getLinearityType);
    clsAmpInfoRecord.def("setLinearityType", &AmpInfoRecord::setLinearityType, "type"_a);
    clsAmpInfoRecord.def("getHasRawInfo", &AmpInfoRecord::getHasRawInfo);
    clsAmpInfoRecord.def("setHasRawInfo", &AmpInfoRecord::setHasRawInfo, "hasRawInfo"_a);
    clsAmpInfoRecord.def("getRawBBox", &AmpInfoRecord::getRawBBox);
    clsAmpInfoRecord.def("setRawBBox", &AmpInfoRecord::setRawBBox, "bbox"_a);
    clsAmpInfoRecord.def("getRawDataBBox", &AmpInfoRecord::getRawDataBBox);
    clsAmpInfoRecord.def("setRawDataBBox", &AmpInfoRecord::setRawDataBBox, "bbox"_a);
    clsAmpInfoRecord.def("getRawFlipX", &AmpInfoRecord::getRawFlipX);
    clsAmpInfoRecord.def("setRawFlipX", &AmpInfoRecord::setRawFlipX, "rawFlipX"_a);
    clsAmpInfoRecord.def("getRawFlipY", &AmpInfoRecord::getRawFlipY);
    clsAmpInfoRecord.def("setRawFlipY", &AmpInfoRecord::setRawFlipY, "rawFlipY"_a);
    clsAmpInfoRecord.def("getRawXYOffset", &AmpInfoRecord::getRawXYOffset);
    clsAmpInfoRecord.def("setRawXYOffset", &AmpInfoRecord::setRawXYOffset, "offset"_a);
    clsAmpInfoRecord.def("getRawHorizontalOverscanBBox", &AmpInfoRecord::getRawHorizontalOverscanBBox);
    clsAmpInfoRecord.def("setRawHorizontalOverscanBBox", &AmpInfoRecord::setRawHorizontalOverscanBBox,
                         "bbox"_a);
    clsAmpInfoRecord.def("getRawVerticalOverscanBBox", &AmpInfoRecord::getRawVerticalOverscanBBox);
    clsAmpInfoRecord.def("setRawVerticalOverscanBBox", &AmpInfoRecord::setRawVerticalOverscanBBox, "bbox"_a);
    clsAmpInfoRecord.def("getRawPrescanBBox", &AmpInfoRecord::getRawPrescanBBox);
    clsAmpInfoRecord.def("setRawPrescanBBox", &AmpInfoRecord::setRawPrescanBBox, "bbox"_a);

    clsAmpInfoTable.def_static("make", &AmpInfoTable::make);
    clsAmpInfoTable.def_static("makeMinimalSchema", &AmpInfoTable::makeMinimalSchema);
    // clsAmpInfoTable.def_static("checkSchema", &AmpInfoTable::checkSchema);
    // clsAmpInfoTable.def_static("getNameKey", &AmpInfoTable::getNameKey);
    // clsAmpInfoTable.def_static("getBBoxMinKey", &AmpInfoTable::getBBoxMinKey);
    // clsAmpInfoTable.def_static("getBBoxExtentKey", &AmpInfoTable::getBBoxExtentKey);
    // clsAmpInfoTable.def_static("getGainKey", &AmpInfoTable::getGainKey);
    // clsAmpInfoTable.def_static("getReadNoiseKey", &AmpInfoTable::getReadNoiseKey);
    // clsAmpInfoTable.def_static("getSaturationKey", &AmpInfoTable::getSaturationKey);
    // clsAmpInfoTable.def_static("getSuspectLevelKey", &AmpInfoTable::getSuspectLevelKey);
    // clsAmpInfoTable.def_static("getReadoutCornerKey", &AmpInfoTable::getReadoutCornerKey);
    // clsAmpInfoTable.def_static("getLinearityCoeffsKey", &AmpInfoTable::getLinearityCoeffsKey);
    // clsAmpInfoTable.def_static("getLinearityTypeKey", &AmpInfoTable::getLinearityTypeKey);
    // clsAmpInfoTable.def_static("getHasRawInfoKey", &AmpInfoTable::getHasRawInfoKey);
    // clsAmpInfoTable.def_static("getRawBBoxMinKey", &AmpInfoTable::getRawBBoxMinKey);
    // clsAmpInfoTable.def_static("getRawBBoxExtentKey", &AmpInfoTable::getRawBBoxExtentKey);
    // clsAmpInfoTable.def_static("getRawDataBBoxMinKey", &AmpInfoTable::getRawDataBBoxMinKey);
    // clsAmpInfoTable.def_static("getRawDataBBoxExtentKey", &AmpInfoTable::getRawDataBBoxExtentKey);
    // clsAmpInfoTable.def_static("getRawFlipXKey", &AmpInfoTable::getRawFlipXKey);
    // clsAmpInfoTable.def_static("getRawFlipYKey", &AmpInfoTable::getRawFlipYKey);
    // clsAmpInfoTable.def_static("getRawXYOffsetKey", &AmpInfoTable::getRawXYOffsetKey);
    // clsAmpInfoTable.def_static("getRawHorizontalOverscanBBoxMinKey",
    //                            &AmpInfoTable::getRawHorizontalOverscanBBoxMinKey);
    // clsAmpInfoTable.def_static("getRawHorizontalOverscanBBoxExtentKey",
    //                            &AmpInfoTable::getRawHorizontalOverscanBBoxExtentKey);
    // clsAmpInfoTable.def_static("getRawVerticalOverscanBBoxMinKey",
    //                            &AmpInfoTable::getRawVerticalOverscanBBoxMinKey);
    // clsAmpInfoTable.def_static("getRawVerticalOverscanBBoxExtentKey",
    //                            &AmpInfoTable::getRawVerticalOverscanBBoxExtentKey);
    // clsAmpInfoTable.def_static("getRawPrescanBBoxMinKey", &AmpInfoTable::getRawPrescanBBoxMinKey);
    // clsAmpInfoTable.def_static("getRawPrescanBBoxExtentKey", &AmpInfoTable::getRawPrescanBBoxExtentKey);

    // clsAmpInfoTable.def("clone", &AmpInfoTable::clone);
    clsAmpInfoTable.def("makeRecord", &AmpInfoTable::makeRecord);
    // clsAmpInfoTable.def("copyRecord", (PTR(AmpInfoRecord) (AmpInfoTable::*)(BaseRecord const &)) & AmpInfoTable::copyRecord);
    // clsAmpInfoTable.def("copyRecord", (PTR(AmpInfoRecord) (AmpInfoTable::*)(BaseRecord const &, SchemaMapper const &)) & AmpInfoTable::copyRecord);

    declareCatalog<AmpInfoRecord>(clsAmpInfoCatalog);

    return mod.ptr();
}

}}}  // namespace lsst::afw::table
