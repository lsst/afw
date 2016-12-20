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
#include <pybind11/stl.h>

#include "lsst/afw/table/Simple.h"
#include "lsst/afw/table/Schema.h"
#include "lsst/afw/table/slots.h"
#include "lsst/afw/table/Source.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst {
namespace afw {
namespace table {

template <typename RecordT>
void declareSourceColumnView(py::module & mod) {
    py::class_<SourceColumnViewT<RecordT>, std::shared_ptr<SourceColumnViewT<RecordT>>>
        cls(mod, "SourceColumnViewT");
    //cls.def_static("make", &SourceColumnViewT<RecordT>::make);
};

PYBIND11_PLUGIN(_source) {
    py::module mod("_source", "Python wrapper for afw _source library");

    /* Module level */
    py::class_<SourceTable, std::shared_ptr<SourceTable>, SimpleTable>
        clsSourceTable(mod, "SourceTable");
    py::class_<SourceRecord, std::shared_ptr<SourceRecord>, SimpleRecord>
        clsSourceRecord(mod, "SourceRecord");

    /* Member types and enums */

    /* Constructors */

    /* Operators */

    /* Members */
    clsSourceTable.def_static("make",
                              (std::shared_ptr<SourceTable> (*)(Schema const &,
                                                                std::shared_ptr<IdFactory> const &))
                                    &SourceTable::make);
    clsSourceTable.def_static("make",
                              (std::shared_ptr<SourceTable> (*)(Schema const &)) &SourceTable::make);
    clsSourceTable.def_static("makeMinimalSchema", &SourceTable::makeMinimalSchema);
    clsSourceTable.def("copyRecord",
                       (std::shared_ptr<SourceRecord> (SourceTable::*)(BaseRecord const &))
                            &SourceTable::copyRecord);
    clsSourceTable.def("copyRecord",
                       (std::shared_ptr<SourceRecord> (SourceTable::*)(BaseRecord const &,
                                                                       SchemaMapper const &))
                            &SourceTable::copyRecord);
    clsSourceTable.def("makeRecord", &SourceTable::makeRecord);

    //clsSourceTable.def("getPsfFluxSlot", &SourceTable::getPsfFluxSlot);
    clsSourceTable.def("definePsfFlux", &SourceTable::definePsfFlux, "name"_a);
    //clsSourceTable.def("getPsfFluxDefinition", &SourceTable::getPsfFluxDefinition);
    //clsSourceTable.def("hasPsfFluxSlot", &SourceTable::hasPsfFluxSlot);
    clsSourceTable.def("getPsfFluxKey", &SourceTable::getPsfFluxKey);
    //clsSourceTable.def("getPsfFluxErrKey", &SourceTable::getPsfFluxErrKey);
    //clsSourceTable.def("getPsfFluxFlagKey", &SourceTable::getPsfFluxFlagKey);

    //clsSourceTable.def("getModelFluxSlot", &SourceTable::getModelFluxSlot);
    //clsSourceTable.def("defineModelFlux", &SourceTable::defineModelFlux, "name"_a);
    //clsSourceTable.def("getModelFluxDefinition", &SourceTable::getModelFluxDefinition);
    //clsSourceTable.def("hasModelFluxSlot", &SourceTable::hasModelFluxSlot);
    //clsSourceTable.def("getModelFluxKey", &SourceTable::getModelFluxKey);
    //clsSourceTable.def("getModelFluxErrKey", &SourceTable::getModelFluxErrKey);
    //clsSourceTable.def("getModelFluxFlagKey", &SourceTable::getModelFluxFlagKey);

    //clsSourceTable.def("getApFluxSlot", &SourceTable::getApFluxSlot);
    //clsSourceTable.def("defineApFlux", &SourceTable::defineApFlux, "name"_a);
    //clsSourceTable.def("getApFluxDefinition", &SourceTable::getApFluxDefinition);
    //clsSourceTable.def("hasApFluxSlot", &SourceTable::hasApFluxSlot);
    //clsSourceTable.def("getApFluxKey", &SourceTable::getApFluxKey);
    //clsSourceTable.def("getApFluxErrKey", &SourceTable::getApFluxErrKey);
    //clsSourceTable.def("getApFluxFlagKey", &SourceTable::getApFluxFlagKey);

    //clsSourceTable.def("getInstFluxSlot", &SourceTable::getInstFluxSlot);
    //clsSourceTable.def("defineInstFlux", &SourceTable::defineInstFlux, "name"_a);
    //clsSourceTable.def("getInstFluxDefinition", &SourceTable::getInstFluxDefinition);
    //clsSourceTable.def("hasInstFluxSlot", &SourceTable::hasInstFluxSlot);
    //clsSourceTable.def("getInstFluxKey", &SourceTable::getInstFluxKey);
    //clsSourceTable.def("getInstFluxErrKey", &SourceTable::getInstFluxErrKey);
    //clsSourceTable.def("getInstFluxFlagKey", &SourceTable::getInstFluxFlagKey);

    //clsSourceTable.def("getCalibFluxSlot", &SourceTable::getCalibFluxSlot);
    //clsSourceTable.def("defineCalibFlux", &SourceTable::defineCalibFlux, "name"_a);
    //clsSourceTable.def("getCalibFluxDefinition", &SourceTable::getCalibFluxDefinition);
    //clsSourceTable.def("hasCalibFluxSlot", &SourceTable::hasCalibFluxSlot);
    //clsSourceTable.def("getCalibFluxKey", &SourceTable::getCalibFluxKey);
    //clsSourceTable.def("getCalibFluxErrKey", &SourceTable::getCalibFluxErrKey);
    //clsSourceTable.def("getCalibFluxFlagKey", &SourceTable::getCalibFluxFlagKey);

    //clsSourceTable.def("getCentroidSlot", &SourceTable::getCentroidSlot);
    clsSourceTable.def("defineCentroid", &SourceTable::defineCentroid, "name"_a);
    //clsSourceTable.def("getCentroidDefinition", &SourceTable::getCentroidDefinition);
    //clsSourceTable.def("hasCentroidSlot", &SourceTable::hasCentroidSlot);
    //clsSourceTable.def("getCentroidKey", &SourceTable::getCentroidKey);
    //clsSourceTable.def("getCentroidErrKey", &SourceTable::getCentroidErrKey);
    //clsSourceTable.def("getCentroidFlagKey", &SourceTable::getCentroidFlagKey);

    //clsSourceTable.def("getShapeSlot", &SourceTable::getShapeSlot);
    //clsSourceTable.def("defineShape", &SourceTable::defineShape, "name"_a);
    //clsSourceTable.def("getShapeDefinition", &SourceTable::getShapeDefinition);
    //clsSourceTable.def("hasShapeSlot", &SourceTable::hasShapeSlot);
    //clsSourceTable.def("getShapeKey", &SourceTable::getShapeKey);
    //clsSourceTable.def("getShapeErrKey", &SourceTable::getShapeErrKey);
    //clsSourceTable.def("getShapeFlagKey", &SourceTable::getShapeFlagKey);

    declareSourceColumnView<SourceRecord>(mod);

    clsSourceRecord.def("getFootprint", &SourceRecord::getFootprint);
    clsSourceRecord.def("setFootprint", &SourceRecord::setFootprint);
    clsSourceRecord.def("getTable", &SourceRecord::getTable);

    //clsSourceRecord.def("getParent", &SourceRecord::getParent);
    //clsSourceRecord.def("setParent", &SourceRecord::setParent, "id"_a);

    //clsSourceRecord.def("getPsfFlux", &SourceRecord::getPsfFlux);
    //clsSourceRecord.def("getPsfFluxErr", &SourceRecord::getPsfFluxErr);
    //clsSourceRecord.def("getPsfFluxFlag", &SourceRecord::getPsfFluxFlag);

    //clsSourceRecord.def("getModelFlux", &SourceRecord::getModelFlux);
    //clsSourceRecord.def("getModelFluxErr", &SourceRecord::getModelFluxErr);
    //clsSourceRecord.def("getModelFluxFlag", &SourceRecord::getModelFluxFlag);

    //clsSourceRecord.def("getApFlux", &SourceRecord::getApFlux);
    //clsSourceRecord.def("getApFluxErr", &SourceRecord::getApFluxErr);
    //clsSourceRecord.def("getApFluxFlag", &SourceRecord::getApFluxFlag);

    //clsSourceRecord.def("getInstFlux", &SourceRecord::getInstFlux);
    //clsSourceRecord.def("getInstFluxErr", &SourceRecord::getInstFluxErr);
    //clsSourceRecord.def("getInstFluxFlag", &SourceRecord::getInstFluxFlag);

    //clsSourceRecord.def("getCalibFlux", &SourceRecord::getCalibFlux);
    //clsSourceRecord.def("getCalibFluxErr", &SourceRecord::getCalibFluxErr);
    //clsSourceRecord.def("getCalibFluxFlag", &SourceRecord::getCalibFluxFlag);

    clsSourceRecord.def("getCentroid", &SourceRecord::getCentroid);
    clsSourceRecord.def("getCentroidErr", &SourceRecord::getCentroidErr);
    //clsSourceRecord.def("getCentroidFlag", &SourceRecord::getCentroidFlag);

    //clsSourceRecord.def("getShape", &SourceRecord::getShape);
    //clsSourceRecord.def("getShapeErr", &SourceRecord::getShapeErr);
    //clsSourceRecord.def("getShapeFlag", &SourceRecord::getShapeFlag);

    //clsSourceRecord.def("getX", &SourceRecord::getX);
    //clsSourceRecord.def("getY", &SourceRecord::getY);
    //clsSourceRecord.def("getIxx", &SourceRecord::getIxx);
    //clsSourceRecord.def("getIyy", &SourceRecord::getIyy);
    //clsSourceRecord.def("getIxy", &SourceRecord::getIxy);

    return mod.ptr();
}

}}}  // namespace lsst::afw::table
