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

#include "lsst/afw/table/Simple.h"
#include "lsst/afw/table/Schema.h"
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
    clsSourceTable.def_static("make", (PTR(SourceTable) (*)(Schema const &, PTR(IdFactory) const &))
        &SourceTable::make);
    clsSourceTable.def_static("make", (PTR(SourceTable) (*)(Schema const &)) &SourceTable::make);
    clsSourceTable.def_static("makeMinimalSchema", &SourceTable::makeMinimalSchema);
    clsSourceTable.def("copyRecord", (PTR(SourceRecord) (SourceTable::*)(BaseRecord const &)) & SourceTable::copyRecord);
    clsSourceTable.def("copyRecord", (PTR(SourceRecord) (SourceTable::*)(BaseRecord const &, SchemaMapper const &)) & SourceTable::copyRecord);
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

    //clsSourceRecord.def("getParent", &SourceTable::getParent);
    //clsSourceRecord.def("setParent", &SourceTable::setParent, "id"_a);

    //clsSourceRecord.def("getPsfFlux", &SourceTable::getPsfFlux);
    //clsSourceRecord.def("getPsfFluxErr", &SourceTable::getPsfFluxErr);
    //clsSourceRecord.def("getPsfFluxFlag", &SourceTable::getPsfFluxFlag);

    //clsSourceRecord.def("getModelFlux", &SourceTable::getModelFlux);
    //clsSourceRecord.def("getModelFluxErr", &SourceTable::getModelFluxErr);
    //clsSourceRecord.def("getModelFluxFlag", &SourceTable::getModelFluxFlag);

    //clsSourceRecord.def("getApFlux", &SourceTable::getApFlux);
    //clsSourceRecord.def("getApFluxErr", &SourceTable::getApFluxErr);
    //clsSourceRecord.def("getApFluxFlag", &SourceTable::getApFluxFlag);

    //clsSourceRecord.def("getInstFlux", &SourceTable::getInstFlux);
    //clsSourceRecord.def("getInstFluxErr", &SourceTable::getInstFluxErr);
    //clsSourceRecord.def("getInstFluxFlag", &SourceTable::getInstFluxFlag);

    //clsSourceRecord.def("getCalibFlux", &SourceTable::getCalibFlux);
    //clsSourceRecord.def("getCalibFluxErr", &SourceTable::getCalibFluxErr);
    //clsSourceRecord.def("getCalibFluxFlag", &SourceTable::getCalibFluxFlag);

    //clsSourceRecord.def("getCentroid", &SourceTable::getCentroid);
    //clsSourceRecord.def("getCentroidErr", &SourceTable::getCentroidErr);
    //clsSourceRecord.def("getCentroidFlag", &SourceTable::getCentroidFlag);

    //clsSourceRecord.def("getShape", &SourceTable::getShape);
    //clsSourceRecord.def("getShapeErr", &SourceTable::getShapeErr);
    //clsSourceRecord.def("getShapeFlag", &SourceTable::getShapeFlag);

    //clsSourceRecord.def("getX", &SourceTable::getX);
    //clsSourceRecord.def("getY", &SourceTable::getY);
    //clsSourceRecord.def("getIxx", &SourceTable::getIxx);
    //clsSourceRecord.def("getIyy", &SourceTable::getIyy);
    //clsSourceRecord.def("getIxy", &SourceTable::getIxy);

    return mod.ptr();
}

}}}  // namespace lsst::afw::table
