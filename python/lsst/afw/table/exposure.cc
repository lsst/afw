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

namespace py = pybind11;
using namespace pybind11::literals;

#include "lsst/afw/coord/Coord.h"
#include "lsst/afw/detection/Psf.h"  // forward-declared by Exposure.h
#include "lsst/afw/fits.h"
#include "lsst/afw/geom/Box.h"
#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/polygon/Polygon.h"  // forward-declared by Exposure.h
#include "lsst/afw/image/ApCorrMap.h"  // forward-declared by Exposure.h
#include "lsst/afw/image/Calib.h"  // forward-declared by Exposure.h
#include "lsst/afw/image/VisitInfo.h"  // forward-declared by Exposure.h
#include "lsst/afw/image/Wcs.h"  // forward-declared by Exposure.h
#include "lsst/afw/table/BaseRecord.h"
#include "lsst/afw/table/BaseTable.h"
#include "lsst/afw/table/Catalog.h"
#include "lsst/afw/table/Exposure.h"
#include "lsst/afw/table/pybind11/catalog.h"
#include "lsst/afw/table/pybind11/columnView.h"
#include "lsst/afw/table/pybind11/sortedCatalog.h"

namespace lsst {
namespace afw {
namespace table {

namespace {

using PyExposureRecord = py::class_<ExposureRecord, std::shared_ptr<ExposureRecord>, BaseRecord>;
using PyExposureTable = py::class_<ExposureTable, std::shared_ptr<ExposureTable>, BaseTable>;
using PyExposureColumnView = py::class_<ColumnViewT<ExposureRecord>,
                                      std::shared_ptr<ColumnViewT<ExposureRecord>>,
                                      BaseColumnView>;
using PyBaseExposureCatalog = py::class_<CatalogT<ExposureRecord>,
                                         std::shared_ptr<CatalogT<ExposureRecord>>>;
using PySortedBaseExposureCatalog =  py::class_<SortedCatalogT<ExposureRecord>,
                                                std::shared_ptr<SortedCatalogT<ExposureRecord>>,
                                                CatalogT<ExposureRecord>>;
using PyExposureCatalog = py::class_<ExposureCatalogT<ExposureRecord>,
                                     std::shared_ptr<ExposureCatalogT<ExposureRecord>>,
                                     SortedCatalogT<ExposureRecord>>;

/**
Declare constructors and member and static functions for a pybind11 ExposureRecord
*/
void declareExposureRecord(PyExposureRecord & cls) {
    table::pybind11::addCastFrom<BaseRecord>(cls);

    cls.def("getId", &ExposureRecord::getId);
    cls.def("setId", &ExposureRecord::setId, "id"_a);
    cls.def("getBBox", &ExposureRecord::getBBox);
    cls.def("setBBox", &ExposureRecord::setBBox, "bbox"_a);
    cls.def("getTable", &ExposureRecord::getTable);
    cls.def_property_readonly("table", &ExposureRecord::getTable);

    //cls.def("writeFits",
    //        (void (Catalog::*)(std::string const &, std::string const &, int) const) &Catalog::writeFits,
    //        "filename"_a, "mode"_a="w", "flags"_a = 0);

    cls.def("contains",
            (bool (ExposureRecord::*)(coord::Coord const &, bool) const) &ExposureRecord::contains,
            "coord"_a, "includeValidPolygon"_a=false);
    cls.def("contains",
                          (bool (ExposureRecord::*)(geom::Point2D const &, image::Wcs const &, bool) const)
                           &ExposureRecord::contains, "point"_a, "wcs"_a, "includeValidPolygon"_a=false);
    cls.def("getWcs", &ExposureRecord::getWcs);
    cls.def("setWcs", &ExposureRecord::setWcs, "wcs"_a);
    cls.def("getPsf", &ExposureRecord::getPsf);
    cls.def("setPsf", &ExposureRecord::setPsf, "psf"_a);
    cls.def("getCalib", &ExposureRecord::getCalib);
    cls.def("setCalib", &ExposureRecord::setCalib, "calib"_a);
    cls.def("getApCorrMap", &ExposureRecord::getApCorrMap);
    cls.def("setApCorrMap", &ExposureRecord::setApCorrMap, "appCorrMap"_a);
    cls.def("getValidPolygon", &ExposureRecord::getValidPolygon);
    cls.def("setValidPolygon", &ExposureRecord::setValidPolygon, "polygon"_a);
    cls.def("getVisitInfo", &ExposureRecord::getVisitInfo);
    cls.def("setVisitInfo", &ExposureRecord::setVisitInfo, "visitInfo"_a);
}

/**
Declare constructors and member and static functions for a pybind11 ExposureTable
*/
void declareExposureTable(PyExposureTable & cls) {
    table::pybind11::addCastFrom<BaseTable>(cls);

    cls.def_static("make", &ExposureTable::make);
    cls.def_static("makeMinimalSchema", &ExposureTable::makeMinimalSchema);
    cls.def_static("checkSchema", &ExposureTable::checkSchema, "schema"_a);

    cls.def_static("getIdKey", &ExposureTable::getIdKey);
    cls.def_static("getBBoxMinKey", &ExposureTable::getBBoxMinKey);
    cls.def_static("getBBoxMaxKey", &ExposureTable::getBBoxMaxKey);

    cls.def("clone", &ExposureTable::clone);
    cls.def("makeRecord", &ExposureTable::makeRecord);
    cls.def("copyRecord",
           (std::shared_ptr<ExposureRecord> (ExposureTable::*)(BaseRecord const &))
               &ExposureTable::copyRecord);
    cls.def("copyRecord",
           (std::shared_ptr<ExposureRecord> (ExposureTable::*)(BaseRecord const &, SchemaMapper const &))
               &ExposureTable::copyRecord);
}

/**
Declare constructors and member and static functions for a pybind11 ExposureCatalog

@tparam RecordT  Record type, e.g. BaseRecord or SimpleRecord.

@param[in] cls  Catalog pybind11 class.
*/
void declareExposureCatalog(PyExposureCatalog & cls) {
    using Catalog = ExposureCatalogT<ExposureRecord>;

    // TODO: Commented-out code is waiting until needed and is untested.
    // Add tests for it and enable it or remove it before the final pybind11 merge.

    /* Constructors */
    cls.def(py::init<Schema const &>(), "schema"_a);
    cls.def(py::init<std::shared_ptr<ExposureTable> const &>(), "table"_a=std::shared_ptr<ExposureTable>());
    // The C++ also defines a templated constructor taking a table and an input iterator,
    // but I'm not sure how to wrap it nor if it's needed; I don't see it instantiated in the old code.
    cls.def(py::init<Catalog const &>(), "other"_a);

    /* Methods */
    // cls.def("writeFits", 
    //         (void (Catalog::*)(fits::Fits &, std::shared_ptr<io::OutputArchive>, int) const)
    //             &Catalog::writeFits,
    //         "fitsfile"_a, "archive"_a, "flags"_a=0);
    cls.def_static("readFits",
                   (Catalog (*)(std::string const &, int, int)) &Catalog::readFits,
                   "filename"_a, "hdu"_a=0, "flags"_a=0);
    cls.def_static("readFits",
                  (Catalog (*)(fits::MemFileManager &, int, int)) &Catalog::readFits,
                  "manager"_a, "hdu"_a=0, "flags"_a=0);
    //cls.def_static("readFits",
    //               (Catalog (*)(fits::Fits &, int)) &Catalog::readFits,
    //               "fitsfile"_a, "flags"_a=0);
    //cls.def_static("readFits",
    //               (Catalog (*)(fits::Fits &, std::shared_ptr<io::InputArchive>, int)) &Catalog::readFits,
    //               "fitsfile"_a, "archive"_a, "flags"_a=0);
    cls.def("subset",
            (Catalog (Catalog::*)(ndarray::Array<bool const, 1> const &) const) &Catalog::subset, "mask"_a);
    cls.def("subset",
            (Catalog (Catalog::*)(std::ptrdiff_t, std::ptrdiff_t, std::ptrdiff_t) const) &Catalog::subset,
            "startd"_a, "stopd"_a, "step"_a);
    cls.def("subsetContaining",
            (Catalog (Catalog::*)(coord::Coord const &, bool) const) &Catalog::subsetContaining,
            "coord"_a, "includeValidPolygon"_a=false);
    cls.def("subsetContaining",
            (Catalog (Catalog::*)(geom::Point2D const &, image::Wcs const &, bool) const)
             &Catalog::subsetContaining,
            "point"_a, "wcs"_a, "includeValidPolygon"_a=false);
};

}  // anonymous namespace


PYBIND11_PLUGIN(_exposure) {
    py::module mod("_exposure", "Python wrapper for afw _exposure library");

    /* Module level */
    PyExposureRecord clsExposureRecord(mod, "ExposureRecord");
    PyExposureTable clsExposureTable(mod, "ExposureTable");
    PyExposureColumnView clsExposureColumnView(mod, "ExposureColumnView");
    PyBaseExposureCatalog clsBaseExposureCatalog(mod, "_BaseExposureCatalog");
    PySortedBaseExposureCatalog clsSortedBaseExposureCatalog(mod, "_SortedBaseExposureCatalog");
    PyExposureCatalog clsExposureCatalog(mod, "ExposureCatalog", py::dynamic_attr());

    /* Members */
    declareExposureRecord(clsExposureRecord);
    declareExposureTable(clsExposureTable);
    table::pybind11::declareColumnView(clsExposureColumnView);
    table::pybind11::declareCatalog(clsBaseExposureCatalog);
    table::pybind11::declareSortedCatalog(clsSortedBaseExposureCatalog);
    declareExposureCatalog(clsExposureCatalog);

    clsExposureRecord.attr("Table") = clsExposureTable;
    clsExposureRecord.attr("ColumnView") = clsExposureColumnView;
    clsExposureRecord.attr("Catalog") = clsExposureCatalog;
    clsExposureTable.attr("Record") = clsExposureRecord;
    clsExposureTable.attr("ColumnView") = clsExposureColumnView;
    clsExposureTable.attr("Catalog") = clsExposureCatalog;
    clsExposureCatalog.attr("Record") = clsExposureRecord;
    clsExposureCatalog.attr("Table") = clsExposureTable;
    clsExposureCatalog.attr("ColumnView") = clsExposureColumnView;

    return mod.ptr();
}

}}}  // namespace lsst::afw::table
