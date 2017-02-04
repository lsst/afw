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

#include "lsst/daf/base/PropertySet.h"
#include "lsst/afw/cameraGeom/Detector.h"
#include "lsst/afw/detection/Psf.h"
#include "lsst/afw/geom/polygon/Polygon.h"
#include "lsst/afw/image/ApCorrMap.h"
#include "lsst/afw/image/Calib.h"
#include "lsst/afw/image/CoaddInputs.h"
#include "lsst/afw/image/Filter.h"
#include "lsst/afw/image/VisitInfo.h"
#include "lsst/afw/image/Wcs.h"
#include "lsst/afw/image/ExposureInfo.h"

namespace py = pybind11;
using namespace py::literals;

namespace lsst {
namespace afw {
namespace image {
namespace {

using PyExposureInfo = py::class_<ExposureInfo, std::shared_ptr<ExposureInfo>>;

PYBIND11_PLUGIN(_exposureInfo) {
    py::module mod("_exposureInfo");

    /* Module level */
    PyExposureInfo cls(mod, "ExposureInfo");

    /* Member types and enums */

    /* Constructors */
    cls.def(
        py::init<std::shared_ptr<Wcs const> const &, std::shared_ptr<detection::Psf const> const &,
                 std::shared_ptr<Calib const> const &, std::shared_ptr<cameraGeom::Detector const> const &,
                 std::shared_ptr<geom::polygon::Polygon const> const &, Filter const &,
                 std::shared_ptr<daf::base::PropertySet> const &, std::shared_ptr<CoaddInputs> const &,
                 std::shared_ptr<ApCorrMap> const &, std::shared_ptr<VisitInfo const>>(),
        "wcs"_a = nullptr, "psf"_a = nullptr, "calib"_a = nullptr, "detector"_a = nullptr,
        "polygon"_a = nullptr, "filter"_a = Filter(), "metadata"_a = nullptr, "coaddInputs"_a = nullptr,
        "apCorrMap"_a = nullptr, "visitInfo"_a = nullptr);
    cls.def(py::init<ExposureInfo>(), "other"_a);
    cls.def(py::init<ExposureInfo, bool>(), "other"_a, "copyMetadata"_a);

    /* Members */
    cls.def("hasWcs", &ExposureInfo::hasWcs);
    cls.def("getWcs", (std::shared_ptr<Wcs> (ExposureInfo::*)()) &ExposureInfo::getWcs);
    cls.def("setWcs", &ExposureInfo::setWcs, "wcs"_a);

    cls.def("hasDetector", &ExposureInfo::hasDetector);
    cls.def("getDetector", &ExposureInfo::getDetector);
    cls.def("setDetector", &ExposureInfo::setDetector, "detector"_a);

    cls.def("getFilter", &ExposureInfo::getFilter);
    cls.def("setFilter", &ExposureInfo::setFilter, "filter"_a);

    cls.def("hasCalib", &ExposureInfo::hasCalib);
    cls.def("getCalib", (std::shared_ptr<Calib> (ExposureInfo::*)()) &ExposureInfo::getCalib);
    cls.def("setCalib", &ExposureInfo::setCalib, "calib"_a);

    cls.def("getMetadata", &ExposureInfo::getMetadata);
    cls.def("setMetadata", &ExposureInfo::setMetadata, "metadata"_a);

    cls.def("hasPsf", &ExposureInfo::hasPsf);
    cls.def("getPsf", &ExposureInfo::getPsf);
    cls.def("setPsf", &ExposureInfo::setPsf, "psf"_a);

    cls.def("hasValidPolygon", &ExposureInfo::hasValidPolygon);
    cls.def("getValidPolygon", &ExposureInfo::getValidPolygon);
    cls.def("setValidPolygon", &ExposureInfo::setValidPolygon, "polygon"_a);

    cls.def("hasApCorrMap", &ExposureInfo::hasApCorrMap);
    cls.def("getApCorrMap",
            (std::shared_ptr<ApCorrMap> (ExposureInfo::*)()) &ExposureInfo::getApCorrMap);
    cls.def("setApCorrMap", &ExposureInfo::setApCorrMap, "apCorrMap"_a);
    cls.def("initApCorrMap", &ExposureInfo::initApCorrMap);

    cls.def("hasCoaddInputs", &ExposureInfo::hasCoaddInputs);
    cls.def("getCoaddInputs", &ExposureInfo::getCoaddInputs);
    cls.def("setCoaddInputs", &ExposureInfo::setCoaddInputs, "coaddInputs"_a);

    cls.def("hasVisitInfo", &ExposureInfo::hasVisitInfo);
    cls.def("getVisitInfo", &ExposureInfo::getVisitInfo);
    cls.def("setVisitInfo", &ExposureInfo::setVisitInfo, "visitInfo"_a);

    return mod.ptr();
}

}}}}  // namespace lsst::afw::image::<anonymous>
