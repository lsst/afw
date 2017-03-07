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

PYBIND11_PLUGIN(_exposureInfo) {
    py::module mod("_exposureInfo", "Python wrapper for afw _exposureInfo library");

    // TODO: Commented-out code is waiting until needed and is untested.
    // Add tests for it and enable it or remove it before the final pybind11 merge.

    /* Module level */
    py::class_<ExposureInfo, std::shared_ptr<ExposureInfo>> clsExposureInfo(mod, "ExposureInfo");

    /* Member types and enums */

    /* Constructors */
    clsExposureInfo.def(py::init<std::shared_ptr<Wcs const> const &,
                                std::shared_ptr<detection::Psf const> const &,
                                std::shared_ptr<Calib const> const &,
                                std::shared_ptr<cameraGeom::Detector const> const &,
                                std::shared_ptr<geom::polygon::Polygon const> const &,
                                Filter const &,
                                std::shared_ptr<daf::base::PropertySet> const &,
                                std::shared_ptr<CoaddInputs> const &,
                                std::shared_ptr<ApCorrMap> const &,
                                std::shared_ptr<VisitInfo const>>(),
                       "wcs"_a=std::shared_ptr<Wcs const>(),
                       "psf"_a=std::shared_ptr<detection::Psf const>(),
                       "calib"_a=std::shared_ptr<Calib const>(),
                       "detector"_a=std::shared_ptr<cameraGeom::Detector const>(),
                       "polygon"_a=std::shared_ptr<geom::polygon::Polygon const>(),
                       "filter"_a=Filter(),
                       "metadata"_a=std::shared_ptr<daf::base::PropertySet>(),
                       "coaddInputs"_a=std::shared_ptr<CoaddInputs>(),
                       "apCorrMap"_a=std::shared_ptr<ApCorrMap>(),
                       "visitInfo"_a=std::shared_ptr<VisitInfo const>());
    clsExposureInfo.def(py::init<ExposureInfo>(), "other"_a);
    clsExposureInfo.def(py::init<ExposureInfo, bool>(), "other"_a, "copyMetadata"_a);

    /* Operators */

    /* Members */
    clsExposureInfo.def("hasWcs", &ExposureInfo::hasWcs);
    clsExposureInfo.def("getWcs", (std::shared_ptr<Wcs> (ExposureInfo::*)()) &ExposureInfo::getWcs);
    clsExposureInfo.def("setWcs", &ExposureInfo::setWcs, "wcs"_a);
    //clsExposureInfo.def("initWcs", &ExposureInfo::initWcs);

    clsExposureInfo.def("hasDetector", &ExposureInfo::hasDetector);
    clsExposureInfo.def("getDetector", &ExposureInfo::getDetector);
    clsExposureInfo.def("setDetector", &ExposureInfo::setDetector, "detector"_a);

    clsExposureInfo.def("getFilter", &ExposureInfo::getFilter);
    clsExposureInfo.def("setFilter", &ExposureInfo::setFilter, "filter"_a);

    clsExposureInfo.def("hasCalib", &ExposureInfo::hasCalib);
    clsExposureInfo.def("getCalib", (std::shared_ptr<Calib> (ExposureInfo::*)()) &ExposureInfo::getCalib);
    clsExposureInfo.def("setCalib", &ExposureInfo::setCalib, "calib"_a);

    clsExposureInfo.def("getMetadata", &ExposureInfo::getMetadata);
    clsExposureInfo.def("setMetadata", &ExposureInfo::setMetadata, "metadata"_a);

    //clsExposureInfo.def("hasPsf", &ExposureInfo::hasPsf);
    clsExposureInfo.def("getPsf", &ExposureInfo::getPsf);
    clsExposureInfo.def("setPsf", &ExposureInfo::setPsf, "psf"_a);

    //clsExposureInfo.def("hasValidPolygon", &ExposureInfo::hasValidPolygon);
    clsExposureInfo.def("getValidPolygon", &ExposureInfo::getValidPolygon);
    clsExposureInfo.def("setValidPolygon", &ExposureInfo::setValidPolygon, "polygon"_a);

    //clsExposureInfo.def("hasApCorrMap", &ExposureInfo::hasApCorrMap);
    clsExposureInfo.def("getApCorrMap",
                        (std::shared_ptr<ApCorrMap> (ExposureInfo::*)()) &ExposureInfo::getApCorrMap);
    clsExposureInfo.def("setApCorrMap", &ExposureInfo::setApCorrMap, "apCorrMap"_a);
    //clsExposureInfo.def("initApCorrMap", &ExposureInfo::initApCorrMap);

    //clsExposureInfo.def("hasCoaddInputs", &ExposureInfo::hasCoaddInputs);
    clsExposureInfo.def("getCoaddInputs", &ExposureInfo::getCoaddInputs);
    clsExposureInfo.def("setCoaddInputs", &ExposureInfo::setCoaddInputs, "coaddInputs"_a);

    //clsExposureInfo.def("hasVisitInfo", &ExposureInfo::hasVisitInfo);
    clsExposureInfo.def("getVisitInfo", &ExposureInfo::getVisitInfo);
    clsExposureInfo.def("setVisitInfo", &ExposureInfo::setVisitInfo, "visitInfo"_a);

    return mod.ptr();
}

}}}  // namespace lsst::afw::image
