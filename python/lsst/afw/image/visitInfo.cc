/*
 * LSST Data Management System
 * Copyright 2008-2017 AURA/LSST.
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

#include <memory>
#include <limits>
#include <sstream>

#include "lsst/utils/python.h"

#include "lsst/daf/base/PropertySet.h"
#include "lsst/afw/coord/Observatory.h"
#include "lsst/afw/coord/Weather.h"
#include "lsst/geom/Angle.h"
#include "lsst/geom/SpherePoint.h"
#include "lsst/afw/table/io/python.h"  // for addPersistableMethods
#include "lsst/afw/table/misc.h"
#include "lsst/afw/image/VisitInfo.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst {
namespace afw {
namespace image {

namespace {

static double const nan(std::numeric_limits<double>::quiet_NaN());
static lsst::geom::Angle const nanAngle(nan);

}  // namespace

PYBIND11_PLUGIN(visitInfo) {
    py::module mod("visitInfo");

    py::module::import("lsst.daf.base");
    py::module::import("lsst.geom");
    py::module::import("lsst.afw.coord.observatory");
    py::module::import("lsst.afw.coord.weather");

    /* Module level */
    py::class_<VisitInfo, std::shared_ptr<VisitInfo>> cls(mod, "VisitInfo");

    /* Member types and enums */
    py::enum_<RotType>(mod, "RotType")
            .value("UNKNOWN", RotType::UNKNOWN)
            .value("SKY", RotType::SKY)
            .value("HORIZON", RotType::HORIZON)
            .value("MOUNT", RotType::MOUNT)
            .export_values();

    /* Constructors */
    cls.def(py::init<table::RecordId, double, double, daf::base::DateTime const &, double,
                     lsst::geom::Angle const &, lsst::geom::SpherePoint const &, lsst::geom::SpherePoint const &, double,
                     lsst::geom::Angle const &, RotType const &, coord::Observatory const &,
                     coord::Weather const &>(),
            "exposureId"_a = 0, "exposureTime"_a = nan, "darkTime"_a = nan, "date"_a = daf::base::DateTime(),
            "ut1"_a = nan, "era"_a = nanAngle, "boresightRaDec"_a = lsst::geom::SpherePoint(nanAngle, nanAngle),
            "boresightAzAlt"_a = lsst::geom::SpherePoint(nanAngle, nanAngle), "boresightAirmass"_a = nan,
            "boresightRotAngle"_a = nanAngle, "rotType"_a = RotType::UNKNOWN,
            "observatory"_a = coord::Observatory(nanAngle, nanAngle, nan),
            "weather"_a = coord::Weather(nan, nan, nan));
    cls.def(py::init<daf::base::PropertySet const &>(), "metadata"_a);
    cls.def(py::init<VisitInfo const &>(), "visitInfo"_a);

    table::io::python::addPersistableMethods<VisitInfo>(cls);

    /* Operators */
    cls.def("__eq__", [](VisitInfo const &self, VisitInfo const &other) { return self == other; },
            py::is_operator());
    cls.def("__ne__", [](VisitInfo const &self, VisitInfo const &other) { return self != other; },
            py::is_operator());

    /* Members */
    cls.def("getExposureId", &VisitInfo::getExposureId);
    cls.def("getExposureTime", &VisitInfo::getExposureTime);
    cls.def("getDarkTime", &VisitInfo::getDarkTime);
    cls.def("getDate", &VisitInfo::getDate);
    cls.def("getUt1", &VisitInfo::getUt1);
    cls.def("getEra", &VisitInfo::getEra);
    cls.def("getBoresightRaDec", &VisitInfo::getBoresightRaDec);
    cls.def("getBoresightAzAlt", &VisitInfo::getBoresightAzAlt);
    cls.def("getBoresightAirmass", &VisitInfo::getBoresightAirmass);
    cls.def("getBoresightParAngle", &VisitInfo::getBoresightParAngle);
    cls.def("getBoresightRotAngle", &VisitInfo::getBoresightRotAngle);
    cls.def("getRotType", &VisitInfo::getRotType);
    cls.def("getObservatory", &VisitInfo::getObservatory);
    cls.def("getWeather", &VisitInfo::getWeather);
    cls.def("isPersistable", &VisitInfo::isPersistable);
    cls.def("getLocalEra", &VisitInfo::getLocalEra);
    cls.def("getBoresightHourAngle", &VisitInfo::getBoresightHourAngle);

    utils::python::addOutputOp(cls, "__str__");

    /* Free Functions */
    mod.def("setVisitInfoMetadata", &detail::setVisitInfoMetadata, "metadata"_a, "visitInfo"_a);
    mod.def("stripVisitInfoKeywords", &detail::stripVisitInfoKeywords, "metadata"_a);

    return mod.ptr();
}
}  // namespace image
}  // namespace afw
}  // namespace lsst
