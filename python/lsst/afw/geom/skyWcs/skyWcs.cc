/*
 * LSST Data Management System
 * Copyright 2017 AURA/LSST.
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the LSST License Statement and
 * the GNU General Public License along with this program. If not,
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */
#include "pybind11/pybind11.h"

#include <utility>
#include <memory>

#include "astshim.h"
#include "Eigen/Core"
#include "pybind11/stl.h"
#include "numpy/arrayobject.h"
#include "ndarray/pybind11.h"

#include "lsst/afw/coord/Coord.h"
#include "lsst/afw/geom/Endpoint.h"
#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/Transform.h"
#include "lsst/afw/geom/SkyWcs.h"

namespace py = pybind11;
using namespace py::literals;

namespace lsst {
namespace afw {
namespace geom {
namespace {

PYBIND11_PLUGIN(skyWcs) {
    py::module mod("skyWcs");

    py::module::import("lsst.afw.geom.transform");

    // Need to import numpy for ndarray and eigen conversions
    if (_import_array() < 0) {
        PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
        return nullptr;
    }

    mod.def("makeCdMatrix", makeCdMatrix, "scale"_a, "orientation"_a = 0 * degrees, "flipX"_a = false);

    py::class_<SkyWcs, std::shared_ptr<SkyWcs>, Transform<Point2Endpoint, IcrsCoordEndpoint>> cls(mod,
                                                                                                  "SkyWcs");

    cls.def(py::init<Point2D const &, coord::IcrsCoord const &, Eigen::Matrix2d const &>(), "crpix"_a, "crval"_a,
            "cdMatrix"_a);
    cls.def(py::init<daf::base::PropertyList &>(), "metadata"_a);
    cls.def(py::init<ast::FrameSet const&>(), "frameSet"_a);

    cls.def("getPixelScale", (Angle (SkyWcs::*)(Point2D const &) const) &SkyWcs::getPixelScale, "pixel"_a);
    cls.def("getPixelScale", (Angle (SkyWcs::*)() const) &SkyWcs::getPixelScale);
    cls.def("getPixelOrigin", &SkyWcs::getPixelOrigin);
    cls.def("getSkyOrigin", &SkyWcs::getSkyOrigin);
    cls.def("getCdMatrix", (Eigen::Matrix2d(SkyWcs::*)(Point2D const &) const) & SkyWcs::getCdMatrix,
            "pixel"_a);
    cls.def("getCdMatrix", (Eigen::Matrix2d(SkyWcs::*)() const) & SkyWcs::getCdMatrix);
    cls.def("copyAtShiftedPixelOrigin", &SkyWcs::copyAtShiftedPixelOrigin, "shift"_a);
    cls.def("pixelToSky", (std::pair<Angle, Angle>(SkyWcs::*)(double, double) const) & SkyWcs::pixelToSky,
            "x"_a, "y"_a);
    cls.def("pixelToSky", (coord::IcrsCoord(SkyWcs::*)(Point2D const &) const) & SkyWcs::pixelToSky, "pixel"_a);
    cls.def("pixelToSky",
            (std::vector<coord::IcrsCoord>(SkyWcs::*)(std::vector<Point2D> const &) const) & SkyWcs::pixelToSky,
            "pixel"_a);
    cls.def("skyToPixel",
            (std::pair<double, double>(SkyWcs::*)(Angle const &, Angle const &) const) & SkyWcs::skyToPixel,
            "ra"_a, "dec"_a);
    cls.def("skyToPixel", (Point2D(SkyWcs::*)(coord::IcrsCoord const &) const) & SkyWcs::skyToPixel, "sky"_a);
    cls.def("skyToPixel",
            (std::vector<Point2D>(SkyWcs::*)(std::vector<coord::IcrsCoord> const &) const) & SkyWcs::skyToPixel,
            "sky"_a);
    // Do not wrap getShortClassName because it returns the name of the class;
    // use `<class>.__name__` or `type(<instance>).__name__` instead.
    // Do not wrap readStream or writeStream because C++ streams are not easy to wrap.
    cls.def_static("readString", &SkyWcs::readString);
    cls.def("writeString", &SkyWcs::writeString);

    return mod.ptr();
}

}  // namespace
}  // namespace geom
}  // namespace afw
}  // namespace lsst
