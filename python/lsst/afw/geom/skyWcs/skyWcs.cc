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

#include "lsst/afw/fits.h"
#include "lsst/afw/coord/Coord.h"
#include "lsst/afw/geom/Endpoint.h"
#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/Transform.h"
#include "lsst/afw/geom/SkyWcs.h"
#include "lsst/afw/table/io/python.h"  // for addPersistableMethods

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
    mod.def("makeFlippedWcs", makeFlippedWcs, "wcs"_a, "flipLR"_a, "flipTB"_a, "center"_a);
    mod.def("makeModifiedWcs", makeModifiedWcs, "pixelTransform"_a, "wcs"_a, "modifyActualPixels"_a);
    mod.def("makeTanSipWcs",
            (std::shared_ptr<SkyWcs>(*)(Point2D const &, coord::IcrsCoord const &, Eigen::Matrix2d const &,
                                        Eigen::MatrixXd const &, Eigen::MatrixXd const &))makeTanSipWcs,
            "crpix"_a, "crval"_a, "cdMatrix"_a, "sipA"_a, "sipB"_a);
    mod.def("makeTanSipWcs",
            (std::shared_ptr<SkyWcs>(*)(Point2D const &, coord::IcrsCoord const &, Eigen::Matrix2d const &,
                                        Eigen::MatrixXd const &, Eigen::MatrixXd const &,
                                        Eigen::MatrixXd const &, Eigen::MatrixXd const &))makeTanSipWcs,
            "crpix"_a, "crval"_a, "cdMatrix"_a, "sipA"_a, "sipB"_a, "sipAp"_a, "sipBp"_a);
    mod.def("makeWcsPairTransform", makeWcsPairTransform, "src"_a, "dst"_a);
    mod.def("getIntermediateWorldCoordsToSky", getIntermediateWorldCoordsToSky, "wcs"_a, "simplify"_a = true);
    mod.def("getPixelToIntermediateWorldCoords", getPixelToIntermediateWorldCoords, "wcs"_a,
            "simplify"_a = true);

    py::class_<SkyWcs, std::shared_ptr<SkyWcs>> cls(mod, "SkyWcs");

    cls.def(py::init<Point2D const &, coord::IcrsCoord const &, Eigen::Matrix2d const &,
                     std::string const &>(),
            "crpix"_a, "crval"_a, "cdMatrix"_a, "projection"_a = "TAN");
    cls.def(py::init<daf::base::PropertySet &, bool>(), "metadata"_a, "strip"_a = false);
    cls.def(py::init<ast::FrameDict const &>(), "frameDict"_a);

    cls.def("__eq__", &SkyWcs::operator==, py::is_operator());
    cls.def("__ne__", &SkyWcs::operator!=, py::is_operator());

    table::io::python::addPersistableMethods<SkyWcs>(cls);

    cls.def("copyAtShiftedPixelOrigin", &SkyWcs::copyAtShiftedPixelOrigin, "shift"_a);
    cls.def("getFitsMetadata", &SkyWcs::getFitsMetadata, "precise"_a);
    cls.def("getPixelScale", (Angle(SkyWcs::*)(Point2D const &) const) & SkyWcs::getPixelScale, "pixel"_a);
    cls.def("getPixelScale", (Angle(SkyWcs::*)() const) & SkyWcs::getPixelScale);
    cls.def("getPixelOrigin", &SkyWcs::getPixelOrigin);
    cls.def("getSkyOrigin", &SkyWcs::getSkyOrigin);
    cls.def("getCdMatrix", (Eigen::Matrix2d(SkyWcs::*)(Point2D const &) const) & SkyWcs::getCdMatrix,
            "pixel"_a);
    cls.def("getCdMatrix", (Eigen::Matrix2d(SkyWcs::*)() const) & SkyWcs::getCdMatrix);
    cls.def("getTanWcs", &SkyWcs::getTanWcs, "pixel"_a);
    cls.def("getFrameDict", [](SkyWcs const &self) { return self.getFrameDict()->copy(); });
    cls.def("getTransform", &SkyWcs::getTransform);

    cls.def_property_readonly("isFits", &SkyWcs::isFits);
    cls.def_property_readonly("isFlipped", &SkyWcs::isFlipped);
    cls.def("linearizePixelToSky",
            (AffineTransform(SkyWcs::*)(coord::IcrsCoord const &, AngleUnit const &) const) &
                    SkyWcs::linearizePixelToSky,
            "coord"_a, "skyUnit"_a);
    cls.def("linearizePixelToSky",
            (AffineTransform(SkyWcs::*)(Point2D const &, AngleUnit const &) const) &
                    SkyWcs::linearizePixelToSky,
            "coord"_a, "skyUnit"_a);
    cls.def("linearizeSkyToPixel",
            (AffineTransform(SkyWcs::*)(coord::IcrsCoord const &, AngleUnit const &) const) &
                    SkyWcs::linearizeSkyToPixel,
            "coord"_a, "skyUnit"_a);
    cls.def("linearizeSkyToPixel",
            (AffineTransform(SkyWcs::*)(Point2D const &, AngleUnit const &) const) &
                    SkyWcs::linearizeSkyToPixel,
            "coord"_a, "skyUnit"_a);
    cls.def("pixelToSky", (coord::IcrsCoord(SkyWcs::*)(Point2D const &) const) & SkyWcs::pixelToSky,
            "pixel"_a);
    cls.def("pixelToSky", (coord::IcrsCoord(SkyWcs::*)(double, double) const) & SkyWcs::pixelToSky, "x"_a,
            "y"_a);
    cls.def("pixelToSky",
            (std::vector<coord::IcrsCoord>(SkyWcs::*)(std::vector<Point2D> const &) const) &
                    SkyWcs::pixelToSky,
            "pixel"_a);
    cls.def("skyToPixel", (Point2D(SkyWcs::*)(coord::IcrsCoord const &) const) & SkyWcs::skyToPixel, "sky"_a);
    cls.def("skyToPixel",
            (std::vector<Point2D>(SkyWcs::*)(std::vector<coord::IcrsCoord> const &) const) &
                    SkyWcs::skyToPixel,
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
