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
#include <pybind11/operators.h>
//#include <pybind11/stl.h>

#include "numpy/arrayobject.h"
#include "ndarray/pybind11.h"
#include "ndarray/converter.h"

#include "lsst/afw/image/Wcs.h"
#include "lsst/afw/table/io/Persistable.h"

#include "lsst/afw/table/io/pybind11.h"

namespace py = pybind11;

using namespace lsst::afw::image;

using lsst::afw::table::io::PersistableFacade;

PYBIND11_PLUGIN(_wcs) {
    py::module mod("_wcs", "Python wrapper for afw _wcs library");

    if (_import_array() < 0) {
            PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
            return nullptr;
    };

    lsst::afw::table::io::declarePersistableFacade<Wcs>(mod, "Wcs");

    /* Module level */
    mod.def("makeWcs", (PTR(Wcs) (*)(PTR(lsst::daf::base::PropertySet) const &, bool)) makeWcs,
            py::arg("fitsMetadata"), py::arg("stripMetadata")=false);
    mod.def("makeWcs", (PTR(Wcs) (*)(lsst::afw::coord::Coord const &, lsst::afw::geom::Point2D const &,
            double, double, double, double)) makeWcs);

    py::class_<Wcs, std::shared_ptr<Wcs>, lsst::afw::table::io::Persistable, PersistableFacade<Wcs>> cls(mod, "Wcs");

    /* Constructors */
    cls.def(py::init<lsst::afw::geom::Point2D const &, lsst::afw::geom::Point2D const &,
        Eigen::Matrix2d const &, std::string const &, std::string const &,
        double, std::string const &, std::string const &, std::string const &>(),
        py::arg("crval"), py::arg("crpix"), py::arg("CD"), py::arg("ctype1")="RA---TAN",
        py::arg("ctype2")="DEC--TAN", py::arg("equinox")=2000, py::arg("raDecSys")="ICRS",
        py::arg("cunits1")="deg", py::arg("cunits2")="deg");

    /* Operators */
    cls.def(py::self == py::self);
    cls.def(py::self != py::self);

    /* Members */
    cls.def("clone", &Wcs::clone);
    cls.def("getSkyOrigin", &Wcs::getSkyOrigin);
    cls.def("getPixelOrigin", &Wcs::getPixelOrigin);
    cls.def("getCDMatrix", &Wcs::getCDMatrix);
    cls.def("flipImage", &Wcs::flipImage);
    cls.def("rotateImageBy90", &Wcs::rotateImageBy90);
    cls.def("getFitsMetadata", &Wcs::getFitsMetadata);
    cls.def("isFlipped", &Wcs::isFlipped);
    cls.def("pixArea", &Wcs::pixArea);
    cls.def("pixelScale", &Wcs::pixelScale);
    cls.def("pixelToSky", (PTR(lsst::afw::coord::Coord) (Wcs::*)(double, double) const) &Wcs::pixelToSky);
    cls.def("pixelToSky", (PTR(lsst::afw::coord::Coord) (Wcs::*)(lsst::afw::geom::Point2D const &) const) &Wcs::pixelToSky);
    cls.def("pixelToSky", (void (Wcs::*)(double, double, lsst::afw::geom::Angle&, lsst::afw::geom::Angle&) const) &Wcs::pixelToSky);
    cls.def("skyToPixel", (lsst::afw::geom::Point2D (Wcs::*)(lsst::afw::geom::Angle, lsst::afw::geom::Angle) const) &Wcs::skyToPixel);
    cls.def("skyToPixel", (lsst::afw::geom::Point2D (Wcs::*)(lsst::afw::coord::Coord const &) const) &Wcs::skyToPixel);
    cls.def("skyToIntermediateWorldCoord", &Wcs::skyToIntermediateWorldCoord);
    cls.def("hasDistortion", &Wcs::hasDistortion);
    cls.def("getCoordSystem", &Wcs::getCoordSystem);
    cls.def("getEquinox", &Wcs::getEquinox);
    cls.def("isSameSkySystem", &Wcs::isSameSkySystem);
    cls.def("getLinearTransform", &Wcs::getLinearTransform);
    cls.def("linearizePixelToSky", (lsst::afw::geom::AffineTransform (Wcs::*)(lsst::afw::coord::Coord const &, lsst::afw::geom::AngleUnit) const) &Wcs::linearizePixelToSky,
            py::arg("coord"), py::arg("skyUnit")=lsst::afw::geom::degrees);
    cls.def("linearizePixelToSky", (lsst::afw::geom::AffineTransform (Wcs::*)(lsst::afw::geom::Point2D const &, lsst::afw::geom::AngleUnit) const) &Wcs::linearizePixelToSky,
            py::arg("coord"), py::arg("skyUnit")=lsst::afw::geom::degrees);
    cls.def("linearizeSkyToPixel", (lsst::afw::geom::AffineTransform (Wcs::*)(lsst::afw::coord::Coord const &, lsst::afw::geom::AngleUnit) const) &Wcs::linearizeSkyToPixel,
            py::arg("coord"), py::arg("skyUnit")=lsst::afw::geom::degrees);
    cls.def("linearizeSkyToPixel", (lsst::afw::geom::AffineTransform (Wcs::*)(lsst::afw::geom::Point2D const &, lsst::afw::geom::AngleUnit) const) &Wcs::linearizeSkyToPixel,
            py::arg("coord"), py::arg("skyUnit")=lsst::afw::geom::degrees);
    cls.def("shiftReferencePixel", (void (Wcs::*)(double, double)) &Wcs::shiftReferencePixel);
    cls.def("shiftReferencePixel", (void (Wcs::*)(lsst::afw::geom::Extent2D const &)) &Wcs::shiftReferencePixel);
    cls.def("isPersistable", &Wcs::isPersistable);

    return mod.ptr();
}
