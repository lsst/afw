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
#include <pybind11/stl.h>
#include <pybind11/operators.h>

#include "lsst/afw/coord/Coord.h"

namespace py = pybind11;

using namespace lsst::afw::coord;

PYBIND11_DECLARE_HOLDER_TYPE(MyCoordType, std::shared_ptr<MyCoordType>);

PYBIND11_PLUGIN(_coord) {
    py::module mod("_coord", "Python wrapper for afw _coord library");

    /* Types and enums */
    py::enum_<CoordSystem>(mod, "CoordSystem")
        .value("UNKNOWN", CoordSystem::UNKNOWN)
        .value("FK5", CoordSystem::FK5)
        .value("ICRS", CoordSystem::ICRS)
        .value("GALACTIC", CoordSystem::GALACTIC)
        .value("ECLIPTIC", CoordSystem::ECLIPTIC)
        .value("TOPOCENTRIC", CoordSystem::TOPOCENTRIC)
    .export_values();

    mod.def("makeCoordEnum", makeCoordEnum);

    py::class_<Coord, std::shared_ptr<Coord>> clsCoord(mod, "Coord");

    /* Constructors */
    clsCoord.def(py::init<lsst::afw::geom::Point2D const, lsst::afw::geom::AngleUnit, double const>(),
        py::arg("p2d"), py::arg("unit") = lsst::afw::geom::degrees, py::arg("epoch") = 2000.0);
    clsCoord.def(py::init<lsst::afw::geom::Point3D const, double const, bool, lsst::afw::geom::Angle const>(),
        py::arg("p2d"), py::arg("epoch") = 2000.0, py::arg("normalize") = true, py::arg("defaultLongitude") = lsst::afw::geom::Angle(0.));
    clsCoord.def(py::init<lsst::afw::geom::Angle const, lsst::afw::geom::Angle const, double const>(),
        py::arg("ra"), py::arg("dec"), py::arg("epoch") = 2000.0);
    clsCoord.def(py::init<std::string const, std::string const, double const>(),
        py::arg("ra"), py::arg("dec"), py::arg("epoch") = 2000.0);
    clsCoord.def(py::init<>());

    /* Operators */
    clsCoord.def("__str__", &Coord::toString);
    clsCoord.def("__repr__", &Coord::toString);
    clsCoord.def("__getitem__", [](const Coord &c, size_t i) {
        if (i >= 2)
            throw py::index_error();
        return c[i];
    });
    clsCoord.def("__eq__", [](Coord &a, const Coord & b) {
        return a == b;
    });
    clsCoord.def("__ne__", [](Coord &a, const Coord & b) {
        return a != b;
    });

    /* Members */
    clsCoord.def("reset", (void (Coord::*)(lsst::afw::geom::Angle const, lsst::afw::geom::Angle const)) &Coord::reset);
    clsCoord.def("reset", (void (Coord::*)(lsst::afw::geom::Angle const, lsst::afw::geom::Angle const, double const)) &Coord::reset);
    clsCoord.def("clone", &Coord::clone);
    clsCoord.def("getEpoch", &Coord::getEpoch);
    clsCoord.def("getPosition", &Coord::getPosition,
        py::arg("unit") = lsst::afw::geom::degrees);
    clsCoord.def("getCoordNames", &Coord::getCoordNames);
    clsCoord.def("getVector", &Coord::getVector);
    clsCoord.def("getClassName", &Coord::getClassName);
    clsCoord.def("getCoordSystem", &Coord::getCoordSystem);
    clsCoord.def("getLongitude", &Coord::getLongitude);
    clsCoord.def("getLatitude", &Coord::getLatitude);
    clsCoord.def("getLongitudeStr", &Coord::getLongitudeStr);
    clsCoord.def("getLatitudeStr", &Coord::getLatitudeStr);
    clsCoord.def("angularSeparation", &Coord::angularSeparation);
    clsCoord.def("getOffsetFrom", &Coord::getOffsetFrom);
    clsCoord.def("getTangentPlaneOffset", &Coord::getTangentPlaneOffset);
    clsCoord.def("rotate", &Coord::rotate);
    clsCoord.def("offset", &Coord::offset);
    clsCoord.def("convert", &Coord::convert,
        py::arg("system"), py::arg("epoch") = 2000);
    clsCoord.def("toFk5", (Fk5Coord (Coord::*)(double const) const) &Coord::toFk5);
    clsCoord.def("toFk5", (Fk5Coord (Coord::*)() const) &Coord::toFk5);
    clsCoord.def("toIcrs", &Coord::toIcrs);
    clsCoord.def("toGalactic", &Coord::toGalactic);
    clsCoord.def("toEcliptic", (EclipticCoord (Coord::*)(double const) const) &Coord::toEcliptic);
    clsCoord.def("toEcliptic", (EclipticCoord (Coord::*)() const) &Coord::toEcliptic);
    clsCoord.def("toTopocentric", &Coord::toTopocentric);

    py::class_<IcrsCoord, std::shared_ptr<IcrsCoord>, Coord> clsIcrsCoord(mod, "IcrsCoord");

    /* Constructors */
    clsIcrsCoord.def(py::init<lsst::afw::geom::Point2D const &, lsst::afw::geom::AngleUnit>(),
        py::arg("p2d"), py::arg("unit") = lsst::afw::geom::degrees);
    clsIcrsCoord.def(py::init<lsst::afw::geom::Point3D const &, bool, lsst::afw::geom::Angle const>(),
        py::arg("p3d"), py::arg("normalize") = true, py::arg("defaultLongitude") = lsst::afw::geom::Angle(0.));
    clsIcrsCoord.def(py::init<lsst::afw::geom::Angle const, lsst::afw::geom::Angle const>(),
        py::arg("ra"), py::arg("dec"));
    clsIcrsCoord.def(py::init<std::string const, std::string const>(),
        py::arg("ra"), py::arg("dec"));
    clsIcrsCoord.def(py::init<>());

    /* Members */
    clsIcrsCoord.def("clone", &IcrsCoord::clone);
    clsIcrsCoord.def("getClassName", &IcrsCoord::getClassName);
    clsIcrsCoord.def("getCoordSystem", &IcrsCoord::getCoordSystem);
    clsIcrsCoord.def("reset", (void (IcrsCoord::*)(lsst::afw::geom::Angle const, lsst::afw::geom::Angle const)) &IcrsCoord::reset);
    clsIcrsCoord.def("reset", (void (IcrsCoord::*)(lsst::afw::geom::Angle const, lsst::afw::geom::Angle const, double const)) &IcrsCoord::reset);
    clsIcrsCoord.def("getRa", &IcrsCoord::getRa);
    clsIcrsCoord.def("getDec", &IcrsCoord::getDec);
    clsIcrsCoord.def("getRaStr", &IcrsCoord::getRaStr);
    clsIcrsCoord.def("getDecStr", &IcrsCoord::getDecStr);
    clsIcrsCoord.def("toFk5", (Fk5Coord (IcrsCoord::*)(double const) const) &IcrsCoord::toFk5);
    clsIcrsCoord.def("toFk5", (Fk5Coord (IcrsCoord::*)() const) &IcrsCoord::toFk5);
    clsIcrsCoord.def("toIcrs", &IcrsCoord::toIcrs);

    py::class_<Fk5Coord, std::shared_ptr<Fk5Coord>, Coord> clsFk5Coord(mod, "Fk5Coord");

    /* Constructors */
    clsFk5Coord.def(py::init<lsst::afw::geom::Point2D const &, lsst::afw::geom::AngleUnit, double const>(),
        py::arg("p2d"), py::arg("unit") = lsst::afw::geom::degrees, py::arg("epoch") = 2000.0);
    clsFk5Coord.def(py::init<lsst::afw::geom::Point3D const &, double const, bool, lsst::afw::geom::Angle const>(),
        py::arg("p3d"), py::arg("epoch") = 2000.0, py::arg("normalize") = true, py::arg("defaultLongitude") = lsst::afw::geom::Angle(0.));
    clsFk5Coord.def(py::init<lsst::afw::geom::Angle const, lsst::afw::geom::Angle const, double const>(),
        py::arg("ra"), py::arg("dec"), py::arg("epoch") = 2000.0);
    clsFk5Coord.def(py::init<std::string const, std::string const, double const>(),
        py::arg("ra"), py::arg("dec"), py::arg("epoch") = 2000.0);
    clsFk5Coord.def(py::init<>());

    /* Members */
    clsFk5Coord.def("clone", &Fk5Coord::clone);
    clsFk5Coord.def("getClassName", &Fk5Coord::getClassName);
    clsFk5Coord.def("getCoordSystem", &Fk5Coord::getCoordSystem);
    clsFk5Coord.def("precess", &Fk5Coord::precess);
    clsFk5Coord.def("getRa", &Fk5Coord::getRa);
    clsFk5Coord.def("getDec", &Fk5Coord::getDec);
    clsFk5Coord.def("getRaStr", &Fk5Coord::getRaStr);
    clsFk5Coord.def("getDecStr", &Fk5Coord::getDecStr);
    clsFk5Coord.def("toFk5", (Fk5Coord (Fk5Coord::*)(double const) const) &Fk5Coord::toFk5);
    clsFk5Coord.def("toFk5", (Fk5Coord (Fk5Coord::*)() const) &Fk5Coord::toFk5);
    clsFk5Coord.def("toIcrs", &Fk5Coord::toIcrs);
    clsFk5Coord.def("toGalactic", &Fk5Coord::toGalactic);
    clsFk5Coord.def("toEcliptic", (EclipticCoord (Fk5Coord::*)(double const) const) &Fk5Coord::toEcliptic);
    clsFk5Coord.def("toEcliptic", (EclipticCoord (Fk5Coord::*)() const) &Fk5Coord::toEcliptic);
    clsFk5Coord.def("toTopocentric", &Fk5Coord::toTopocentric);

    py::class_<GalacticCoord, std::shared_ptr<GalacticCoord>, Coord> clsGalacticCoord(mod, "GalacticCoord");

    /* Constructors */
    clsGalacticCoord.def(py::init<lsst::afw::geom::Point2D const &, lsst::afw::geom::AngleUnit>(),
        py::arg("p2d"), py::arg("unit") = lsst::afw::geom::degrees);
    clsGalacticCoord.def(py::init<lsst::afw::geom::Point3D const &, bool, lsst::afw::geom::Angle const>(),
        py::arg("p3d"), py::arg("normalize") = true, py::arg("defaultLongitude") = lsst::afw::geom::Angle(0.));
    clsGalacticCoord.def(py::init<lsst::afw::geom::Angle const, lsst::afw::geom::Angle const>(),
        py::arg("l"), py::arg("b"));
    clsGalacticCoord.def(py::init<std::string const, std::string const>(),
        py::arg("l"), py::arg("b"));
    clsGalacticCoord.def(py::init<>());

    /* Members */
    clsGalacticCoord.def("clone", &GalacticCoord::clone);
    clsGalacticCoord.def("getClassName", &GalacticCoord::getClassName);
    clsGalacticCoord.def("getCoordSystem", &GalacticCoord::getCoordSystem);
    clsGalacticCoord.def("reset", (void (GalacticCoord::*)(lsst::afw::geom::Angle const, lsst::afw::geom::Angle const)) &GalacticCoord::reset);
    clsGalacticCoord.def("reset", (void (GalacticCoord::*)(lsst::afw::geom::Angle const, lsst::afw::geom::Angle const, double const)) &GalacticCoord::reset);
    clsGalacticCoord.def("getCoordNames", &GalacticCoord::getCoordNames);
    clsGalacticCoord.def("getL", &GalacticCoord::getL);
    clsGalacticCoord.def("getB", &GalacticCoord::getB);
    clsGalacticCoord.def("getLStr", &GalacticCoord::getLStr);
    clsGalacticCoord.def("getBStr", &GalacticCoord::getBStr);
    clsGalacticCoord.def("toFk5", (Fk5Coord (GalacticCoord::*)(double const) const) &GalacticCoord::toFk5);
    clsGalacticCoord.def("toFk5", (Fk5Coord (GalacticCoord::*)() const) &GalacticCoord::toFk5);
    clsGalacticCoord.def("toGalactic", &GalacticCoord::toGalactic);

    py::class_<EclipticCoord, std::shared_ptr<EclipticCoord>, Coord> clsEclipticCoord(mod, "EclipticCoord");

    /* Constructors */
    clsEclipticCoord.def(py::init<lsst::afw::geom::Point2D const &, lsst::afw::geom::AngleUnit, double const>(),
        py::arg("p2d"), py::arg("unit") = lsst::afw::geom::degrees, py::arg("epoch") = 2000.0);
    clsEclipticCoord.def(py::init<lsst::afw::geom::Point3D const &, double const, bool, lsst::afw::geom::Angle const>(),
        py::arg("p3d"), py::arg("epoch") = 2000.0, py::arg("normalize") = true, py::arg("defaultLongitude") = lsst::afw::geom::Angle(0.));
    clsEclipticCoord.def(py::init<lsst::afw::geom::Angle const, lsst::afw::geom::Angle const, double const>(),
        py::arg("lambda"), py::arg("beta"), py::arg("epoch") = 2000.0);
    clsEclipticCoord.def(py::init<std::string const, std::string const, double const>(),
        py::arg("lambda"), py::arg("beta"), py::arg("epoch") = 2000.0);
    clsEclipticCoord.def(py::init<>());

    /* Members */
    clsEclipticCoord.def("clone", &EclipticCoord::clone);
    clsEclipticCoord.def("getClassName", &EclipticCoord::getClassName);
    clsEclipticCoord.def("getCoordSystem", &EclipticCoord::getCoordSystem);
    clsEclipticCoord.def("getCoordNames", &EclipticCoord::getCoordNames);
    clsEclipticCoord.def("getLambda", &EclipticCoord::getLambda);
    clsEclipticCoord.def("getBeta", &EclipticCoord::getBeta);
    clsEclipticCoord.def("getLambdaStr", &EclipticCoord::getLambdaStr);
    clsEclipticCoord.def("getBetaStr", &EclipticCoord::getBetaStr);
    clsEclipticCoord.def("toFk5", (Fk5Coord (EclipticCoord::*)(double const) const) &EclipticCoord::toFk5);
    clsEclipticCoord.def("toFk5", (Fk5Coord (EclipticCoord::*)() const) &EclipticCoord::toFk5);
    clsGalacticCoord.def("toEcliptic", (EclipticCoord (GalacticCoord::*)(double const) const) &GalacticCoord::toEcliptic);
    clsGalacticCoord.def("toEcliptic", (EclipticCoord (GalacticCoord::*)() const) &GalacticCoord::toEcliptic);
    clsEclipticCoord.def("precess", &EclipticCoord::precess);

    py::class_<TopocentricCoord, std::shared_ptr<TopocentricCoord>, Coord> clsTopocentricCoord(mod, "TopocentricCoord");

    /* Constructors */
    clsTopocentricCoord.def(py::init<lsst::afw::geom::Point2D const &, lsst::afw::geom::AngleUnit, double const, Observatory const &>(),
        py::arg("p2d"), py::arg("unit") = lsst::afw::geom::degrees, py::arg("epoch"), py::arg("observatory"));
    clsTopocentricCoord.def(py::init<lsst::afw::geom::Point3D const &, double const, Observatory const &,  bool, lsst::afw::geom::Angle const>(),
        py::arg("p3d"), py::arg("epoch") = 2000.0, py::arg("normalize"), py::arg("observatory"), py::arg("defaultLongitude") = lsst::afw::geom::Angle(0.));
    clsTopocentricCoord.def(py::init<lsst::afw::geom::Angle const, lsst::afw::geom::Angle const, double const, Observatory const &>(),
        py::arg("az"), py::arg("alt"), py::arg("epoch"), py::arg("observatory"));
    clsTopocentricCoord.def(py::init<std::string const, std::string const, double const, Observatory const &>(),
        py::arg("az"), py::arg("alt"), py::arg("epoch"), py::arg("observatory"));

    /* Members */
    clsTopocentricCoord.def("clone", &TopocentricCoord::clone);
    clsTopocentricCoord.def("getClassName", &TopocentricCoord::getClassName);
    clsTopocentricCoord.def("getCoordSystem", &TopocentricCoord::getCoordSystem);
    clsTopocentricCoord.def("getObservatory", &TopocentricCoord::getObservatory);
    clsTopocentricCoord.def("getCoordNames", &TopocentricCoord::getCoordNames);
    clsTopocentricCoord.def("getAzimuth", &TopocentricCoord::getAzimuth);
    clsTopocentricCoord.def("getAltitude", &TopocentricCoord::getAltitude);
    clsTopocentricCoord.def("getAzimuthStr", &TopocentricCoord::getAzimuthStr);
    clsTopocentricCoord.def("getAltitudeStr", &TopocentricCoord::getAltitudeStr);
    clsTopocentricCoord.def("toFk5", (Fk5Coord (TopocentricCoord::*)(double const) const) &TopocentricCoord::toFk5);
    clsTopocentricCoord.def("toFk5", (Fk5Coord (TopocentricCoord::*)() const) &TopocentricCoord::toFk5);
    clsTopocentricCoord.def("toTopocentric", (TopocentricCoord (TopocentricCoord::*)(Observatory const &, lsst::daf::base::DateTime const &) const) &TopocentricCoord::toTopocentric);
    clsTopocentricCoord.def("toTopocentric", (TopocentricCoord (TopocentricCoord::*)() const) &TopocentricCoord::toTopocentric);
    
    /* Non members */
    mod.def("makeCoord", (std::shared_ptr<Coord> (*)(CoordSystem const, lsst::afw::geom::Angle const, lsst::afw::geom::Angle const, double const)) makeCoord);
    mod.def("makeCoord", (std::shared_ptr<Coord> (*)(CoordSystem const, std::string const, std::string const)) makeCoord);
    mod.def("makeCoord", (std::shared_ptr<Coord> (*)(CoordSystem const, lsst::afw::geom::Point2D const &, lsst::afw::geom::AngleUnit, double const)) makeCoord);
    mod.def("makeCoord", (std::shared_ptr<Coord> (*)(CoordSystem const, lsst::afw::geom::Point3D const &, double const, bool, lsst::afw::geom::Angle const)) makeCoord,
        py::arg("system"), py::arg("p3d"), py::arg("epoch"), py::arg("normalize") = true, py::arg("defaultLongitude") = lsst::afw::geom::Angle(0.));
    mod.def("makeCoord", (std::shared_ptr<Coord> (*)(CoordSystem const)) makeCoord);
    mod.def("makeCoord", (std::shared_ptr<Coord> (*)(CoordSystem const, lsst::afw::geom::Angle const, lsst::afw::geom::Angle const)) makeCoord);
    mod.def("makeCoord", (std::shared_ptr<Coord> (*)(CoordSystem const, std::string const, std::string const)) makeCoord);
    mod.def("makeCoord", (std::shared_ptr<Coord> (*)(CoordSystem const, lsst::afw::geom::Point2D const &, lsst::afw::geom::AngleUnit)) makeCoord);
    mod.def("makeCoord", (std::shared_ptr<Coord> (*)(CoordSystem const, lsst::afw::geom::Point3D const &, bool, lsst::afw::geom::Angle const)) makeCoord,
        py::arg("system"), py::arg("p3d"), py::arg("normalize") = true, py::arg("defaultLongitude") = lsst::afw::geom::Angle(0.));
    mod.def("averageCoord", averageCoord,
        py::arg("coords"), py::arg("system") = CoordSystem::UNKNOWN);

    return mod.ptr();
}
