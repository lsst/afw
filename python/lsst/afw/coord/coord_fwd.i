// -*- lsst-c++ -*-

/* 
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
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
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */

%define coordLib_DOCSTRING
"
Python interface to lsst::afw::coord
"
%enddef

%feature("autodoc", "1");
%module(package="lsst.afw.coord", docstring=coordLib_DOCSTRING) coordLib

%include "lsst/p_lsstSwig.i"

namespace lsst { namespace afw { namespace coord {

class Coord;
class Fk5Coord;
class IcrsCoord;
class GalacticCoord;
class EclipticCoord;
class TopocentricCoord;
class Observatory;

}}} // namespace lsst::afw::coord

%shared_ptr(lsst::afw::coord::Coord);
%shared_ptr(lsst::afw::coord::Fk5Coord);
%shared_ptr(lsst::afw::coord::IcrsCoord);
%shared_ptr(lsst::afw::coord::GalacticCoord);
%shared_ptr(lsst::afw::coord::EclipticCoord);
%shared_ptr(lsst::afw::coord::TopocentricCoord);
