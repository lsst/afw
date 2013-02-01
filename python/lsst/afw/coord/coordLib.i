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

%include "lsst/afw/coord/coord_fwd.i"

%include "std_pair.i"
%template(pairSS) std::pair<std::string, std::string>;

%import "lsst/afw/geom/Angle.i"

%{
#include "lsst/afw/geom/Angle.h"
%}
%template(pairAngleAngle) std::pair<lsst::afw::geom::Angle, lsst::afw::geom::Angle>;

%include "Observatory.i"
%include "Coord.i"
