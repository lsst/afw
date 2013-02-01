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

%define geomLib_DOCSTRING
"
Basic geometry classes for Euclidean coordinate systems.
"
%enddef

%module(package="lsst.afw.geom", docstring=geomLib_DOCSTRING) geomLib

%include "lsst/p_lsstSwig.i"

%lsst_exceptions();

namespace lsst { namespace afw { namespace geom {

template <typename T, int N> class Point;
template <typename T, int N> class Extent;

typedef Point<double,2> PointD;
typedef Extent<double,2> ExtentD;
typedef Point<double,2> Point2D;
typedef Extent<double,2> Extent2D;
typedef Point<double,3> Point3D;
typedef Extent<double,3> Extent3D;
typedef Point<int,2> PointI;
typedef Extent<int,2> ExtentI;
typedef Point<int,2> Point2I;
typedef Extent<int,2> Extent2I;
typedef Point<int,3> Point3I;
typedef Extent<int,3> Extent3I;

class Angle;
class AngleUnit;
class Span;
class Box2I;
class Box2D;
class LinearTransform;
class AffineTransform;

}}} // namespace lsst::afw::geom

%shared_ptr(lsst::afw::geom::Span);

