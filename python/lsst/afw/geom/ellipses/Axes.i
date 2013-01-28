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

%module(package="lsst.afw.geom.ellipses") ellipsesLib

%include "lsst/afw/geom/ellipses/BaseCore.i"

%{
#include "lsst/afw/geom/ellipses/Axes.h"
%}

%EllipseCore_PREINCLUDE(Axes);

%include "lsst/afw/geom/ellipses/Axes.h"

%EllipseCore_POSTINCLUDE(Axes);

%extend lsst::afw::geom::ellipses::Axes {
    %pythoncode {
    def __repr__(self):
        return "Axes(a=%r, b=%r, theta=%r)" % (self.getA(), self.getB(), self.getTheta())
    def __reduce__(self):
        return (Axes, (self.getA(), self.getB(), self.getTheta()))
    def __str__(self):
        return "(a=%s, b=%s, theta=%s)" % (self.getA(), self.getB(), self.getTheta())
    }
}
