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

%feature("autodoc", "1");
%module(package="lsst.afw.geom.ellipses") ellipsesLib

%include "lsst/p_lsstSwig.i"

%lsst_exceptions();

%include "lsst/afw/geom/ellipses_fwd.h"

%shared_ptr(lsst::afw::geom::ellipses::BaseCore);
%shared_ptr(lsst::afw::geom::ellipses::Ellipse);
%shared_ptr(lsst::afw::geom::ellipses::Axes);
%shared_ptr(lsst::afw::geom::ellipses::Quadrupole);


%define %Separable_PREINCLUDE(ELLIPTICITY, RADIUS)
%shared_ptr(
    lsst::afw::geom::ellipses::Separable<
        lsst::afw::geom::ellipses::ELLIPTICITY, 
        lsst::afw::geom::ellipses::RADIUS
    > 
);
%rename(assign) lsst::afw::geom::ellipses::Separable<
        lsst::afw::geom::ellipses::ELLIPTICITY,
        lsst::afw::geom::ellipses::RADIUS
    >::operator=;
%enddef

%Separable_PREINCLUDE(Distortion, DeterminantRadius);
%Separable_PREINCLUDE(Distortion, TraceRadius);
%Separable_PREINCLUDE(Distortion, LogDeterminantRadius);
%Separable_PREINCLUDE(Distortion, LogTraceRadius);

%Separable_PREINCLUDE(ConformalShear, DeterminantRadius);
%Separable_PREINCLUDE(ConformalShear, TraceRadius);
%Separable_PREINCLUDE(ConformalShear, LogDeterminantRadius);
%Separable_PREINCLUDE(ConformalShear, LogTraceRadius);

%Separable_PREINCLUDE(ReducedShear, DeterminantRadius);
%Separable_PREINCLUDE(ReducedShear, TraceRadius);
%Separable_PREINCLUDE(ReducedShear, LogDeterminantRadius);
%Separable_PREINCLUDE(ReducedShear, LogTraceRadius);
