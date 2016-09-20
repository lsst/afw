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

%define tableLib_DOCSTRING
"
Python interface to lsst::afw::table classes
"
%enddef

%feature("autodoc", "1");
%module(package="lsst.afw.table", docstring=tableLib_DOCSTRING) tableLib

#pragma SWIG nowarn=389                 // operator[]  ignored
#pragma SWIG nowarn=503                 // comparison operators ignored
#pragma SWIG nowarn=520                 // base class not similarly marked as smart pointer
#pragma SWIG nowarn=401                 // nothing known about base class
#pragma SWIG nowarn=302                 // redefine identifier (SourceSet<> -> SourceSet)

%include "lsst/p_lsstSwig.i"

%initializeNumPy(afw_table)
%{
#include "ndarray/swig.h"
#include "ndarray/converter/eigen.h"
%}
%include "ndarray.i"

%lsst_exceptions()

%include "Base.i"
%include "aggregates.i"
%include "Simple.i"
%include "Source.i"
%include "Match.i"
%include "Exposure.i"
%include "AmpInfo.i"
%include "arrays.i"
