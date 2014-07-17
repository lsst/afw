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
 
%define gpuLib_DOCSTRING
"
Python interface to lsst::afw::gpu classes
"
%enddef

%feature("autodoc", "1");
%module(package="lsst.afw.gpu",docstring=gpuLib_DOCSTRING) gpuLib

%{
#include "lsst/afw/gpu/DevicePreference.h"
#include "lsst/afw/gpu/GpuExceptions.h"
#include "lsst/afw/gpu/IsGpuBuild.h"
#include "lsst/pex/exceptions.h"
%}

%import "lsst/pex/exceptions/exceptionsLib.i"

%include "lsst/p_lsstSwig.i"

%lsst_exceptions();

%include "lsst/afw/gpu/DevicePreference.h"
%include "lsst/afw/gpu/GpuExceptions.h"
%include "lsst/afw/gpu/IsGpuBuild.h"

%declareException(GpuMemoryError, lsst.pex.exceptions.RuntimeError,
                  lsst::afw::gpu::GpuMemoryError)
%declareException(GpuRuntimeError, lsst.pex.exceptions.RuntimeError,
                  lsst::afw::gpu::GpuRuntimeError)
