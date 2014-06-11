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
 
%define mathLib_DOCSTRING
"
Python interface to lsst::afw::math classes
"
%enddef

%feature("autodoc", "1");
%module(package="lsst.afw.math",docstring=mathLib_DOCSTRING) mathLib

%{
#   include "lsst/daf/base.h"
#   include "lsst/pex/logging.h"
#   include "lsst/pex/policy.h"
#   include "lsst/afw/image.h"
#   include "lsst/afw/geom.h"
#   include "lsst/afw/math.h"
#   include "lsst/afw/cameraGeom.h"

#   pragma clang diagnostic ignored "-Warray-bounds" // PyTupleObject has an array declared as [1]
%}

// Enable ndarray's NumPy typemaps; types are declared in %included files.
%{
#define PY_ARRAY_UNIQUE_SYMBOL LSST_AFW_MATH_NUMPY_ARRAY_API
#include "numpy/arrayobject.h"
#include "ndarray/swig.h"
#include "ndarray/swig/eigen.h"
%}
%init %{
    import_array();
%}
%include "ndarray.i"

%include "lsst/p_lsstSwig.i"

// vectors of plain old types; template vectors of more complex types in objectVectors.i
%template(vectorF) std::vector<float>;
%template(vectorD) std::vector<double>;
%template(vectorI) std::vector<int>;
%template(vectorVectorF) std::vector<std::vector<float> >;
%template(vectorVectorD) std::vector<std::vector<double> >;
%template(vectorVectorI) std::vector<std::vector<int> >;

%import "lsst/afw/image/imageLib.i"

%lsst_exceptions();


%include "function.i"
%include "kernel.i"
%include "minimize.i"
%include "statistics.i"
%include "interpolate.i"
%include "background.i"
%include "warpExposure.i"
%include "spatialCell.i"
%include "random.i"
%include "stack.i"
%include "GaussianProcess.i"
%include "objectVectors.i" // must come last

%include "LeastSquares.i"
%include "BoundedField.i"
%include "ChebyshevBoundedField.i"




%inline %{
    struct InitGsl {
        InitGsl() {
            static int first = true;

            if (first) {
                (void)gsl_set_error_handler_off();
            }
        }
    };

    InitGsl _initGsl;                   // created at import time, to initialise the GSL library
%}
