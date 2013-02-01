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

%include "lsst/afw/detection/detection_fwd.i"

//---------- Warning suppression ----------------------------------------------------------------------------

// Suppress swig complaints
// I had trouble getting %warnfilter to work; hence the pragmas
#pragma SWIG nowarn=314                 // print is a python keyword (--> _print)
#pragma SWIG nowarn=362                 // operator=  ignored

//---------- Dependencies that don't need to be seen by downstream imports ----------------------------------

%import "lsst/afw/utils.i" 
%import "lsst/afw/geom/Span.i"
%import "lsst/afw/image/Image.i"
%import "lsst/afw/image/Mask.i"
%import "lsst/afw/image/MaskedImage.i"
%import "lsst/afw/cameraGeom/cameraGeom_fwd.i"
%import "lsst/afw/geom/ellipses/ellipses_fwd.i"
%import "lsst/afw/math/kernel.i"
%import "lsst/afw/table/table_fwd.i"

//---------- ndarray and Eigen NumPy conversion typemaps ----------------------------------------------------

%{
#define PY_ARRAY_UNIQUE_SYMBOL LSST_AFW_DETECTION_NUMPY_ARRAY_API
#include "numpy/arrayobject.h"
#include "ndarray/swig.h"
#include "ndarray/swig/eigen.h"
%}
%include "ndarray.i"

%init %{
    import_array();
%}

%declareNumPyConverters(ndarray::Array<unsigned short,1,1>);
%declareNumPyConverters(ndarray::Array<float,1,1>);
// more in FootprintArray.i

//---------- STL Typemaps and Template Instantiations -------------------------------------------------------

%include "std_string.i"

%{
#include "lsst/afw/geom/Box.h"
%}

%template(VectorBox2I) std::vector<lsst::afw::geom::Box2I>;

//---------- afw::detection classes and functions -----------------------------------------------------------

%include "Footprint.i"
%include "FootprintArray.i"
%include "Psf.i"
%include "DoubleGaussianPsf.i"
%include "FootprintSet.i"

%pythoncode %{
from lsst.afw.geom import Span
%}
