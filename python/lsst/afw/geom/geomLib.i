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

%include "lsst/afw/geom/geom_fwd.i"

%lsst_exceptions();

//----- NumPy typemaps --------------------------------------------------------------------------------------

%{
#define PY_ARRAY_UNIQUE_SYMBOL LSST_AFW_GEOM_NUMPY_ARRAY_API
#include "numpy/arrayobject.h"
#include "ndarray/swig.h"
#include "ndarray/swig/eigen.h"
%}

%init %{
    import_array();
%}

%include "ndarray.i"

%declareNumPyConverters(Eigen::Vector2d);
%declareNumPyConverters(Eigen::Matrix2d);
%declareNumPyConverters(Eigen::Matrix3d);
%declareNumPyConverters(Eigen::Matrix<double,2,1,Eigen::DontAlign>);
%declareNumPyConverters(Eigen::Matrix<double,3,1,Eigen::DontAlign>);
%declareNumPyConverters(Eigen::Matrix<int,2,1,Eigen::DontAlign>);
%declareNumPyConverters(Eigen::Matrix<int,3,1,Eigen::DontAlign>);
%declareNumPyConverters(lsst::afw::geom::LinearTransform::ParameterVector);
%declareNumPyConverters(lsst::afw::geom::LinearTransform::Matrix);
%declareNumPyConverters(lsst::afw::geom::AffineTransform::ParameterVector);
%declareNumPyConverters(lsst::afw::geom::AffineTransform::Matrix);

//----- Wrapped classes and functions -----------------------------------------------------------------------

%include "CoordinateBase.i"
%include "CoordinateExpr.i"
%include "Extent.i"
%include "Point.i"
%include "LinearTransform.i"
%include "AffineTransform.i"
%include "Box.i"
%include "Angle.i"
%include "Span.i"
