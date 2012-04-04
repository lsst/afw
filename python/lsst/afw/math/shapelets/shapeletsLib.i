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
 
%define shapeletsLib_DOCSTRING
"
Python interface to lsst::afw::math::shapelets classes and functions
"
%enddef

%feature("autodoc", "1");
%module(package="lsst.afw.math.shapelets", docstring=shapeletsLib_DOCSTRING) shapeletsLib

%{
#   include "lsst/afw/geom.h"
#   include "lsst/afw/detection.h"
#   include "lsst/afw/image.h"
#   include "lsst/afw/cameraGeom.h"
#   include "lsst/pex/logging.h"
#   include "lsst/afw/math/shapelets.h"
%}

%include "lsst/p_lsstSwig.i"
%include "std_list.i"

%{
#include "lsst/afw/geom.h"
#define PY_ARRAY_UNIQUE_SYMBOL LSST_AFW_MATH_SHAPELETS_NUMPY_ARRAY_API
#include "numpy/arrayobject.h"
#include "ndarray/swig.h"
#include "ndarray/swig/eigen.h"
%}

%init %{
    import_array();
%}

%include "ndarray.i"

%declareNumPyConverters(Eigen::MatrixXd);
%declareNumPyConverters(ndarray::Array<lsst::afw::math::shapelets::Pixel,1>);
%declareNumPyConverters(ndarray::Array<lsst::afw::math::shapelets::Pixel,1,1>);
%declareNumPyConverters(ndarray::Array<lsst::afw::math::shapelets::Pixel const,1,1>);
%declareNumPyConverters(ndarray::Array<lsst::afw::math::shapelets::Pixel const,2,-2>);
%declareNumPyConverters(ndarray::Array<lsst::afw::math::shapelets::Pixel,3,-3>);
%declareNumPyConverters(Eigen::Matrix<lsst::afw::math::shapelets::Pixel,5,Eigen::Dynamic>);

%feature(valuewrapper) lsst::afw::math::shapelets::ShapeletFunction;
%feature(valuewrapper) lsst::afw::math::shapelets::MultiShapeletFunction;
%template(MultiShapeletElementList) std::list<lsst::afw::math::shapelets::ShapeletFunction>;

%import "lsst/afw/geom/geomLib.i"
%import "lsst/afw/geom/ellipses/ellipsesLib.i"
%import "lsst/afw/image/imageLib.i"
%import "lsst/afw/detection/detectionLib.i"

%lsst_exceptions();

%include "lsst/afw/math/shapelets/constants.h"
%include "lsst/afw/math/shapelets/ConversionMatrix.h"
%include "lsst/afw/math/shapelets/ShapeletFunction.h"
%include "lsst/afw/math/shapelets/MultiShapeletFunction.h"
%include "lsst/afw/math/shapelets/BasisEvaluator.h"

%returnCopy(lsst::afw::math::shapelets::ModelBuilder::getRegion)

%include "lsst/afw/math/shapelets/ModelBuilder.h"

%template(ModelBuilder) lsst::afw::math::shapelets::ModelBuilder::ModelBuilder<float>;
%template(ModelBuilder) lsst::afw::math::shapelets::ModelBuilder::ModelBuilder<double>;
%template(addToImage) lsst::afw::math::shapelets::ModelBuilder::addToImage<float>;
%template(addToImage) lsst::afw::math::shapelets::ModelBuilder::addToImage<double>;
