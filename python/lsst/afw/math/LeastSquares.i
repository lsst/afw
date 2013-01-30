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

%include "lsst/afw/math/math_fwd.i"

%{
#include "lsst/afw/math/LeastSquares.h"
%}

%declareNumPyConverters(ndarray::Array<double,1,1>);
%declareNumPyConverters(ndarray::Array<double,2,2>);
%declareNumPyConverters(ndarray::Array<double,1,0>);
%declareNumPyConverters(ndarray::Array<double,2,0>);

%declareNumPyConverters(ndarray::Array<double const,1,1>);
%declareNumPyConverters(ndarray::Array<double const,2,2>);
%declareNumPyConverters(ndarray::Array<double const,1,0>);
%declareNumPyConverters(ndarray::Array<double const,2,0>);

%include "lsst/afw/math/LeastSquares.h"

%template(fromDesignMatrix) lsst::afw::math::LeastSquares::fromDesignMatrix<double,double,0,0>;
%template(setDesignMatrix) lsst::afw::math::LeastSquares::setDesignMatrix<double,double,0,0>;
%template(fromNormalEquations) lsst::afw::math::LeastSquares::fromNormalEquations<double,double,0,0>;
%template(setNormalEquations) lsst::afw::math::LeastSquares::setNormalEquations<double,double,0,0>;
