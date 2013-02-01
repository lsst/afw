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

%include "lsst/afw/geom/ellipses/ellipses_fwd.i"

%{
#define PY_ARRAY_UNIQUE_SYMBOL LSST_AFW_GEOM_ELLIPSES_NUMPY_ARRAY_API
#include "numpy/arrayobject.h"
#include "ndarray/swig.h"
#include "ndarray/swig/eigen.h"
%}

%init%{
    import_array();
%}

%include "ndarray.i"

%declareNumPyConverters(lsst::afw::geom::ellipses::BaseCore::Jacobian);
%declareNumPyConverters(lsst::afw::geom::ellipses::BaseCore::ParameterVector);
%declareNumPyConverters(lsst::afw::geom::ellipses::Ellipse::ParameterVector);
%declareNumPyConverters(lsst::afw::geom::ellipses::EllipticityBase::Jacobian);
%declareNumPyConverters(lsst::afw::geom::ellipses::Quadrupole::Matrix);

%include "lsst/afw/geom/ellipses/BaseCore.i"
%include "lsst/afw/geom/ellipses/Ellipse.i"
%include "lsst/afw/geom/ellipses/Axes.i"
%include "lsst/afw/geom/ellipses/Quadrupole.i"
%include "lsst/afw/geom/ellipses/Separable.i"
%include "lsst/afw/geom/ellipses/Parametric.i"
%include "lsst/afw/geom/ellipses/PixelRegion.i"
