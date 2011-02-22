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
#   include "lsst/afw/math/shapelets.h"
%}

%include "lsst/p_lsstSwig.i"

%{
#include "lsst/afw/geom.h"
#define PY_ARRAY_UNIQUE_SYMBOL LSST_AFW_MATH_SHAPELETS_NUMPY_ARRAY_API
#include "numpy/arrayobject.h"
#include "lsst/ndarray/python.h"
#include "lsst/ndarray/python/eigen.h"
%}

%init %{
    import_array();
%}

%include "lsst/ndarray/ndarray.i"

%declareNumPyConverters(Eigen::MatrixXd);
%declareNumPyConverters(lsst::ndarray::Array<lsst::afw::math::shapelets::Pixel,1>);
%declareNumPyConverters(lsst::ndarray::Array<lsst::afw::math::shapelets::Pixel,1,1>);

%pythoncode %{
import lsst.utils

def version(HeadURL = r"$HeadURL: svn+ssh://svn.lsstcorp.org/DMS/afw/trunk/python/lsst/afw/math/shapelets/shapeletsLib.i $"):
    """Return a version given a HeadURL string. If a different version is setup, return that too"""

    version_svn = lsst.utils.guessSvnVersion(HeadURL)

    try:
        import eups
    except ImportError:
        return version_svn
    else:
        try:
            version_eups = eups.setup("afw")
        except AttributeError:
            return version_svn

    if version_eups == version_svn:
        return version_svn
    else:
        return "%s (setup: %s)" % (version_svn, version_eups)

%}

%import "lsst/afw/geom/geomLib.i"
%import "lsst/afw/geom/ellipses/ellipsesLib.i"

%lsst_exceptions();

%include "lsst/afw/math/shapelets/constants.h"
%include "lsst/afw/math/shapelets/ConversionMatrix.h"
%include "lsst/afw/math/shapelets/ShapeletFunction.h"
