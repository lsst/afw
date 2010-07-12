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
 
%define geomLib_DOCSTRING
"
Python interface to lsst::afw::geom classes
"
%enddef

%feature("autodoc", "1");
%module(package="lsst.afw.geom",docstring=geomLib_DOCSTRING) geomLib

#pragma SWIG nowarn=381                 // operator&&  ignored
#pragma SWIG nowarn=382                 // operator||  ignored
#pragma SWIG nowarn=361                 // operator!  ignored
#pragma SWIG nowarn=503                 // comparison operators ignored

%{
#include "lsst/afw/geom.h"
#define PY_ARRAY_UNIQUE_SYMBOL LSST_AFW_GEOM_NUMPY_ARRAY_API
#include "numpy/arrayobject.h"
#include "lsst/afw/numpyTypemaps.h"
%}


%init %{
    import_array();
%}

%include "lsst/p_lsstSwig.i"

%pythoncode %{
import lsst.utils

def version(HeadURL = r"$HeadURL: svn+ssh://svn.lsstcorp.org/DMS/afw/trunk/python/lsst/afw/geom/geomLib.i $"):
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

%lsst_exceptions();

%import "lsst/afw/eigen.i"

%declareEigenMatrix(Eigen::Matrix<bool,2,1,Eigen::DontAlign>);
%declareEigenMatrix(Eigen::Matrix<int,2,1,Eigen::DontAlign>);
%declareEigenMatrix(Eigen::Matrix<double,2,1,Eigen::DontAlign>);

%declareEigenMatrix(Eigen::Matrix<bool,3,1,Eigen::DontAlign>);
%declareEigenMatrix(Eigen::Matrix<int,3,1,Eigen::DontAlign>);
%declareEigenMatrix(Eigen::Matrix<double,3,1,Eigen::DontAlign>);

%declareEigenMatrix(Eigen::Matrix2d);
%declareEigenMatrix(Eigen::Matrix3d);
%declareEigenMatrix(lsst::afw::geom::AffineTransform::Matrix);
%declareEigenMatrix(lsst::afw::geom::LinearTransform::Matrix);

%include "CoordinateBase.i"
%include "CoordinateExpr.i"
%include "Extent.i"
%include "Point.i"

%include "lsst/afw/geom/CoordinateBase.h"
%include "lsst/afw/geom/CoordinateExpr.h"
%include "lsst/afw/geom/Extent.h"
%include "lsst/afw/geom/Point.h"

%CoordinateBase_POSTINCLUDE_2(bool, CoordinateExpr2, lsst::afw::geom::CoordinateExpr<2>);
%CoordinateBase_POSTINCLUDE_3(bool, CoordinateExpr3, lsst::afw::geom::CoordinateExpr<3>);

%CoordinateBase_POSTINCLUDE_2(int, Extent2I, lsst::afw::geom::Extent<int,2>);
%CoordinateBase_POSTINCLUDE_3(int, Extent3I, lsst::afw::geom::Extent<int,3>);

%CoordinateBase_POSTINCLUDE_2(double, Extent2D, lsst::afw::geom::Extent<double,2>);
%CoordinateBase_POSTINCLUDE_3(double, Extent3D, lsst::afw::geom::Extent<double,3>);

%CoordinateBase_POSTINCLUDE_2(int, Point2I, lsst::afw::geom::Point<int,2>);
%CoordinateBase_POSTINCLUDE_3(int, Point3I, lsst::afw::geom::Point<int,3>);

%CoordinateBase_POSTINCLUDE_2(double, Point2D, lsst::afw::geom::Point<double,2>);
%CoordinateBase_POSTINCLUDE_3(double, Point3D, lsst::afw::geom::Point<double,3>);

%include "LinearTransform.i"
%include "AffineTransform.i"
%include "Box.i"
%include "ellipses.i"
