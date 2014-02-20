// -*- lsst-c++ -*-

/* 
 * LSST Data Management System
 * Copyright 2008, 2009, 2010, 2011, 2012, 2013, 2014 LSST Corporation.
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
#include <vector>
#include "lsst/daf/base.h"
#include "lsst/afw/geom.h"
#define PY_ARRAY_UNIQUE_SYMBOL LSST_AFW_GEOM_NUMPY_ARRAY_API
#include "numpy/arrayobject.h"
#include "ndarray/swig.h"
#include "ndarray/swig/eigen.h"
%}

%init %{
    import_array();
%}

%include "lsst/p_lsstSwig.i"
%import "lsst/daf/base/baseLib.i"

%lsst_exceptions();

%template(Point2IVector) std::vector<lsst::afw::geom::Point2I>;
%template(Point2DVector) std::vector<lsst::afw::geom::Point2D>;

%include "ndarray.i"

%declareNumPyConverters(Eigen::Matrix<double,2,1,Eigen::DontAlign>);
%declareNumPyConverters(Eigen::Matrix<double,3,1,Eigen::DontAlign>);
%declareNumPyConverters(Eigen::Matrix<int,2,1,Eigen::DontAlign>);
%declareNumPyConverters(Eigen::Matrix<int,3,1,Eigen::DontAlign>);

%include "CoordinateBase.i"
%include "CoordinateExpr.i"
%include "Extent.i"
%include "Point.i"

%Extent_PREINCLUDE(int,2);
%Extent_PREINCLUDE(int,3);
%Extent_PREINCLUDE(double,2);
%Extent_PREINCLUDE(double,3);

%Point_PREINCLUDE(int,2);
%Point_PREINCLUDE(int,3);
%Point_PREINCLUDE(double,2);
%Point_PREINCLUDE(double,3);

%include "lsst/afw/geom/CoordinateBase.h"
%include "lsst/afw/geom/CoordinateExpr.h"
%include "lsst/afw/geom/Extent.h"
%include "lsst/afw/geom/Point.h"

%Extent_POSTINCLUDE(int,2,I);
%Extent_POSTINCLUDE(int,3,I);
%Extent_POSTINCLUDE(double,2,D);
%Extent_POSTINCLUDE(double,3,D);

%Point_POSTINCLUDE(int,2,I);
%Point_POSTINCLUDE(int,3,I);
%Point_POSTINCLUDE(double,2,D);
%Point_POSTINCLUDE(double,3,D);

%CoordinateExpr_POSTINCLUDE(2);
%CoordinateExpr_POSTINCLUDE(3);

%extend lsst::afw::geom::Point<int,2> {
    %template(Point2I) Point<double>;
    %pythoncode {
    def __reduce__(self):
        return (Point2I, (self.getX(), self.getY()))
    }
};

%extend lsst::afw::geom::Point<int,3> {
    %template(Point3I) Point<double>;
    %pythoncode {
    def __reduce__(self):
        return (Point3I, (self.getX(), self.getY(), self.getZ()))
    }
};

%extend lsst::afw::geom::Point<double,2> {
    %template(Point2D) Point<int>;
    %pythoncode {
    def __reduce__(self):
        return (Point2D, (self.getX(), self.getY()))
    }
};

%extend lsst::afw::geom::Point<double,3> {
    %template(Point3D) Point<int>;
    %pythoncode {
    def __reduce__(self):
        return (Point3D, (self.getX(), self.getY(), self.getZ()))
    }
};

%extend lsst::afw::geom::Extent<double,2> {
    %template(Extent2D) Extent<int>;
    %pythoncode {
    def __reduce__(self):
        return (Extent2D, (self.getX(), self.getY()))
    }
};

%extend lsst::afw::geom::Extent<double,3> {
    %template(Extent3D) Extent<int>;
    %pythoncode {
    def __reduce__(self):
        return (Extent3D, (self.getX(), self.getY(), self.getZ()))
    }
};

%extend lsst::afw::geom::Extent<int,2> {
    %pythoncode {
    def __reduce__(self):
        return (Extent2I, (self.getX(), self.getY()))
    }
};

%extend lsst::afw::geom::Extent<int,3> {
    %pythoncode {
    def __reduce__(self):
        return (Extent3I, (self.getX(), self.getY(), self.getZ()))
    }
};

%returnCopy(lsst::afw::geom::AffineTransform::getTranslation);
%returnCopy(lsst::afw::geom::AffineTransform::getLinear);

%include "LinearTransform.i"
%include "AffineTransform.i"
%include "Box.i"
%include "Angle.i"
%include "Span.i"
%include "XYTransform.i"
%include "TransformMap.i"

