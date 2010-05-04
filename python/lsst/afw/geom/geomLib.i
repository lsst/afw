// -*- lsst-c++ -*-
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
%include "Extent.i"
%include "Point.i"

%include "lsst/afw/geom/CoordinateBase.h"

%include "lsst/afw/geom/CoordinateExpr.h"
%include "lsst/afw/geom/Extent.h"
%include "lsst/afw/geom/Point.h"

%template(CoordinateExprBase2) lsst::afw::geom::CoordinateBase<lsst::afw::geom::CoordinateExpr<2>,bool,2>;
%template(CoordinateExprBase3) lsst::afw::geom::CoordinateBase<lsst::afw::geom::CoordinateExpr<3>,bool,3>;
%template(CoordinateExpr2) lsst::afw::geom::CoordinateExpr<2>;
%template(CoordinateExpr3) lsst::afw::geom::CoordinateExpr<3>;
%CoordinateBase_POSTINCLUDE(bool, 2, lsst::afw::geom::CoordinateExpr<2>);
%CoordinateBase_POSTINCLUDE(bool, 3, lsst::afw::geom::CoordinateExpr<3>);

%Point_POSTINCLUDE(double, 2, D);
%Point_POSTINCLUDE(double, 3, D);
%Point_POSTINCLUDE(int, 2, I);
%Point_POSTINCLUDE(int, 3, I);

%Extent_POSTINCLUDE(double, 2, D);
%Extent_POSTINCLUDE(double, 3, D);
%Extent_POSTINCLUDE(int, 2, I);
%Extent_POSTINCLUDE(int, 3, I);

%include "LinearTransform.i"
%include "AffineTransform.i"
%include "Box.i"
%include "ellipses.i"
