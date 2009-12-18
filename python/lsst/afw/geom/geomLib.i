// -*- lsst-c++ -*-
%define geomLib_DOCSTRING
"
Python interface to lsst::afw::geom classes
"
%enddef

%feature("autodoc", "1");
%module(package="lsst.afw.geom",docstring=geomLib_DOCSTRING) geomLib

%{
#include "lsst/afw/geom.h"
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

%include "AffineTransform.i"
%include "ellipses.i"
