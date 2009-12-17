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

    //typedef lsst::afw::geom::ellipses::detail::CoreImpl< lsst::afw::geom::ellipses::Axes, lsst::afw::geom::ellipses::AxesEllipse > AxesImpl;
    //typedef lsst::afw::geom::ellipses::detail::EllipseImpl< lsst::afw::geom::ellipses::Axes, lsst::afw::geom::ellipses::AxesEllipse > AxesEllipseImpl;
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

%ignore lsst::afw::geom::ellipses::BaseEllipse::operator[];
%ignore lsst::afw::geom::ellipses::BaseEllipse::getVector;
%ignore lsst::afw::geom::ellipses::BaseEllipse::setVector;
%ignore lsst::afw::geom::ellipses::BaseEllipse::transform;
%ignore lsst::afw::geom::ellipses::BaseCore::operator[];
%ignore lsst::afw::geom::ellipses::BaseCore::getVector;
%ignore lsst::afw::geom::ellipses::BaseCore::setVector;
%ignore lsst::afw::geom::ellipses::BaseCore::transform;

%rename(set) lsst::afw::geom::ellipses::BaseEllipse::operator=;
%rename(set) lsst::afw::geom::ellipses::BaseCore::operator=;

%include "boost_shared_ptr.i"

SWIG_SHARED_PTR(BaseEllipsePtr, lsst::afw::geom::ellipses::BaseEllipse);
SWIG_SHARED_PTR(BaseCorePtr, lsst::afw::geom::ellipses::BaseCore);

%include "lsst/afw/geom/ellipses/BaseEllipse.h"

%extend lsst::afw::geom::ellipses::BaseCore {
    double _getitem_nochecking(int i) { return self->operator[](i); }
    void _setitem_nochecking(int i, double value) {
        self->operator[](i) = value;
    }
    %pythoncode {
    def __len__(self):
        return 3
    def __getitem__(self, k):
        if k < 0 or k > 2: raise IndexError(k)
        return self._getitem_nochecking(k)
    def __setitem__(self, k, v):
        if k < 0 or k > 2: raise IndexError(k)
        self._setitem_nochecking(k, v)
    }
}

%extend lsst::afw::geom::ellipses::BaseEllipse {
    double _getitem_nochecking(int i) { return self->operator[](i); }
    void _setitem_nochecking(int i, double value) {
        self->operator[](i) = value;
    }
    void setCore(lsst::afw::geom::ellipses::BaseCore const & core) {
    	self->getCore() = core;
    }
    %pythoncode {
    def __len__(self):
        return 5
    def __getitem__(self, k):
        if k < 0 or k > 4: raise IndexError(k)
        return self.get(k)
    def __setitem__(self, k, v):
        if k < 0 or k > 4: raise IndexError(k)
        self.set(k, v)
    }
}

%define %Ellipse_PREINCLUDE(NAME)
SWIG_SHARED_PTR_DERIVED(
    NAME ## EllipseImplPtr,
    lsst::afw::geom::ellipses::BaseEllipse,
    lsst::afw::geom::ellipses::detail::EllipseImpl< lsst::afw::geom::ellipses::NAME, lsst::afw::geom::ellipses::NAME ## Ellipse >
)
SWIG_SHARED_PTR_DERIVED(
    NAME ## ImplPtr,
    lsst::afw::geom::ellipses::BaseCore,
    lsst::afw::geom::ellipses::detail::CoreImpl< lsst::afw::geom::ellipses::NAME, lsst::afw::geom::ellipses::NAME ## Ellipse >
)
%enddef

%define %Ellipse_MIDINCLUDE(NAME)
%template(NAME ## EllipseImpl) lsst::afw::geom::ellipses::detail::EllipseImpl< lsst::afw::geom::ellipses::NAME, lsst::afw::geom::ellipses::NAME ## Ellipse >;
%template(NAME ## Impl) lsst::afw::geom::ellipses::detail::CoreImpl< lsst::afw::geom::ellipses::NAME, lsst::afw::geom::ellipses::NAME ## Ellipse >;
SWIG_SHARED_PTR_DERIVED(
    NAME ## EllipsePtr,
    lsst::afw::geom::ellipses::BaseEllipse,
    lsst::afw::geom::ellipses::NAME ## Ellipse
)
SWIG_SHARED_PTR_DERIVED(
    NAME ## Ptr,
    lsst::afw::geom::ellipses::BaseCore,
    lsst::afw::geom::ellipses::NAME 
)
%rename(set) lsst::afw::geom::ellipses::NAME ## Ellipse::operator=;
%rename(set) lsst::afw::geom::ellipses::NAME::operator=;
%enddef

%define %Ellipse_POSTINCLUDE(NAME)
%extend lsst::afw::geom::ellipses::NAME {
    %pythoncode { 
    def __repr__(self):
        return "NAME(%.10g,%.10g,%.10g)" % tuple(self)
    def __str__(self):
        return "NAME(%g,%g,%g)" % tuple(self)
    }
}
%extend lsst::afw::geom::ellipses::NAME ## Ellipse {
    %pythoncode {
    def __repr__(self):
        return "NAME ## Ellipse(%r,%r)" % (self.getCore(), self.getCenter())
    def __str__(self):
        return "(%s,%s)" % (self.getCore(), self.getCenter())
    }
}
%enddef

%Ellipse_PREINCLUDE(Axes);
%Ellipse_PREINCLUDE(Distortion);
%Ellipse_PREINCLUDE(LogShear);
%Ellipse_PREINCLUDE(Quadrupole);

%include "lsst/afw/geom/ellipses/EllipseImpl.h"

%Ellipse_MIDINCLUDE(Axes);
%Ellipse_MIDINCLUDE(Distortion);
%Ellipse_MIDINCLUDE(LogShear);
%Ellipse_MIDINCLUDE(Quadrupole);

%include "lsst/afw/geom/ellipses/Axes.h"
%include "lsst/afw/geom/ellipses/Distortion.h"
%include "lsst/afw/geom/ellipses/Quadrupole.h"
%include "lsst/afw/geom/ellipses/LogShear.h"

%Ellipse_POSTINCLUDE(Axes);
%Ellipse_POSTINCLUDE(Distortion);
%Ellipse_POSTINCLUDE(LogShear);
%Ellipse_POSTINCLUDE(Quadrupole);
