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
 
%define ellipsesLib_DOCSTRING
"
Python interface to lsst::afw::geom::ellipses classes and functions
"
%enddef

%feature("autodoc", "1");
%module(package="lsst.afw.geom.ellipses", docstring=ellipsesLib_DOCSTRING) ellipsesLib

%{
#include "lsst/afw/geom/ellipses.h"
#define PY_ARRAY_UNIQUE_SYMBOL LSST_AFW_GEOM_ELLIPSES_NUMPY_ARRAY_API
#include "numpy/arrayobject.h"
#include "lsst/ndarray/python.h"
#include "lsst/ndarray/python/eigen.h"
%}

%init%{
    import_array();
%}

%include "lsst/p_lsstSwig.i"



%pythoncode %{
import lsst.utils

def version(HeadURL = r"$HeadURL$"):
    """Return a version given a HeadURL string. If a different version is setup, return that too"""

    version_svn = lsst.utils.guessSvnVersion(HeadURL)

    try:
        import eups
    except ImportError:
        return version_svn
    else:
        try:
            version_eups = eups.getSetupVersion("afw")
        except AttributeError:
            return version_svn

    if version_eups == version_svn:
        return version_svn
    else:
        return "%s (setup: %s)" % (version_svn, version_eups)

%}

%lsst_exceptions();

%include "lsst/ndarray/ndarray.i"
%import "lsst/afw/geom/geomLib.i"


%ignore lsst::afw::geom::ellipses::BaseCore::transform;
%ignore lsst::afw::geom::ellipses::BaseCore::convolve;
%ignore lsst::afw::geom::ellipses::BaseCore::getGridTransform;
%ignore lsst::afw::geom::ellipses::BaseCore::readParameters;
%ignore lsst::afw::geom::ellipses::BaseCore::writeParameters;
%ignore lsst::afw::geom::ellipses::BaseCore::as;
%rename(assign) lsst::afw::geom::ellipses::BaseCore::operator=;

%declareNumPyConverters(lsst::afw::geom::ellipses::BaseCore::Jacobian);
%declareNumPyConverters(lsst::afw::geom::ellipses::BaseCore::ParameterVector);

SWIG_SHARED_PTR(BaseCorePtr, lsst::afw::geom::ellipses::BaseCore);

%include "lsst/afw/geom/ellipses/BaseCore.h"

%define %EllipseCore_PREINCLUDE(NAME)
%feature(notabstract) lsst::afw::geom::ellipses::NAME;
%implicitconv lsst::afw::geom::ellipses::NAME;
SWIG_SHARED_PTR_DERIVED(
    NAME##Ptr,
    lsst::afw::geom::ellipses::BaseCore,
    lsst::afw::geom::ellipses::NAME 
);
%ignore lsst::afw::geom::ellipses::NAME::writeParameters;
%ignore lsst::afw::geom::ellipses::NAME::readParameters;
%rename(assign) lsst::afw::geom::ellipses::NAME::operator=;
%enddef

%define %EllipseCore_POSTINCLUDE(NAME)
%extend lsst::afw::geom::ellipses::NAME {
    lsst::afw::geom::ellipses::NAME::Ptr _transform(lsst::afw::geom::LinearTransform const & t) {
       return boost::static_pointer_cast<lsst::afw::geom::ellipses::NAME>(self->transform(t).copy());
    }
    void _transformInPlace(lsst::afw::geom::LinearTransform const & t) {
       self->transform(t).inPlace();
    }
    lsst::afw::geom::LinearTransform _getGridTransform() {
        return self->getGridTransform();
    }
    lsst::afw::geom::ellipses::NAME::Ptr _convolve(lsst::afw::geom::ellipses::BaseCore const & other) {
       return boost::static_pointer_cast<lsst::afw::geom::ellipses::NAME>(self->convolve(other).copy());
    }
    static lsst::afw::geom::ellipses::NAME::Ptr cast(lsst::afw::geom::ellipses::BaseCore::Ptr const & p) {
       return boost::dynamic_pointer_cast<lsst::afw::geom::ellipses::NAME>(p);
    }
    %pythoncode {
    def transform(self, t): return self._transform(t)
    def transformInPlace(self, t): self._transformInPlace(t)
    def convolve(self, t): return self._convolve(t)
    def getGridTransform(self): return self._getGridTransform()
    }
}
%enddef

%EllipseCore_PREINCLUDE(Axes);
%EllipseCore_PREINCLUDE(Quadrupole);

%include "lsst/afw/geom/ellipses/Axes.h"
%include "lsst/afw/geom/ellipses/Quadrupole.h"

%EllipseCore_POSTINCLUDE(Axes);
%EllipseCore_POSTINCLUDE(Quadrupole);

%extend lsst::afw::geom::ellipses::Axes {
    %pythoncode {
    def __repr__(self):
        return "Axes(a=%r, b=%r, theta=%r)" % (self.getA(), self.getB(), self.getTheta())
    def __str__(self):
        return "(a=%s, b=%s, theta=%s)" % (self.getA(), self.getB(), self.getTheta())
    }
}

%extend lsst::afw::geom::ellipses::Quadrupole {
    %pythoncode {
    def __repr__(self):
        return "Quadrupole(ixx=%r, iyy=%r, ixy=%r)" % (self.getIXX(), self.getIYY(), self.getIXY())
    def __str__(self):
        return "(ixx=%s, iyy=%s, ixy=%s)" % (self.getIXX(), self.getIYY(), self.getIXY())
    }
}

%include "Separable.i"

%feature("valuewrapper") lsst::afw::geom::ellipses::Ellipse;
%ignore lsst::afw::geom::ellipses::Ellipse::getCore;
%ignore lsst::afw::geom::ellipses::Ellipse::transform;
%ignore lsst::afw::geom::ellipses::Ellipse::getGridTransform;
%ignore lsst::afw::geom::ellipses::Ellipse::readParameters;
%ignore lsst::afw::geom::ellipses::Ellipse::writeParameters;

%rename(assign) lsst::afw::geom::ellipses::Ellipse::operator=;
%rename(_getCorePtr) lsst::afw::geom::ellipses::Ellipse::getCorePtr;

SWIG_SHARED_PTR(EllipsePtr, lsst::afw::geom::ellipses::Ellipse);
%declareNumPyConverters(lsst::afw::geom::ellipses::Ellipse::ParameterVector);

%include "lsst/afw/geom/ellipses/Ellipse.h"

%extend lsst::afw::geom::ellipses::Ellipse {
    lsst::afw::geom::ellipses::Ellipse _transform(lsst::afw::geom::AffineTransform const & t) {
        return self->transform(t);
    }
    void _transformInPlace(lsst::afw::geom::AffineTransform const & t) {
        self->transform(t).inPlace();
    }
    lsst::afw::geom::AffineTransform _getGridTransform() {
        return self->getGridTransform();
    }
    %pythoncode {
    def transform(self, t): return self._transform(t)
    def transformInPlace(self, t): self._transformInPlace(t)
    def getGridTransform(self, t): return self._getGridTransform(t)
    def getCore(self): return self._getCorePtr().cast()
    def __repr__(self):
        return "Ellipse(%r, %r)" % (self.getCore(), self.getCenter())
    def __str__(self):
        return "(%s, %s)" % (self.getCore(), self.getCenter())
    }
}

%include "lsst/afw/geom/ellipses/Parametric.h"
