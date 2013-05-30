// -*- lsst-c++ -*-
/*
 * LSST Data Management System
 * Copyright 2008-2013 LSST Corporation.
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
#include "lsst/afw/geom.h"
#include "lsst/afw/geom/ellipses.h"
#define PY_ARRAY_UNIQUE_SYMBOL LSST_AFW_GEOM_ELLIPSES_NUMPY_ARRAY_API
#include "numpy/arrayobject.h"
#include "ndarray/swig.h"
#include "ndarray/swig/eigen.h"
%}

%init%{
    import_array();
%}

%include "lsst/p_lsstSwig.i"

%lsst_exceptions();

%include "ndarray.i"
%import "lsst/afw/geom/geomLib.i"

%pythondynamic lsst::afw::geom::ellipses::Ellipse;

%ignore lsst::afw::geom::ellipses::EllipseCore::transform;
%ignore lsst::afw::geom::ellipses::EllipseCore::convolve;
%ignore lsst::afw::geom::ellipses::EllipseCore::getGridTransform;
%ignore lsst::afw::geom::ellipses::EllipseCore::readParameters;
%ignore lsst::afw::geom::ellipses::EllipseCore::writeParameters;
%ignore lsst::afw::geom::ellipses::EllipseCore::as;
%rename(assign) lsst::afw::geom::ellipses::EllipseCore::operator=;

%declareNumPyConverters(lsst::afw::geom::ellipses::Quadrupole::Matrix);
%declareNumPyConverters(lsst::afw::geom::ellipses::EllipseCore::Jacobian);
%declareNumPyConverters(lsst::afw::geom::ellipses::EllipseCore::ParameterVector);
%declareNumPyConverters(lsst::afw::geom::ellipses::EllipseCore::Transformer::DerivativeMatrix);
%declareNumPyConverters(lsst::afw::geom::ellipses::EllipseCore::Transformer::TransformDerivativeMatrix);

%shared_ptr(lsst::afw::geom::ellipses::EllipseCore);

%include "lsst/afw/geom/ellipses/EllipseCore.h"

%addStreamRepr(lsst::afw::geom::ellipses::EllipseCore);
%extend lsst::afw::geom::ellipses::EllipseCore {
    %pythoncode %{
        def transform(self, transform, inPlace=False, doDerivatives=False):
            """Transform an EllipseCore via a LinearTransform

            If inPlace, self will be transformed in-place and returned;
            otherwise, a new transformed ellipse with the same EllipseCore type
            will be returned.

            If doDerivatives, a tuple of (transformed, dEllipse, dTransform) will be
            returned, where dEllipse is the derivative of the transformed ellipse
            w.r.t. the input ellipse, and dTransform is the derivative of the
            transformed ellipse w.r.t. the LinearTransform elements.
            """
            if doDerivatives:
                dEllipse = _ellipsesLib.EllipseCore__transformDEllipse(self, transform)
                dTransform = _ellipsesLib.EllipseCore__transformDTransform(self, transform)
            if inPlace:
                r = self
            else:
                r = self.clone()
            _ellipsesLib.EllipseCore__transformInPlace(r, transform)
            if doDerivatives:
                return r, dEllipse, dTransform
            else:
                return r
    %}

    %feature("shadow") as_ %{
        def as_(self, cls):
            """Return a new EllipseCore equivalent to self, with type specified
            by the given string name or type object.
            """
            if isinstance(cls, basestring):
                return $action(self, cls)
            return cls(self)
    %}

    PTR(lsst::afw::geom::ellipses::EllipseCore) as_(std::string const & name) {
        return self->as(name);
    }

    %feature("shadow") _transformInPlace %{%}
    %feature("shadow") _transformDEllipse %{%}
    %feature("shadow") _transformDTransform %{%}

    void _transformInPlace(lsst::afw::geom::LinearTransform const & t) {
       self->transform(t).inPlace();
    }
    lsst::afw::geom::ellipses::EllipseCore::Transformer::DerivativeMatrix _transformDEllipse(
        lsst::afw::geom::LinearTransform const & t
    ) const {
        return self->transform(t).d();
    }
    lsst::afw::geom::ellipses::EllipseCore::Transformer::TransformDerivativeMatrix _transformDTransform(
        lsst::afw::geom::LinearTransform const & t
    ) const {
        return self->transform(t).dTransform();
    }
}

%define %EllipseCore_PREINCLUDE(NAME)
%feature(notabstract) lsst::afw::geom::ellipses::NAME;
%implicitconv lsst::afw::geom::ellipses::NAME;
%shared_ptr(lsst::afw::geom::ellipses::NAME);
%rename(assign) lsst::afw::geom::ellipses::NAME::operator=;
%enddef

%define %EllipseCore_POSTINCLUDE(NAME)
%extend lsst::afw::geom::ellipses::NAME {
    %feature("shadow") _convolve %{
        def convolve(self, t):
            return $action(self, t)
    %}
    %feature("shadow") _getGridTransform %{
        def getGridTransform(self):
            return $action(self)
    %}

    PTR(lsst::afw::geom::ellipses::NAME) _convolve(
        lsst::afw::geom::ellipses::EllipseCore const & other
    ) {
        return boost::static_pointer_cast<lsst::afw::geom::ellipses::NAME>(
            self->convolve(other).copy()
        );
    }
    lsst::afw::geom::LinearTransform _getGridTransform() {
        return self->getGridTransform();
    }

    static PTR(lsst::afw::geom::ellipses::NAME) cast(
        PTR(lsst::afw::geom::ellipses::EllipseCore) const & p
    ) {
       return boost::dynamic_pointer_cast<lsst::afw::geom::ellipses::NAME>(p);
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
    def __reduce__(self):
        return (Axes, (self.getA(), self.getB(), self.getTheta()))
    }
}

%extend lsst::afw::geom::ellipses::Quadrupole {
    %pythoncode {
    def __reduce__(self):
        return (Quadrupole, (self.getIxx(), self.getIyy(), self.getIxy()))
    }
}

%include "Separable.i"

%feature("valuewrapper") lsst::afw::geom::ellipses::Ellipse;
%ignore lsst::afw::geom::ellipses::Ellipse::getCore;
%ignore lsst::afw::geom::ellipses::Ellipse::transform;
%ignore lsst::afw::geom::ellipses::Ellipse::convolve;
%ignore lsst::afw::geom::ellipses::Ellipse::getGridTransform;
%ignore lsst::afw::geom::ellipses::Ellipse::readParameters;
%ignore lsst::afw::geom::ellipses::Ellipse::writeParameters;

%rename(assign) lsst::afw::geom::ellipses::Ellipse::operator=;

%shared_ptr(lsst::afw::geom::ellipses::Ellipse);
%declareNumPyConverters(lsst::afw::geom::ellipses::Ellipse::ParameterVector);
%declareNumPyConverters(lsst::afw::geom::ellipses::Ellipse::Transformer::DerivativeMatrix);
%declareNumPyConverters(lsst::afw::geom::ellipses::Ellipse::Transformer::TransformDerivativeMatrix);

%addStreamRepr(lsst::afw::geom::ellipses::Ellipse);

%extend lsst::afw::geom::ellipses::Ellipse {
    %pythoncode %{
        def transform(self, transform, inPlace=False, doDerivatives=False):
            """Transform an Ellipse via an AffineTransform

            If inPlace, self will be transformed in-place and returned;
            otherwise, a new transformed ellipse with the same EllipseCore type
            will be returned.

            If doDerivatives, a tuple of (transformed, dEllipse, dTransform) will be
            returned, where dEllipse is the derivative of the transformed ellipse
            w.r.t. the input ellipse, and dTransform is the derivative of the
            transformed ellipse w.r.t. the AffineTransform elements.
            """
            if doDerivatives:
                dEllipse = _ellipsesLib.Ellipse__transformDEllipse(self, transform)
                dTransform = _ellipsesLib.Ellipse__transformDTransform(self, transform)
            if inPlace:
                r = self
            else:
                r = Ellipse(self)
            _ellipsesLib.Ellipse__transformInPlace(r, transform)
            if doDerivatives:
                return r, dEllipse, dTransform
            else:
                return r
    %}
    %feature("shadow") _transformInPlace %{%}
    %feature("shadow") _transformDEllipse %{%}
    %feature("shadow") _transformDTransform %{%}
    %feature("shadow") _convolve %{
        def convolve(self, t):
            return $action(self, t)
    %}
    %feature("shadow") _getGridTransform %{
        def getGridTransform(self):
            return $action(self)
    %}
    %feature("shadow") getCorePtr %{
        def getCore(self):
            return $action(self).cast()
    %}

    void _transformInPlace(lsst::afw::geom::AffineTransform const & t) {
        self->transform(t).inPlace();
    }
    lsst::afw::geom::ellipses::Ellipse::Transformer::DerivativeMatrix _transformDEllipse(
        lsst::afw::geom::AffineTransform const & t
    ) const {
        return self->transform(t).d();
    }
    lsst::afw::geom::ellipses::Ellipse::Transformer::TransformDerivativeMatrix _transformDTransform(
        lsst::afw::geom::AffineTransform const & t
    ) const {
        return self->transform(t).dTransform();
    }

    lsst::afw::geom::ellipses::Ellipse _convolve(lsst::afw::geom::ellipses::Ellipse const & other) const {
        return self->convolve(other);
    }
    lsst::afw::geom::AffineTransform _getGridTransform() {
        return self->getGridTransform();
    }
    %pythoncode %{
    def __reduce__(self):
        return (Ellipse, (self.getCore(), self.getCenter()))
    %}
}

%include "lsst/afw/geom/ellipses/Ellipse.h"
%include "lsst/afw/geom/ellipses/Parametric.h"
%include "lsst/afw/geom/ellipses.h" // just for Separable typedefs

%include "PixelRegion.i"
