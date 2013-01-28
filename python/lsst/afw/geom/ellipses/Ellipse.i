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

%module(package="lsst.afw.geom.ellipses") ellipsesLib

%include "lsst/afw/geom/ellipses/BaseCore.i"

%{
#include "lsst/afw/geom/ellipses/Ellipse.h"
#include "lsst/afw/geom/ellipses/Transformer.h"
#include "lsst/afw/geom/ellipses/Convolution.h"
#include "lsst/afw/geom/ellipses/GridTransform.h"
%}

%import "lsst/afw/geom/geom_fwd.i"
%import "lsst/afw/geom/AffineTransform.i"

%pythondynamic lsst::afw::geom::ellipses::Ellipse;

%feature("valuewrapper") lsst::afw::geom::ellipses::Ellipse;
%ignore lsst::afw::geom::ellipses::Ellipse::getCore;
%ignore lsst::afw::geom::ellipses::Ellipse::transform;
%ignore lsst::afw::geom::ellipses::Ellipse::convolve;
%ignore lsst::afw::geom::ellipses::Ellipse::getGridTransform;
%ignore lsst::afw::geom::ellipses::Ellipse::readParameters;
%ignore lsst::afw::geom::ellipses::Ellipse::writeParameters;

%rename(assign) lsst::afw::geom::ellipses::Ellipse::operator=;

%shared_ptr(lsst::afw::geom::ellipses::Ellipse);

%extend lsst::afw::geom::ellipses::Ellipse {
    %feature("shadow") _transform %{
        def transform(self, t):
            return $action(self, t)
    %}
    %feature("shadow") _transformInPlace %{
        def transformInPlace(self, t):
            $action(self, t)
    %}
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


    lsst::afw::geom::ellipses::Ellipse _transform(lsst::afw::geom::AffineTransform const & t) {
        return self->transform(t);
    }
    void _transformInPlace(lsst::afw::geom::AffineTransform const & t) {
        self->transform(t).inPlace();
    }
    lsst::afw::geom::ellipses::Ellipse _convolve(lsst::afw::geom::ellipses::Ellipse const & other) {
        return self->convolve(other);
    }
    lsst::afw::geom::AffineTransform _getGridTransform() {
        return self->getGridTransform();
    }
    %pythoncode {
    def __repr__(self):
        return "Ellipse(%r, %r)" % (self.getCore(), self.getCenter())
    def __reduce__(self):
        return (Ellipse, (self.getCore(), self.getCenter()))
    def __str__(self):
        return "(%s, %s)" % (self.getCore(), self.getCenter())
    }
}

%include "lsst/afw/geom/ellipses/Ellipse.h"
