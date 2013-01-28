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

%{
#include "lsst/afw/geom/ellipses/BaseCore.h"
#include "lsst/afw/geom/ellipses/Transformer.h"
#include "lsst/afw/geom/ellipses/Convolution.h"
#include "lsst/afw/geom/ellipses/GridTransform.h"
%}

%import "lsst/afw/geom/geom_fwd.i"
%import "lsst/afw/geom/LinearTransform.i"

%ignore lsst::afw::geom::ellipses::BaseCore::transform;
%ignore lsst::afw::geom::ellipses::BaseCore::convolve;
%ignore lsst::afw::geom::ellipses::BaseCore::getGridTransform;
%ignore lsst::afw::geom::ellipses::BaseCore::readParameters;
%ignore lsst::afw::geom::ellipses::BaseCore::writeParameters;
%ignore lsst::afw::geom::ellipses::BaseCore::as;
%rename(assign) lsst::afw::geom::ellipses::BaseCore::operator=;

%shared_ptr(lsst::afw::geom::ellipses::BaseCore);

%include "lsst/afw/geom/ellipses/BaseCore.h"

%define %EllipseCore_PREINCLUDE(NAME)
%feature(notabstract) lsst::afw::geom::ellipses::NAME;
%implicitconv lsst::afw::geom::ellipses::NAME;
%shared_ptr(lsst::afw::geom::ellipses::NAME);
%ignore lsst::afw::geom::ellipses::NAME::writeParameters;
%ignore lsst::afw::geom::ellipses::NAME::readParameters;
%rename(assign) lsst::afw::geom::ellipses::NAME::operator=;
%enddef

%define %EllipseCore_POSTINCLUDE(NAME)
%extend lsst::afw::geom::ellipses::NAME {
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

    lsst::afw::geom::ellipses::NAME::Ptr _transform(
        lsst::afw::geom::LinearTransform const & t
    ) {
        return boost::static_pointer_cast<lsst::afw::geom::ellipses::NAME>(
            self->transform(t).copy()
        );
    }
    void _transformInPlace(lsst::afw::geom::LinearTransform const & t) {
       self->transform(t).inPlace();
    }
    lsst::afw::geom::ellipses::NAME::Ptr _convolve(
        lsst::afw::geom::ellipses::BaseCore const & other
    ) {
        return boost::static_pointer_cast<lsst::afw::geom::ellipses::NAME>(
            self->convolve(other).copy()
        );
    }
    lsst::afw::geom::LinearTransform _getGridTransform() {
        return self->getGridTransform();
    }

    static lsst::afw::geom::ellipses::NAME::Ptr cast(
        lsst::afw::geom::ellipses::BaseCore::Ptr const & p
    ) {
       return boost::dynamic_pointer_cast<lsst::afw::geom::ellipses::NAME>(p);
    }
}
%enddef
