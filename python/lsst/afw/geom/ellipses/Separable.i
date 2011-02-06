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

%{
#include "lsst/afw/geom/ellipses/radii.h"
#include "lsst/afw/geom/ellipses/LogShear.h"
#include "lsst/afw/geom/ellipses/Distortion.h"
#include "lsst/afw/geom/ellipses/Separable.h"
%}

%declareNumPyConverters(lsst::afw::geom::ellipses::EllipticityBase::Jacobian);


%rename(assign) lsst::afw::geom::ellipses::GeometricRadius::operator=;
%rename(assign) lsst::afw::geom::ellipses::ArithmeticRadius::operator=;
%rename(assign) lsst::afw::geom::ellipses::LogGeometricRadius::operator=;

%rename(assign) lsst::afw::geom::ellipses::Distortion::operator=;
%rename(assign) lsst::afw::geom::ellipses::LogShear::operator=;

%include "lsst/afw/geom/ellipses/radii.h"
%include "lsst/afw/geom/ellipses/EllipticityBase.h"
%include "lsst/afw/geom/ellipses/Distortion.h"
%include "lsst/afw/geom/ellipses/LogShear.h"

%ignore lsst::afw::geom::ellipses::Separable::writeParameters;
%ignore lsst::afw::geom::ellipses::Separable::readParameters;
%rename(assign) lsst::afw::geom::ellipses::Separable::operator=;

%define %Separable_PREINCLUDE(ELLIPTICITY, RADIUS)
SWIG_SHARED_PTR_DERIVED(
    Separable ## ELLIPTICITY ## RADIUS ## Ptr,
    lsst::afw::geom::ellipses::BaseCore,
    lsst::afw::geom::ellipses::Separable<
        lsst::afw::geom::ellipses::ELLIPTICITY, 
        lsst::afw::geom::ellipses::RADIUS
    > 
);
%rename(assign) lsst::afw::geom::ellipses::Separable<
        lsst::afw::geom::ellipses::ELLIPTICITY,
        lsst::afw::geom::ellipses::RADIUS
    >::operator=;
%enddef


%define %Separable_POSTINCLUDE(ELLIPTICITY, RADIUS)
%template(Separable ## ELLIPTICITY ## RADIUS) 
    lsst::afw::geom::ellipses::Separable<
        lsst::afw::geom::ellipses::ELLIPTICITY, 
        lsst::afw::geom::ellipses::RADIUS
    >;
%enddef


%Separable_PREINCLUDE(Distortion, GeometricRadius);
%Separable_PREINCLUDE(Distortion, ArithmeticRadius);
%Separable_PREINCLUDE(Distortion, LogGeometricRadius);
%Separable_PREINCLUDE(Distortion, LogArithmeticRadius);

%Separable_PREINCLUDE(LogShear, GeometricRadius);
%Separable_PREINCLUDE(LogShear, ArithmeticRadius);
%Separable_PREINCLUDE(LogShear, LogGeometricRadius);
%Separable_PREINCLUDE(LogShear, LogArithmeticRadius);

%include "lsst/afw/geom/ellipses/Separable.h"

%extend lsst::afw::geom::ellipses::Separable {
    lsst::afw::geom::ellipses::Separable<Ellipticity_, Radius_>::Ptr _transform(
            lsst::afw::geom::LinearTransform const & t
    ) {
        return boost::static_pointer_cast<lsst::afw::geom::ellipses::Separable<Ellipticity_, Radius_> >(
            self->transform(t).copy()
        );
    }
    void _transformInPlace(lsst::afw::geom::LinearTransform const & t) {
       self->transform(t).inPlace();
    }
    lsst::afw::geom::ellipses::Separable<Ellipticity_, Radius_>::Ptr _convolve(
            lsst::afw::geom::ellipses::BaseCore const & other
    ) {
        return boost::static_pointer_cast<lsst::afw::geom::ellipses::Separable<Ellipticity_, Radius_> >(
            self->convolve(other).copy()
        );
    }
    static lsst::afw::geom::ellipses::Separable<Ellipticity_, Radius_>::Ptr cast(
        lsst::afw::geom::ellipses::BaseCore::Ptr const & p
    ) {
        return boost::dynamic_pointer_cast<lsst::afw::geom::ellipses::Separable<Ellipticity_, Radius_> >(p);
    }
    %pythoncode {
    def transform(self, t): return self._transform(t)
    def transformInPlace(self, t): self._transformInPlace(t)
    def convolve(self, t): return self._convolve(t)
    }
}

%Separable_POSTINCLUDE(Distortion, GeometricRadius);
%Separable_POSTINCLUDE(Distortion, ArithmeticRadius);
%Separable_POSTINCLUDE(Distortion, LogGeometricRadius);
%Separable_POSTINCLUDE(Distortion, LogArithmeticRadius);

%Separable_POSTINCLUDE(LogShear, GeometricRadius);
%Separable_POSTINCLUDE(LogShear, ArithmeticRadius);
%Separable_POSTINCLUDE(LogShear, LogGeometricRadius);
%Separable_POSTINCLUDE(LogShear, LogArithmeticRadius);
