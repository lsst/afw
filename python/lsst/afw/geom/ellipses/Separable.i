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

%{
#include "lsst/afw/geom/ellipses/ConformalShear.h"
#include "lsst/afw/geom/ellipses/ReducedShear.h"
#include "lsst/afw/geom/ellipses/Distortion.h"
#include "lsst/afw/geom/ellipses/Separable.h"
%}

%include "std_complex.i"

%declareNumPyConverters(lsst::afw::geom::ellipses::EllipticityBase::Jacobian);

%rename(assign) lsst::afw::geom::ellipses::Distortion::operator=;
%rename(assign) lsst::afw::geom::ellipses::ConformalShear::operator=;
%rename(assign) lsst::afw::geom::ellipses::ReducedShear::operator=;
%ignore lsst::afw::geom::ellipses::detail::EllipticityBase::getComplex;

%define %Ellipticity_POSTINCLUDE(ELLIPTICITY)
%extend lsst::afw::geom::ellipses::ELLIPTICITY {
    %feature("shadow") _getComplex %{
        def getComplex(self):
            return $action(self)
    %}
    std::complex<double> _getComplex() const {
        return self->getComplex();
    }
    void setComplex(std::complex<double> other) {
        self->getComplex() = other;
    }
    %pythoncode {
    def __str__(self):
        return "(%g, %g)" % (self.getE1(), self.getE2())
    def __repr__(self):
        return "%s(%g, %g)" % (self.getName(), self.getE1(), self.getE2())
    }
}
%enddef

%include "lsst/afw/geom/ellipses/EllipticityBase.h"
%include "lsst/afw/geom/ellipses/Distortion.h"
%include "lsst/afw/geom/ellipses/ConformalShear.h"
%include "lsst/afw/geom/ellipses/ReducedShear.h"

%Ellipticity_POSTINCLUDE(Distortion);
%Ellipticity_POSTINCLUDE(ConformalShear);
%Ellipticity_POSTINCLUDE(ReducedShear);

%ignore lsst::afw::geom::ellipses::Separable::writeParameters;
%ignore lsst::afw::geom::ellipses::Separable::readParameters;
%rename(assign) lsst::afw::geom::ellipses::Separable::operator=;

%define %Separable_PREINCLUDE(ELLIPTICITY)
%feature(notabstract) lsst::afw::geom::ellipses::NAME;
%implicitconv lsst::afw::geom::ellipses::NAME;
%shared_ptr(lsst::afw::geom::ellipses::Separable<lsst::afw::geom::ellipses::ELLIPTICITY>);
%rename(assign) lsst::afw::geom::ellipses::Separable<lsst::afw::geom::ellipses::ELLIPTICITY>::operator=;
%enddef


%define %Separable_POSTINCLUDE(ELLIPTICITY)
%template(ELLIPTICITY ## EllipseCore)
    lsst::afw::geom::ellipses::Separable<lsst::afw::geom::ellipses::ELLIPTICITY>;
%enddef


%Separable_PREINCLUDE(Distortion);
%Separable_PREINCLUDE(ConformalShear);
%Separable_PREINCLUDE(ReducedShear);

%include "lsst/afw/geom/ellipses/Separable.h"

%extend lsst::afw::geom::ellipses::Separable {
    static PTR(lsst::afw::geom::ellipses::Separable<Ellipticity_>) cast(
        PTR(lsst::afw::geom::ellipses::EllipseCore) const & p
    ) {
        return boost::dynamic_pointer_cast<lsst::afw::geom::ellipses::Separable<Ellipticity_> >(p);
    }
}

%Separable_POSTINCLUDE(Distortion);
%Separable_POSTINCLUDE(ConformalShear);
%Separable_POSTINCLUDE(ReducedShear);
