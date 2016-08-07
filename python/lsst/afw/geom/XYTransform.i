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

/************************************************************************************************************/

%{
#include "lsst/afw/geom/Functor.h"
#include "lsst/afw/geom/XYTransform.h"
#include "lsst/afw/geom/SeparableXYTransform.h"
%}

%shared_ptr(lsst::afw::geom::Functor);
%shared_ptr(lsst::afw::geom::LinearFunctor);

%shared_ptr(lsst::afw::geom::XYTransform);
%shared_ptr(lsst::afw::geom::IdentityXYTransform);
%shared_ptr(lsst::afw::geom::InvertedXYTransform);
%shared_ptr(lsst::afw::geom::AffineXYTransform);
%shared_ptr(lsst::afw::geom::RadialXYTransform);
%shared_ptr(lsst::afw::geom::MultiXYTransform);
%shared_ptr(lsst::afw::geom::SeparableXYTransform);

%template(XYTransformVector) std::vector<CONST_PTR(lsst::afw::geom::XYTransform)>;

%include "lsst/afw/geom/Functor.h"
%include "lsst/afw/geom/XYTransform.h"
%include "lsst/afw/geom/SeparableXYTransform.h"
