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
#include "lsst/afw/math/minimize.h"
%}

%import "Minuit2/GenericFunction.h"
%include "Minuit2/FCNBase.h"
%include "lsst/afw/math/Function.h"
%include "lsst/afw/math/FunctionLibrary.h"
%include "lsst/afw/math/minimize.h"

%template(pairDD) std::pair<double,double>;
%template(vectorPairDD) std::vector<std::pair<double,double> >;

%template(minimize)             lsst::afw::math::minimize<float>;
%template(minimize)             lsst::afw::math::minimize<double>;
