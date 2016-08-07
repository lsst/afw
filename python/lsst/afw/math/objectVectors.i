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

/**
* Define vectors of objects (not POTS=plain old types) that are required for functions.

Call this last to avoid problems for overloaded constructors (or other functions, I assume).
If one of these lines appears before %importing the file declaring the type
and if the associated type is used as an argument in an overloaded constructor
that can alternatively be a vector of POTS, then:
- Calling the constructor with a python list containing a mix of the objects and types causes an abort
- SWIG will warn about a shadowed overloaded constructor (or function, presumably)
*/
%template(Function1FList) std::vector<std::shared_ptr<lsst::afw::math::Function1<float> > >;
%template(Function1DList) std::vector<std::shared_ptr<lsst::afw::math::Function1<double> > >;
%template(Function2FList) std::vector<std::shared_ptr<lsst::afw::math::Function2<float> > >;
%template(Function2DList) std::vector<std::shared_ptr<lsst::afw::math::Function2<double> > >;

%template(KernelList) std::vector<std::shared_ptr<lsst::afw::math::Kernel> >;
