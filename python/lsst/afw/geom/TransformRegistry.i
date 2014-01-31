/* 
 * LSST Data Management System
 * Copyright 2014 LSST Corporation.
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
#include "lsst/afw/geom/TransformRegistry.h"
%}

// unfortunately the following rename silently fails with SWIG 2.0.4 (though it stops the SWIG warning)
// so use %ignore and %extend instead
// %rename(__getitem__) lsst::afw::geom::TransformRegistry::operator[];
%ignore lsst::afw::geom::TransformRegistry::operator[];

// note: the following fail with SWIG 2.0.4 if the method names have () after them; why?
%rename(__contains__) lsst::afw::geom::TransformRegistry::contains;
%rename(__len__) lsst::afw::geom::TransformRegistry::size;
%ignore lsst::afw::geom::TransformRegistry::begin;
%ignore lsst::afw::geom::TransformRegistry::end;

%include "lsst/afw/geom/TransformRegistry.h"

%extend lsst::afw::geom::TransformRegistry {
    // rename operator[]
    CONST_PTR(lsst::afw::geom::XYTransform) __getitem__(
            lsst::afw::geom::TransformRegistry::_CoordSysType const &coordSys
        ) const { return (*$self)[coordSys]; }

    %pythoncode {
        def __iter__(self):
            return iter(self.getCoordSysList())
    }
}
