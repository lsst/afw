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
#include "lsst/afw/geom/TransformRegistry.h"
%}

// unfortunately the following renames don't work, so use %extend
// %rename(__getitem__) lsst::afw::geom::TransformRegistry::operator[];
// %rename(__contains__) lsst::afw::geom::TransformRegistry::contains();
// %rename(__len__) lsst::afw::geom::TransformRegistry::size();
%ignore lsst::afw::geom::TransformRegistry::operator[]; // this avoids a SWIG warning
%ignore lsst::afw::geom::TransformRegistry::contains(); // this does nothing, alas
%ignore lsst::afw::geom::TransformRegistry::size(); // this does nothing, alas

%include "lsst/afw/geom/TransformRegistry.h"

%extend lsst::afw::geom::TransformRegistry {
    // rename operator[]
    CONST_PTR(lsst::afw::geom::XYTransform) __getitem__(
            lsst::afw::geom::TransformRegistry::_CoordSysType const &coordSys
        ) const { return (*($self))[coordSys]; }

    // rename contains
    bool __contains__(
        lsst::afw::geom::TransformRegistry::_CoordSysType const &coordSys
    ) const { return $self->contains(coordSys); }

    // rename size
    size_t __len__() const { return $self->size(); }

    %pythoncode {
        def __iter__(self):
            for coordSys in self.getCoordSysList():
                yield self[coordSys]
    }
}
