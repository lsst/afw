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
#include "lsst/afw/geom/Box.h"
%}

%rename(set) lsst::afw::geom::BoxI::operator=;
%rename(__eq__) lsst::afw::geom::BoxI::operator==;
%rename(__ne__) lsst::afw::geom::BoxI::operator!=;
%copyctor lsst::afw::geom::BoxI;
%rename(set) lsst::afw::geom::BoxD::operator=;
%rename(__eq__) lsst::afw::geom::BoxD::operator==;
%rename(__ne__) lsst::afw::geom::BoxD::operator!=;
%copyctor lsst::afw::geom::BoxD;

%include "lsst/afw/geom/Box.h"

%extend lsst::afw::geom::BoxI {             
    %pythoncode {
    def __repr__(self):
        return "BoxI(%r, %r)" % (self.getMin(), self.getDimensions())
             
    def __str__(self):
        return "BoxI(%s, %s)" % (self.getMin(), self.getMax())
    }
}

%extend lsst::afw::geom::BoxD {             
    %pythoncode {
    def __repr__(self):
        return "BoxD(%r, %r)" % (self.getMin(), self.getDimensions())
             
    def __str__(self):
        return "BoxD(%s, %s)" % (self.getMin(), self.getMax())
    }
}
