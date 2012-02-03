// -*- lsst-c++ -*-
#if 0
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
#include "lsst/afw/detection/Source.h"
#include "lsst/afw/detection/SourceMatch.h"
%}

%shared_ptr(lsst::afw::detection::PersistableSourceMatchVector);

%include "lsst/afw/detection/SourceMatch.h"

%template(SourceMatchVector) std::vector<lsst::afw::detection::SourceMatch>;

%shared_ptr(lsst::afw::detection::PersistableSourceMatchVector);

%extend lsst::afw::detection::SourceMatch {
    %pythoncode {
    def __repr__(self):
        return "SourceMatch(%s,\n            %s,\n            %g)" % \
        (repr(self.first), repr(self.second), self.distance)

    def __str__(self):
        s1, s2 = self.first, self.second
        return "((id %d, RA,Dec (%g,%g) deg; X,Y (%g,%g))\n (id %d, RA,Dec (%g,%g) deg; X,Y (%g,%g))\n dist %g [pix or radians])" % (
            s1.getId(), s1.getRa().asDegrees(), s1.getDec().asDegrees(), s1.getXAstrom(), s1.getYAstrom(),
            s2.getId(), s2.getRa().asDegrees(), s2.getDec().asDegrees(), s2.getXAstrom(), s2.getYAstrom(),
            self.distance)

    def __getitem__(self, i):
        """Treat a SourceMatch as a tuple of length 3:
        (first, second, distance)"""
        if i > 2 or i < -3:
            raise IndexError(i)
        if i < 0:
            i += 3
        if i == 0:
            return self.first
        elif i == 1:
            return self.second
        else:
            return self.distance

    def __setitem__(self, i, val):
        """Treat a SourceMatch as a tuple of length 3:
        (first, second, distance)"""
        if i > 2 or i < -3:
            raise IndexError(i)
        if i < 0:
            i += 3
        if i == 0:
            self.first = val
        elif i == 1:
            self.second = val
        else:
            self.distance = val

    def __len__(self):
        return 3

    def clone(self):
        return self.__class__(self.first, self.second, self.distance)

    def __getstate__(self):
        return self.first, self.second, self.distance

    def __setstate__(self, state):
        self.__init__(*state)
    }
}

%lsst_persistable(lsst::afw::detection::PersistableSourceMatchVector);
#endif
