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

/*
 * Wrappers for the Match object and matching functions, including instantiation of SimpleMatch, SourceMatch,
 * and ReferenceMatch.
 */

%include "lsst/afw/table/Source.i"

%{
#include "lsst/afw/table/Match.h"
%}

%include "lsst/afw/table/Match.h"

%extend lsst::afw::table::Match {
    %pythoncode {
    def __repr__(self):
        return "Match(%s,\n            %s,\n            %g)" % \
        (repr(self.first), repr(self.second), self.distance)

    def __str__(self):
        s1, s2 = self.first, self.second
        return "((id %d, RA,Dec (%g,%g) deg; X,Y (%g,%g))\n (id %d, RA,Dec (%g,%g) deg; X,Y (%g,%g))\n dist %g [pix or radians])" % (
            s1.getId(), s1.getRa().asDegrees(), s1.getDec().asDegrees(), s1.getX(), s1.getY(),
            s2.getId(), s2.getRa().asDegrees(), s2.getDec().asDegrees(), s2.getX(), s2.getY(),
            self.distance)

    def __getitem__(self, i):
        """Treat a Match as a tuple of length 3:
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
        """Treat a Match as a tuple of length 3:
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

%define %declareMatch(NAME, R1, R2)

%template(NAME##Match) lsst::afw::table::Match<lsst::afw::table::R1##Record,lsst::afw::table::R2##Record>;

%template(NAME##MatchVector)
    std::vector< lsst::afw::table::Match<lsst::afw::table::R1##Record,lsst::afw::table::R2##Record> >;

%extend std::vector< lsst::afw::table::Match<lsst::afw::table::R1##Record,lsst::afw::table::R2##Record> > {
    %pythoncode {
        def __getstate__(self):
            """Pickler"""
            if not self:
                return ()
            firstSchema = self[0].first.schema
            secondSchema = self[0].second.schema
            firstTable = self[0].first.table
            secondTable = self[0].second.table
            first = R1##Catalog(firstSchema)
            second = R2##Catalog(secondSchema)
            distance = []
            for match in self:
                first.append(match.first)
                second.append(match.second)
                distance.append(match.distance)

            from lsst.pex.exceptions import LsstCppException, NotFoundError
            def getSlot(table, name):
                """Return key for a slot, specified by name, or None"""
                getter = getattr(table, "get" + name + "Key", None)
                if getter is None:
                    return None
                try:
                    return getter()
                except LsstCppException as e:
                    if isinstance(e.message, NotFoundError):
                        return None
                    raise

            def copySlots(tableFrom, tableTo):
                """Copy slots from one table to another

                Slots are identified from the method names ("define<name>")
                and assumed to carry measurement, error and flag parts.
                """
                import re
                getKey = lambda x: getattr(tableFrom, x)() if hasattr(tableFrom, x) else None
                for method in dir(tableTo):
                    m = re.search(r"define(?P<name>.+)", method)
                    if not m:
                        continue
                    name = m.group("name")
                    meas = getSlot(tableFrom, name)
                    err = getSlot(tableFrom, name + "Err")
                    flag = getSlot(tableFrom, name + "Flag")
                    setter = getattr(tableTo, "define" + name, None)
                    if setter is not None:
                        setter(meas, err, flag)

            copySlots(firstTable, first.table)
            copySlots(secondTable, second.table)
            return (first, second, distance)

        def __setstate__(self, state):
            """Unpickler"""
            first, second, distance = state
            self.__init__([NAME##Match(f,s,d) for f,s,d in zip(first, second, distance)])
    }
}

// swig can't parse the template declarations for these because of the nested names in the
// return values, so we repeat them here and pretend they aren't templates, using typedefs
// that swig can understand.

namespace lsst { namespace afw { namespace table {

NAME##MatchVector matchRaDec(
    R1##Catalog const & cat1,
    R2##Catalog const & cat2,
    Angle radius, bool closest=true
);

NAME##MatchVector unpackMatches(
    BaseCatalog const & matches,
    R1##Catalog const & cat1,
    R2##Catalog const & cat2
);

}}} // namespace lsst::afw::table

%enddef

%declareMatch(Simple, Simple, Simple)
%declareMatch(Reference, Simple, Source)
%declareMatch(Source, Source, Source)

namespace lsst { namespace afw { namespace table {

SimpleMatchVector matchRaDec(
    SimpleCatalog const & cat,
    Angle radius, bool symmetric=true
);

SourceMatchVector matchRaDec(
    SourceCatalog const & cat,
    Angle radius, bool symmetric=true
);

SourceMatchVector matchXy(
    SourceCatalog const & cat1,
    SourceCatalog const & cat2,
    double radius, bool closest=true
);

SourceMatchVector matchXy(
    SourceCatalog const & cat,
    double radius, bool symmetric=true
);

}}} // namespace lsst::afw::table

// swig can't disambiguate between the different packMatches overloads (which is actually
// understandable, because they'd all match a Python list), so instead we provide
// a pure-Python implementation that works on any sequence.
%pythoncode %{
    def packMatches(matches):
        schema = Schema()
        outKey1 = schema.addField("first", type="L", doc="ID for first source record in match.")
        outKey2 = schema.addField("second", type="L", doc="ID for second source record in match.")
        keyD = schema.addField("distance", type=float, doc="Distance between matches sources.")
        result = BaseCatalog(schema)
        result.table.preallocate(len(matches))
        for match in matches:
            record = result.addNew()
            record.set(outKey1, match.first.getId())
            record.set(outKey2, match.second.getId())
            record.set(keyD, match.distance)
        return result
%}
