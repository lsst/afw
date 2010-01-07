// -*- lsst-c++ -*-

%{
#include "lsst/afw/detection/Source.h"
#include "lsst/afw/detection/SourceMatch.h"
%}

%include "lsst/afw/detection/SourceMatch.h"

%extend lsst::afw::detection::SourceMatch {
    %pythoncode {
    def __repr__(self):
        s1 = repr(self.first)
        s2 = repr(self.second)
        return "SourceMatch(%s,\n            %s,\n            %g)" % (
               s1, s2, self.distance)

    def __str__(self):
        s1, s2 = self.first, self.second
        return "((%d, (%g,%g), (%g,%g))\n (%d, (%g,%g), (%g,%g))\n %g)" % (
               s1.getId(), s1.getRa(), s1.getDec(), s1.getX(), s1.getY(),
               s2.getId(), s2.getRa(), s2.getDec(), s2.getX(), s2.getY(),
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

    }
}

%template(SourceMatchSet) std::vector<lsst::afw::detection::SourceMatch>;

