// -*- lsst-c++ -*-

/*
 * LSST Data Management System
 * See COPYRIGHT file at the top of the source tree.
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the LSST License Statement and
 * the GNU General Public License along with this program. If not,
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */


%{
#include "lsst/afw/geom/SpherePoint.h"
%}

%include "lsst/afw/geom/Angle.h"
%include "lsst/afw/geom/Point.h"

%useValueEquality(lsst::afw::geom::SpherePoint);

%include "lsst/afw/geom/SpherePoint.h"

// add __str__ and __repr__ methods
%extend lsst::afw::geom::SpherePoint {
    std::string __str__() const {
        std::ostringstream os;
        os << std::fixed << (*self);
        return os.str();
    }
    %pythoncode %{
def __repr__(self):
    argList = ["%r*afwGeom.degrees" % (pos.asDegrees(),) for pos in self]
    return "SpherePoint(%s)" % (", ".join(argList))
    %}
}

// Allow negative indices in Python but not C++
%extend lsst::afw::geom::SpherePoint {
    lsst::afw::geom::Angle _getItemImpl(size_t index) const {
        return (*self)[index];
    }
    %pythoncode %{
def __getitem__(self, index):
    if index < -len(self) or index > 1:
        raise IndexError("Invalid index: %d" % index)
    try:
        if index >= 0:
            return self._getItemImpl(index)
        else:
            return self._getItemImpl(len(self) + index)
    except lsst.pex.exceptions.OutOfRangeError as e:
        raise_from(IndexError(), e)
    %}
}

// Add __iter__ to allow  'ra,dec = point' statement in Python
%extend lsst::afw::geom::SpherePoint {
    %pythoncode %{
def __iter__(self):
    for i in (0, 1):
        yield self[i]
def __len__(self):
    return 2
    %}
}

// Add __reduce__
%extend lsst::afw::geom::SpherePoint {
    %pythoncode %{
def __reduce__(self):
    return (SpherePoint, (self.getLongitude(), self.getLatitude()))
    %}
}
