// -*- lsst-C++ -*-

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
#include <memory>
#include "lsst/afw/math/SpatialCell.h"
%}

%shared_ptr(lsst::afw::math::CandidateVisitor)
%shared_ptr(lsst::afw::math::SpatialCellCandidate);
%shared_ptr(lsst::afw::math::SpatialCellImageCandidate);
%shared_ptr(lsst::afw::math::SpatialCell);

%rename(__incr__) lsst::afw::math::SpatialCellCandidateIterator::operator++;
%rename(__deref__) lsst::afw::math::SpatialCellCandidateIterator::operator*;
%rename(__eq__) lsst::afw::math::SpatialCellCandidateIterator::operator==;
%rename(__ne__) lsst::afw::math::SpatialCellCandidateIterator::operator!=;

%include "lsst/afw/math/SpatialCell.h"

%template(SpatialCellCandidateList) std::vector<std::shared_ptr<lsst::afw::math::SpatialCellCandidate> >;
%template(SpatialCellList) std::vector<std::shared_ptr<lsst::afw::math::SpatialCell> >;

%extend lsst::afw::math::SpatialCell {
    %pythoncode %{
        def __getitem__(self, ind):
            return [c for c in self.begin()][ind]

        def __iter__(self):
            return self.begin().__iter__()
    %}
}

%extend lsst::afw::math::SpatialCellCandidateIterator {
    %pythoncode %{
    def __iter__(self):
        while True:
            try:
                yield self.__deref__()
            except:
                raise StopIteration

            self.__incr__()
    %}
}

%castShared(lsst::afw::math::SpatialCellImageCandidate, lsst::afw::math::SpatialCellCandidate)