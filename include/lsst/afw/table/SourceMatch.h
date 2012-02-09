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
 
#ifndef LSST_AFW_TABLE_SOURCEMATCH_H
#define LSST_AFW_TABLE_SOURCEMATCH_H

#include <vector>

#include "lsst/afw/table/Source.h"
#include "lsst/afw/geom/Angle.h"

namespace lsst { namespace afw { namespace table {

struct SourceMatch {
    PTR(SourceRecord) first;
    PTR(SourceRecord) second;
    // match distance, in RADIANS or PIXELS depending on the type of match requested.
    double distance;

    SourceMatch() : first(), second(), distance(0.0) {}
    SourceMatch(PTR(SourceRecord) const & s1, PTR(SourceRecord) const & s2, double dist)
        : first(s1), second(s2), distance(dist) {}
    ~SourceMatch() {}
};

typedef std::vector<SourceMatch> SourceMatchVector;

SourceMatchVector matchRaDec(SourceVector const &set1, SourceVector const &set2,
                             geom::Angle radius, bool closest=true);
SourceMatchVector matchRaDec(SourceVector const &set, geom::Angle radius,
                             bool symmetric = true);
SourceMatchVector matchXy(SourceVector const &set1, SourceVector const &set2,
                          double radius, bool closest=true);
SourceMatchVector matchXy(SourceVector const &set, double radius, bool symmetric = true);

/**
 *  @brief Return a BaseTable vector representation of a SourceMatchVector that can be used
 *         to persist it.
 *
 *  The schema of returned object has "first" (RecordId), "second" (RecordID), and "distance"
 *  (double) fields.
 *
 *  @param[in]  matches     A std::vector of SourceMatch objects to convert to table form.
 */
BaseVector makeSourceMatchTable(SourceMatchVector const & matches);

/**
 *  @brief Reconstruct a SourceMatchVector from a BaseVector representation of the matches
 *         and a pair of SourceVectors that hold the sources themselves.
 *
 *  @note The SourceVector arguments must be sorted in ascending ID order on input; this will
 *        allow us to use binary search algorithms to find the sources referred to by the match
 *        table.
 *
 *  If a source ID cannot be found in the given tables, that source pointer will be set to null
 *  in the returned match vector.
 *
 *  @param[in]  matches     A BaseTable vector representation, as produced by makeSourceMatchTable.
 *  @param[in]  first       A SourceVector containing the sources used on the 'first' side of the match,
 *                          sorted by ascending ID.
 *  @param[in]  second      A SourceVector containing the sources used on the 'second' side of the match,
 *                          sorted by ascending ID.  May be the same as first.
 */
SourceMatchVector makeSourceMatchVector(
    BaseVector const & matches, 
    SourceVector const & first,
    SourceVector const & second
);

}}} // namespace lsst::afw::table

#endif // #ifndef LSST_AFW_TABLE_SOURCEMATCH_H
