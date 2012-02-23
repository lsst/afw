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
5B5B5B5B * This program is distributed in the hope that it will be useful,
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
#include "lsst/afw/table/Match.h"

namespace lsst { namespace afw { namespace table {

typedef Match<SourceRecord,SourceRecord> SourceMatch;

typedef std::vector<SourceMatch> SourceMatchVector;

/** 
 * Compute all tuples (s1,s2,d) where s1 belings to @a set1, s2 belongs to @a set2 and
 * d, the distance between s1 and s2, is at most @a radius. If set1 and
 * set2 are identical, then this call is equivalent to @c matchRaDec(set1,radius,true).
 * The match is performed in ra, dec space.
 *
 * @param[in] set1     first set of sources
 * @param[in] set2     second set of sources
 * @param[in] radius   match radius
 * @param[in] closest  if true then just return the closest match
 */
inline SourceMatchVector matchRaDec(
    SourceCatalog const &set1, SourceCatalog const &set2,
    geom::Angle radius, bool closest=true
) {
    return SourceMatch::static_vector_cast(
        matchRaDec(set1, SourceTable::getCoordKey(), set2, SourceTable::getCoordKey(), radius, closest)
    );             
}

/** 
 * Compute all tuples (s1,s2,d) where s1 != s2, s1 and s2 both belong to @a set,
 * and d, the distance between s1 and s2, is at most @a radius. The
 * match is performed in ra, dec space.
 *
 * @param[in] set          the set of sources to self-match
 * @param[in] radius       match radius
 * @param[in] symmetric    if set to @c true symmetric matches are produced: i.e.
 *                         if (s1, s2, d) is reported, then so is (s2, s1, d).
 */
inline SourceMatchVector matchRaDec(
    SourceCatalog const &set, geom::Angle radius,
    bool symmetric = true
) {
    return SourceMatch::static_vector_cast(
        matchRaDec(set, SourceTable::getCoordKey(), radius, symmetric)
    );
}

/**
 * Compute all tuples (s1,s2,d) where s1 belings to @a set1, s2 belongs to @a set2 and
 * d, the distance between s1 and s2, in pixels, is at most @a radius. If set1 and
 * set2 are identical, then this call is equivalent to @c matchXy(set1,radius,true).
 * The match is performed in pixel space (2d cartesian).
 *
 * @param[in] set1     first set of sources
 * @param[in] set2     second set of sources
 * @param[in] radius   match radius (pixels)
 * @param[in] closest  if true then just return the closest match
 */
inline SourceMatchVector matchXy(
    SourceCatalog const &set1, SourceCatalog const &set2,
    double radius, bool closest=true
) {
    return SourceMatch::static_vector_cast(
        matchXy(
            set1, set1.getTable()->getCentroidKey(),
            set2, set2.getTable()->getCentroidKey(),
            radius, closest
        )
    );
}

/**
 * Compute all tuples (s1,s2,d) where s1 != s2, s1 and s2 both belong to @a set,
 * and d, the distance between s1 and s2, in pixels, is at most @a radius. The
 * match is performed in pixel space (2d cartesian).
 *
 * @param[in] set          the set of sources to self-match
 * @param[in] radius       match radius (pixels)
 * @param[in] symmetric    if set to @c true symmetric matches are produced: i.e.
 *                         if (s1, s2, d) is reported, then so is (s2, s1, d).
 */
inline SourceMatchVector matchXy(SourceCatalog const &set, double radius, bool symmetric = true) {
    return SourceMatch::static_vector_cast(
        matchXy(set, set.getTable()->getCentroidKey(), radius, symmetric)
    );
}

/**
 *  @brief Return a table representation of an SourceMatchVector that can be used to persist it.
 *
 *  The schema of the returned object has "first" (RecordId), "second" (RecordID), and "distance"
 *  (double) fields.
 *
 *  @param[in]  matches     A std::vector of Match objects to convert to table form.
 */
inline BaseCatalog packMatches(SourceMatchVector const & matches) {
    return packMatches(
        BaseMatchVector(matches.begin(), matches.end()),
        SourceTable::getIdKey(),
        SourceTable::getIdKey()
    );
}

/**
 *  @brief Reconstruct a SourceMatchVector from a BaseCatalog representation of the matches
 *         and a pair of SourceCatalogs that hold the records themselves.
 *
 *  @note The table Catalog arguments must be sorted in ascending ID order on input; this will
 *        allow us to use binary search algorithms to find the sources referred to by the match
 *        table.
 *
 *  If an ID cannot be found in the given tables, that pointer will be set to null
 *  in the returned match vector.
 *
 *  @param[in]  matches     A normalized BaseCatalog representation, as produced by packMatches.
 *  @param[in]  first       A CatalogT containing the sources used on the 'first' side of the match,
 *                          sorted by ascending ID.
 *  @param[in]  second      A CatalogT containing the sources used on the 'second' side of the match,
 *                          sorted by ascending ID.  May be the same as first.
 */
inline SourceMatchVector unpackMatches(
    BaseCatalog const & matches, 
    SourceCatalog const & first,
    SourceCatalog const & second
) {
    return SourceMatch::static_vector_cast(
        unpackMatches(
            matches, first, SourceTable::getIdKey(), second, SourceTable::getIdKey()
        )
    );
}

}}} // namespace lsst::afw::table

#endif // #ifndef LSST_AFW_TABLE_SOURCEMATCH_H
