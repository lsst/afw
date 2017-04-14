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

#ifndef LSST_AFW_TABLE_MATCH_H
#define LSST_AFW_TABLE_MATCH_H

#include <vector>

#include "lsst/pex/config.h"
#include "lsst/afw/table/fwd.h"
#include "lsst/afw/table/BaseRecord.h"
#include "lsst/afw/table/Simple.h"
#include "lsst/afw/table/Source.h"
#include "lsst/afw/table/Catalog.h"
#include "lsst/afw/geom/Angle.h"

namespace lsst { namespace afw { namespace table {

/**
 * Pass parameters to algorithms that match list of sources
 */
class MatchControl {
public:
    MatchControl()
        : findOnlyClosest(true),
          symmetricMatch(true),
          includeMismatches(false)
    { }
    LSST_CONTROL_FIELD(findOnlyClosest, bool, "Return only the closest match if more than one is found " \
                       "(default: true)");
    LSST_CONTROL_FIELD(symmetricMatch, bool,  "Produce symmetric matches (default: true):\n" \
                       "i.e. if (s1, s2, d) is reported, then so is (s2, s1, d)");
    LSST_CONTROL_FIELD(includeMismatches, bool, "Include failed matches (i.e. one 'match' is NULL) " \
                       "(default: false)");
};

/**
 *  Lightweight representation of a geometric match between two records.
 *
 *  This is a template so it can hold derived record classes without a lot of
 *  casting and properly use Angle for the distance when we do spherical coordinate
 *  matches.
 */
template <typename Record1, typename Record2>
struct Match {
    std::shared_ptr<Record1> first;
    std::shared_ptr<Record2> second;
    double distance; // may be pixels or radians

    Match() : first(), second(), distance(0.0) {}

    Match(std::shared_ptr<Record1> const & r1, std::shared_ptr<Record2> const & r2, double dist)
        : first(r1), second(r2), distance(dist) {}

    template <typename R1, typename R2>
    Match(Match<R1,R2> const & other) : first(other.first), second(other.second), distance(other.distance) {}

};

typedef Match<SimpleRecord,SimpleRecord> SimpleMatch;
typedef Match<SimpleRecord,SourceRecord> ReferenceMatch;
typedef Match<SourceRecord,SourceRecord> SourceMatch;

typedef std::vector<SimpleMatch> SimpleMatchVector;
typedef std::vector<ReferenceMatch> ReferenceMatchVector;
typedef std::vector<SourceMatch> SourceMatchVector;

/**
 * Compute all tuples (s1,s2,d) where s1 belings to `cat1`, s2 belongs to `cat2` and
 * d, the distance between s1 and s2, in pixels, is at most `radius`. If cat1 and
 * cat2 are identical, then this call is equivalent to `matchXy(cat1,radius)`.
 * The match is performed in pixel space (2d cartesian).
 */
SourceMatchVector matchXy(
    SourceCatalog const &cat1,          ///< first catalog
    SourceCatalog const &cat2,          ///< second catalog
    double radius,                      ///< match radius (pixels)
    MatchControl const& mc=MatchControl() ///< how to do the matching (obeys MatchControl::findOnlyClosest)
);

/**
 * Compute all tuples (s1,s2,d) where s1 != s2, s1 and s2 both belong to `cat`,
 * and d, the distance between s1 and s2, in pixels, is at most `radius`. The
 * match is performed in pixel space (2d cartesian).
 */
SourceMatchVector matchXy(
    SourceCatalog const &cat,          ///< the catalog to self-match
    double radius,                     ///< match radius (pixels)
    MatchControl const& mc=MatchControl() ///< how to do the matching (obeys MatchControl::symmetricMatch)
);

/**
 * Compute all tuples (s1,s2,d) where s1 belings to `cat1`, s2 belongs to `cat2` and
 * d, the distance between s1 and s2, in pixels, is at most `radius`. If cat1 and
 * cat2 are identical, then this call is equivalent to `matchXy(cat1,radius)`.
 * The match is performed in pixel space (2d cartesian).
 *
 * @deprecated  Please use the matchXy(..., MatchControl const&) API
 *
 * @param[in] cat1     first catalog
 * @param[in] cat2     second catalog
 * @param[in] radius   match radius (pixels)
 * @param[in] closest  if true then just return the closest match
 */
SourceMatchVector matchXy(
    SourceCatalog const &cat1,
    SourceCatalog const &cat2,
    double radius,
    bool closest
);

/**
 * Compute all tuples (s1,s2,d) where s1 != s2, s1 and s2 both belong to `cat`,
 * and d, the distance between s1 and s2, in pixels, is at most `radius`. The
 * match is performed in pixel space (2d cartesian).
 *
 * @deprecated  Please use the matchXy(..., MatchControl const&) API
 *
 * @param[in] cat          the catalog to self-match
 * @param[in] radius       match radius (pixels)
 * @param[in] symmetric    if cat to `true` symmetric matches are produced: i.e.
 *                         if (s1, s2, d) is reported, then so is (s2, s1, d).
 */
SourceMatchVector matchXy(SourceCatalog const &cat, double radius, bool symmetric);

/**
 * Compute all tuples (s1,s2,d) where s1 belings to `cat1`, s2 belongs to `cat2` and
 * d, the distance between s1 and s2, is at most `radius`. If cat1 and
 * cat2 are identical, then this call is equivalent to `matchRaDec(cat1,radius)`.
 * The match is performed in ra, dec space.
 *
 * This is instantiated for Simple-Simple, Simple-Source, and Source-Source catalog combinations.
 */
template <typename Cat1, typename Cat2>
std::vector< Match< typename Cat1::Record, typename Cat2::Record> > matchRaDec(
    Cat1 const & cat1,                  ///< first catalog
    Cat2 const & cat2,                  ///< second catalog
    Angle radius,                       ///< match radius
    MatchControl const& mc=MatchControl() ///< how to do the matching (obeys MatchControl::findOnlyClosest)
);

/*
 * Compute all tuples (s1,s2,d) where s1 != s2, s1 and s2 both belong to `cat`,
 * and d, the distance between s1 and s2, is at most `radius`. The
 * match is performed in ra, dec space.
 *
 * This is instantiated for Simple and Source catalogs.
 */
template <typename Cat>
std::vector< Match< typename Cat::Record, typename Cat::Record> > matchRaDec(
    Cat const & cat,                    ///< the catalog to self-match
    Angle radius,                       ///< match radius
    MatchControl const& mc=MatchControl() ///< how to do the matching (obeys MatchControl::symmetricMatch)
);

/**
 * Compute all tuples (s1,s2,d) where s1 belings to `cat1`, s2 belongs to `cat2` and
 * d, the distance between s1 and s2, is at most `radius`. If cat1 and
 * cat2 are identical, then this call is equivalent to `matchRaDec(cat1,radius)`.
 * The match is performed in ra, dec space.
 *
 * @deprecated.  Use the matchRaDec(..., MatchControl) version
 *
 * @param[in] cat1     first catalog
 * @param[in] cat2     second catalog
 * @param[in] radius   match radius
 * @param[in] closest  if true then just return the closest match
 *
 * This is instantiated for Simple-Simple, Simple-Source, and Source-Source catalog combinations.
 */
template <typename Cat1, typename Cat2>
std::vector< Match< typename Cat1::Record, typename Cat2::Record> > matchRaDec(
    Cat1 const & cat1,
    Cat2 const & cat2,
    Angle radius, bool closest
);

/**
 * Compute all tuples (s1,s2,d) where s1 != s2, s1 and s2 both belong to `cat`,
 * and d, the distance between s1 and s2, is at most `radius`. The
 * match is performed in ra, dec space.
 *
 * @deprecated.  Use the matchRaDec(..., MatchControl) version
 *
 * @param[in] cat          the catalog to self-match
 * @param[in] radius       match radius
 * @param[in] symmetric    if cat to `true` symmetric matches are produced: i.e.
 *                         if (s1, s2, d) is reported, then so is (s2, s1, d).
 *
 * This is instantiated for Simple and Source catalogs.
 */
template <typename Cat>
std::vector< Match< typename Cat::Record, typename Cat::Record> > matchRaDec(
    Cat const & cat,
    Angle radius,
    bool symmetric
);

/**
 *  Return a table representation of a MatchVector that can be used to persist it.
 *
 *  The schema of the returned object has "first" (RecordId), "second" (RecordID), and "distance"
 *  (double) fields.
 *
 *  @param[in]  matches     A std::vector of Match objects to convert to table form.
 */
template <typename Record1, typename Record2>
BaseCatalog packMatches(std::vector< Match<Record1,Record2> > const & matches);

/**
 *  @brief Reconstruct a MatchVector from a BaseCatalog representation of the matches
 *         and a pair of catalogs.
 *
 *  @note The first and second catalog arguments must be sorted in ascending ID order on input; this will
 *        allow us to use binary search algorithms to find the records referred to by the match
 *        table.
 *
 *  If an ID cannot be found in the given tables, that pointer will be set to null
 *  in the returned match vector.
 *
 *  @param[in]  matches     A normalized BaseCatalog representation, as produced by packMatches.
 *  @param[in]  cat1        A CatalogT containing the records used on the 'first' side of the match,
 *                          sorted by ascending ID.
 *  @param[in]  cat2        A CatalogT containing the records used on the 'second' side of the match,
 *                          sorted by ascending ID.  May be the same as first.
 *
 * This is instantiated for Simple-Simple, Simple-Source, and Source-Source catalog combinations.
 */
template <typename Cat1, typename Cat2>
std::vector< Match< typename Cat1::Record, typename Cat2::Record> >
unpackMatches(BaseCatalog const & matches, Cat1 const & cat1, Cat2 const & cat2);

}}} // namespace lsst::afw::table

#endif // #ifndef LSST_AFW_TABLE_MATCH_H
