// -*- lsst-c++ -*-

/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */
 
#ifndef LSST_AFW_TABLE_MATCH_H
#define LSST_AFW_TABLE_MATCH_H

#include <vector>

#include "lsst/afw/table/fwd.h"
#include "lsst/afw/table/BaseRecord.h"
#include "lsst/afw/table/Simple.h"
#include "lsst/afw/table/Source.h"
#include "lsst/afw/table/Catalog.h"
#include "lsst/afw/geom/Angle.h"

namespace lsst { namespace afw { namespace table {

/**
 *  @brief Lightweight representation of a geometric match between two records.
 *
 *  This is a template so it can hold derived record classes without a lot of 
 *  casting and properly use Angle for the distance when we do spherical coordinate
 *  matches.
 */
template <typename Record1, typename Record2>
struct Match {
    PTR(Record1) first;
    PTR(Record2) second;
    double distance; // may be pixels or radians

    Match() : first(), second(), distance(0.0) {}
    
    Match(PTR(Record1) const & r1, PTR(Record2) const & r2, double dist)
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
 * Compute all tuples (s1,s2,d) where s1 belings to @a cat1, s2 belongs to @a cat2 and
 * d, the distance between s1 and s2, in pixels, is at most @a radius. If cat1 and
 * cat2 are identical, then this call is equivalent to @c matchXy(cat1,radius,true).
 * The match is performed in pixel space (2d cartesian).
 *
 * @param[in] cat1     first catalog
 * @param[in] cat2     second catalog
 * @param[in] radius   match radius (pixels)
 * @param[in] closest  if true then just return the closest match
 */
SourceMatchVector matchXy(
    SourceCatalog const &cat1, SourceCatalog const &cat2,
    double radius, bool closest=true
);

/**
 * Compute all tuples (s1,s2,d) where s1 != s2, s1 and s2 both belong to @a cat,
 * and d, the distance between s1 and s2, in pixels, is at most @a radius. The
 * match is performed in pixel space (2d cartesian).
 *
 * @param[in] cat          the catalog to self-match
 * @param[in] radius       match radius (pixels)
 * @param[in] symmetric    if cat to @c true symmetric matches are produced: i.e.
 *                         if (s1, s2, d) is reported, then so is (s2, s1, d).
 */
SourceMatchVector matchXy(SourceCatalog const &cat, double radius, bool symmetric = true);

#ifndef SWIG // swig will be confused by the nested names below; repeated with typedefs in match.i

/** 
 * Compute all tuples (s1,s2,d) where s1 belings to @a cat1, s2 belongs to @a cat2 and
 * d, the distance between s1 and s2, is at most @a radius. If cat1 and
 * cat2 are identical, then this call is equivalent to @c matchRaDec(cat1,radius,true).
 * The match is performed in ra, dec space.
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
    Angle radius, bool closest = true
);

/*
 * Compute all tuples (s1,s2,d) where s1 != s2, s1 and s2 both belong to @a cat,
 * and d, the distance between s1 and s2, is at most @a radius. The
 * match is performed in ra, dec space.
 *
 * @param[in] cat          the catalog to self-match
 * @param[in] radius       match radius
 * @param[in] symmetric    if cat to @c true symmetric matches are produced: i.e.
 *                         if (s1, s2, d) is reported, then so is (s2, s1, d).
 * @param[in] key          key used to extract the center
 *
 * This is instantiated for Simple and Source catalogs.
 */
template <typename Cat>
std::vector< Match< typename Cat::Record, typename Cat::Record> > matchRaDec(
    Cat const & cat,
    Angle radius,
    bool symmetric = true
);

/**
 *  @brief Return a table representation of a MatchVector that can be used to persist it.
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

#endif // !SWIG

}}} // namespace lsst::afw::table

#endif // #ifndef LSST_AFW_TABLE_MATCH_H
