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
 
/** @file
  * @ingroup afw
  */
#include <algorithm>
#include <cmath>

#include "boost/scoped_array.hpp"

#include "lsst/utils/ieee.h"
#include "lsst/pex/exceptions.h"
#include "lsst/pex/logging/Trace.h"
#include "lsst/afw/table/Match.h"
#include "lsst/afw/geom/Angle.h"

namespace lsst { namespace afw { namespace table { namespace {

struct RecordPos {
    double dec;
    double x;
    double y;
    double z;
    // JFB removed extra pointer here; this may have performance implications, but hopefully not
    // significant ones.  BaseVector iterators yield temporary BaseRecord PTRs, so storing
    // their address was no longer an option.
    PTR(BaseRecord) src;
};

bool operator<(RecordPos const &s1, RecordPos const &s2) {
    return (s1.dec < s2.dec);
}

struct CmpRecordPtr {

    bool operator()(PTR(BaseRecord) const s1, PTR(BaseRecord) const s2) {
        return s1->get(yKey) < s2->get(yKey);
    }

    explicit CmpRecordPtr(Key<double> const & yKey_) : yKey(yKey_) {}

    Key<double> yKey;
};

/**
 * Extract source positions from @a set, convert them to cartesian coordinates
 * (for faster distance checks) and sort the resulting array of @c RecordPos
 * instances by declination. Records with positions containing a NaN are skipped.
 *
 * @param[in] set          set of sources to process
 * @param[out] positions   pointer to an array of at least @c set.size()
 *                         RecordPos instances
 * @return                 The number of sources with positions not containing a NaN.
 */
size_t makeRecordPositions(
    BaseVector const &set,
    Key<Coord> const & key,
    RecordPos *positions
) {
    size_t n = 0;
    Key<Angle> raKey = key.getRa();
    Key<Angle> decKey = key.getDec();
    for (BaseVector::const_iterator i(set.begin()), e(set.end()); i != e; ++i) {
        geom::Angle ra = i->get(raKey);
        geom::Angle dec = i->get(decKey);
        if (lsst::utils::isnan(ra.asRadians()) || lsst::utils::isnan(dec.asRadians())) {
            continue;
        }
        double cosDec    = std::cos(dec);
        positions[n].dec = dec.asRadians();
        positions[n].x   = std::cos(ra)*cosDec;
        positions[n].y   = std::sin(ra)*cosDec;
        positions[n].z   = std::sin(dec);
        positions[n].src = i;
        ++n;
    }
    std::sort(positions, positions + n);
    if (n < set.size()) {
        lsst::pex::logging::TTrace<1>("afw.table.matchRaDec",
                                      "At least one source had ra or dec equal to NaN");
    }
    return n;
}

} // <anonymous>


BaseMatchVector matchRaDec(
    BaseVector const & set1, Key<Coord> const & key1,
    BaseVector const & set2, Key<Coord> const & key2,
    geom::Angle radius, bool closest
) {
    if (&set1 == &set2) {
        return matchRaDec(set1, key1, radius, true);
    }
    if (radius < 0.0 || (radius > (45. * geom::degrees))) {
        throw LSST_EXCEPT(pex::exceptions::RangeErrorException, 
                          "match radius out of range (0 to 45 degrees)");
    }
    if (set1.size() == 0 || set2.size() == 0) {
        return BaseMatchVector();
    }
    // setup match parameters
    double const d2Limit = radius.toUnitSphereDistanceSquared();
    
    // Build position lists
    size_t len1 = set1.size();
    size_t len2 = set2.size();
    boost::scoped_array<RecordPos> pos1(new RecordPos[len1]);
    boost::scoped_array<RecordPos> pos2(new RecordPos[len2]);
    len1 = makeRecordPositions(set1, key1, pos1.get());
    len2 = makeRecordPositions(set2, key2, pos2.get());
    
    BaseMatchVector matches;
    for (size_t i = 0, start = 0; i < len1; ++i) {
        double minDec = pos1[i].dec - radius.asRadians();
        while (start < len2 && pos2[start].dec < minDec) { ++start; }
        if (start == len2) {
            break;
        }
        double maxDec = pos1[i].dec + radius.asRadians();
        size_t closestIndex = -1;          // Index of closest match (if any)
        double d2Include = d2Limit;     // Squared distance for inclusion of match
        bool found = false;             // Found anything?
        for (size_t j = start; j < len2 && pos2[j].dec <= maxDec; ++j) {
            double dx = pos1[i].x - pos2[j].x;
            double dy = pos1[i].y - pos2[j].y;
            double dz = pos1[i].z - pos2[j].z;
            double d2 = dx*dx + dy*dy + dz*dz;
            if (d2 < d2Include) {
                if (closest) {
                    d2Include = d2;
                    closestIndex = j;
                    found = true;
                } else {
                    matches.push_back(
                        BaseMatch(pos1[i].src, pos2[j].src, geom::Angle::fromUnitSphereDistanceSquared(d2))
                    );
                }
            }
        }
        if (closest && found) {
            matches.push_back(
                BaseMatch(pos1[i].src, pos2[closestIndex].src, geom::Angle::fromUnitSphereDistanceSquared(d2Include))
            );
        }
    }
    return matches;
}

BaseMatchVector matchRaDec(
    BaseVector const &set, Key<Coord> const & key, geom::Angle radius, bool symmetric
) {
    if (radius < 0.0 || radius > (45.0 * geom::degrees)) {
        throw LSST_EXCEPT(pex::exceptions::RangeErrorException,
                          "match radius out of range (0 to 45 degrees)");
    }
    if (set.size() == 0) {
        return BaseMatchVector();
    }
    // setup match parameters
    double const d2Limit = radius.toUnitSphereDistanceSquared();

    // Build position list
    size_t len = set.size();
    boost::scoped_array<RecordPos> pos(new RecordPos[len]);
    len = makeRecordPositions(set, key, pos.get());

    BaseMatchVector matches;
    for (size_t i = 0; i < len; ++i) {
        double maxDec = pos[i].dec + radius.asRadians();
        for (size_t j = i + 1; j < len && pos[j].dec <= maxDec; ++j) {
            double dx = pos[i].x - pos[j].x;
            double dy = pos[i].y - pos[j].y;
            double dz = pos[i].z - pos[j].z;
            double d2 = dx*dx + dy*dy + dz*dz;
            if (d2 < d2Limit) {
                geom::Angle d = geom::Angle::fromUnitSphereDistanceSquared(d2);
                matches.push_back(BaseMatch(pos[i].src, pos[j].src, d));
                if (symmetric) {
                    matches.push_back(BaseMatch(pos[j].src, pos[i].src, d));
                }
            }
        }
    }
    return matches;
}


BaseMatchVector matchXy(BaseVector const &set1, Key< Point<double> > const & key1,
                        BaseVector const &set2, Key< Point<double> > const & key2,
                        double radius, bool closest) {
    if (&set1 == &set2) {
        return matchXy(set1, key1, radius);
    }
    // setup match parameters
    double const r2 = radius*radius;

    // copy and sort array of pointers on y
    size_t const len1 = set1.size();
    size_t const len2 = set2.size();
    boost::scoped_array<PTR(BaseRecord)> pos1(new PTR(BaseRecord)[len1]);
    boost::scoped_array<PTR(BaseRecord)> pos2(new PTR(BaseRecord)[len2]);
    size_t n = 0;
    for (BaseVector::const_iterator i(set1.begin()), e(set1.end()); i != e; ++i, ++n) {
        pos1[n] = i;
    }
    n = 0;
    for (BaseVector::const_iterator i(set2.begin()), e(set2.end()); i != e; ++i, ++n) {
        pos2[n] = i;
    }

    Key<double> xKey1 = key1.getX();
    Key<double> yKey1 = key1.getY();
    Key<double> xKey2 = key2.getX();
    Key<double> yKey2 = key2.getY();

    std::sort(pos1.get(), pos1.get() + len1, CmpRecordPtr(yKey1));
    std::sort(pos2.get(), pos2.get() + len2, CmpRecordPtr(yKey2));

    BaseMatchVector matches;
    for (size_t i = 0, start = 0; i < len1; ++i) {
        double y = pos1[i]->get(yKey1);
        double minY = y - radius;
        while (start < len2 && pos2[start]->get(key2.getY()) < minY) { ++start; }
        if (start == len2) {
            break;
        }
        double x = pos1[i]->get(xKey1);
        double maxY = y + radius;
        double y2;
        size_t closestIndex = -1;          // Index of closest match (if any)
        double r2Include = r2;          // Squared radius for inclusion of match
        bool found = false;             // Found anything?
        for (size_t j = start; j < len2 && (y2 = pos2[j]->get(yKey2)) <= maxY; ++j) {
            double dx = x - pos2[j]->get(xKey2);
            double dy = y - y2;
            double d2 = dx*dx + dy*dy;
            if (d2 < r2Include) {
                if (closest) {
                    r2Include = d2;
                    closestIndex = j;
                    found = true;
                } else {
                    matches.push_back(BaseMatch(pos1[i], pos2[j], std::sqrt(d2)));
                }
            }
        }
        if (closest && found) {
            matches.push_back(BaseMatch(pos1[i], pos2[closestIndex], std::sqrt(r2Include)));
        }
    }
    return matches;
}

BaseMatchVector matchXy(
    BaseVector const & set, Key< Point<double> > const & key, double radius, bool symmetric
) {
    // setup match parameters
    double const r2 = radius*radius;

    // copy and sort array of pointers on y
    size_t const len = set.size();
    boost::scoped_array<PTR(BaseRecord)> pos(new PTR(BaseRecord)[len]);
    size_t n = 0;
    for (BaseVector::const_iterator i(set.begin()), e(set.end()); i != e; ++i, ++n) {
        pos[n] = i;
    }

    Key<double> xKey = key.getX();
    Key<double> yKey = key.getY();

    std::sort(pos.get(), pos.get() + len, CmpRecordPtr(yKey));

    BaseMatchVector matches;
    for (size_t i = 0; i < len; ++i) {
        double x = pos[i]->get(xKey);
        double y = pos[i]->get(yKey);
        double maxY = y + radius;
        double y2;
        for (size_t j = i + 1; j < len && (y2 = pos[j]->get(yKey)) <= maxY; ++j) {
            double dx = x - pos[j]->get(xKey);
            double dy = y - y2;
            double d2 = dx*dx + dy*dy;
            if (d2 < r2) {
                double d = std::sqrt(d2);
                matches.push_back(BaseMatch(pos[i], pos[j], d));
                if (symmetric) {
                    matches.push_back(BaseMatch(pos[j], pos[i], d));
                }
            }
        }
    }
    return matches;
}

BaseVector packMatches(
    BaseMatchVector const & matches,
    Key<RecordId> const & idKey1,
    Key<RecordId> const & idKey2
) {
    Schema schema;
    Key<RecordId> outKey1 = schema.addField<RecordId>("first", "ID for first source record in match.");
    Key<RecordId> outKey2 = schema.addField<RecordId>("second", "ID for second source record in match.");
    Key<double> keyD = schema.addField<double>("distance", "Distance between matches sources.");
    BaseVector result(schema);
    result.getTable()->preallocate(matches.size());
    result.reserve(matches.size());
    for (BaseMatchVector::const_iterator i = matches.begin(); i != matches.end(); ++i) {
        PTR(BaseRecord) record = result.addNew();
        record->set(outKey1, i->first->get(idKey1));
        record->set(outKey2, i->second->get(idKey2));
        record->set(keyD, i->distance);
    }
    return result;
}

BaseMatchVector unpackMatches(
    BaseVector const & matches, 
    BaseVector const & first, Key<RecordId> const & idKey1,
    BaseVector const & second, Key<RecordId> const & idKey2
) {
    Key<RecordId> inKey1 = matches.getSchema()["first"];
    Key<RecordId> inKey2 = matches.getSchema()["second"];
    Key<double> keyD = matches.getSchema()["distance"];
    if (!first.isSorted(idKey1) || !second.isSorted(idKey2)) 
        throw LSST_EXCEPT(
            pex::exceptions::InvalidParameterException,
            "Vectors passed to unpackMatches must be sorted."
        );
    BaseMatchVector result;
    result.resize(matches.size());
    BaseMatchVector::iterator j = result.begin();
    for (BaseVector::const_iterator i = matches.begin(); i != matches.end(); ++i, ++j) {
        j->first = first.find(i->get(inKey1), idKey1);
        j->second = second.find(i->get(inKey2), idKey2);
        j->distance = i->get(keyD);
    }
    return result;
}

}}} // namespace lsst::afw::table
