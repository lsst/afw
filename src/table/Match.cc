// -*- lsst-c++ -*-

/*
 * LSST Data Management System
 * Copyright 2008, 2009, 2010, 2016 LSST Corporation.
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

#include <algorithm>
#include <cmath>
#include <memory>

#include "lsst/pex/exceptions.h"
#include "lsst/log/Log.h"
#include "lsst/geom/Angle.h"
#include "lsst/afw/table/Match.h"

namespace lsst {
namespace afw {
namespace table {
namespace {

template <typename RecordT>
struct RecordPos {
    double dec;
    double x;
    double y;
    double z;
    // JFB removed extra pointer here; this may have performance implications, but hopefully not
    // significant ones.  BaseCatalog iterators yield temporary BaseRecord shared_ptrs, so storing
    // their address was no longer an option.
    std::shared_ptr<RecordT> src;
};

template <typename Record1, typename Record2>
bool operator<(RecordPos<Record1> const &s1, RecordPos<Record2> const &s2) {
    return (s1.dec < s2.dec);
}

struct CmpRecordPtr {
    bool operator()(std::shared_ptr<SourceRecord> const s1, std::shared_ptr<SourceRecord> const s2) {
        return s1->getY() < s2->getY();
    }
};

/**
 * @internal Extract source positions from `set`, convert them to cartesian coordinates
 * (for faster distance checks) and sort the resulting array of `RecordPos`
 * instances by declination. Records with positions containing a NaN are skipped.
 *
 * @param[in] set          set of sources to process
 * @param[out] positions   pointer to an array of at least `set.size()`
 *                         RecordPos instances
 * @returns                 The number of sources with positions not containing a NaN.
 */
template <typename Cat>
size_t makeRecordPositions(Cat const &cat, RecordPos<typename Cat::Record> *positions) {
    size_t n = 0;
    Key<lsst::geom::Angle> raKey = Cat::Table::getCoordKey().getRa();
    Key<lsst::geom::Angle> decKey = Cat::Table::getCoordKey().getDec();
    for (typename Cat::const_iterator i(cat.begin()), e(cat.end()); i != e; ++i) {
        lsst::geom::Angle ra = i->get(raKey);
        lsst::geom::Angle dec = i->get(decKey);
        if (std::isnan(ra.asRadians()) || std::isnan(dec.asRadians())) {
            continue;
        }
        double cosDec = std::cos(dec);
        positions[n].dec = dec.asRadians();
        positions[n].x = std::cos(ra) * cosDec;
        positions[n].y = std::sin(ra) * cosDec;
        positions[n].z = std::sin(dec);
        positions[n].src = i;
        ++n;
    }
    std::sort(positions, positions + n);
    if (n < cat.size()) {
        LOGLS_WARN("afw.table.matchRaDec", "At least one source had ra or dec equal to NaN");
    }
    return n;
}

template size_t makeRecordPositions(SimpleCatalog const &, RecordPos<SimpleRecord> *);
template size_t makeRecordPositions(SourceCatalog const &, RecordPos<SourceRecord> *);

template <typename Cat1, typename Cat2>
bool doSelfMatchIfSame(std::vector<Match<typename Cat1::Record, typename Cat2::Record> > &result,
                       Cat1 const &cat1, Cat2 const &cat2, lsst::geom::Angle radius) {
    // types are different, so the catalogs are never the same.
    return false;
}

template <typename Cat>
bool doSelfMatchIfSame(std::vector<Match<typename Cat::Record, typename Cat::Record> > &result,
                       Cat const &cat1, Cat const &cat2, lsst::geom::Angle radius) {
    if (&cat1 == &cat2) {
        result = matchRaDec(cat1, radius);
        return true;
    }
    return false;
}

/**
 * @internal Return the squared distance between two unit vectors separated by an angle.
 *
 * This distance is given by @f$ |\vec{u} - \vec{v}| = 2 \sin(\theta/2) @f$.
 *
 * @param theta the angle between two unit vectors
 * @returns the squared distance between the two vectors
 */
double toUnitSphereDistanceSquared(lsst::geom::Angle theta) noexcept {
    return 2. * (1. - std::cos(theta.asRadians()));
    // == 4.0 * pow(std::sin(0.5 * theta.asRadians()), 2.0)
}

/**
 * @internal Return the angle between two unit vectors.
 *
 * This angle is given by @f$ \sin(\theta/2) = |\vec{u} - \vec{v}|/2 @f$.
 *
 * @param d2 the squared distance between two unit vectors
 * @returns the angle between the two vectors
 */
lsst::geom::Angle fromUnitSphereDistanceSquared(double d2) noexcept {
    // acos(1 - 0.5*d2) doesn't require sqrt but isn't as precise for small d2
    return 2.0*std::asin(0.5*std::sqrt(d2))*lsst::geom::radians;
}

}  // anonymous

template <typename Cat1, typename Cat2>
std::vector<Match<typename Cat1::Record, typename Cat2::Record> > matchRaDec(Cat1 const &cat1,
                                                                             Cat2 const &cat2, lsst::geom::Angle radius,
                                                                             bool closest) {
    MatchControl mc;
    mc.findOnlyClosest = closest;

    return matchRaDec(cat1, cat2, radius, mc);
}

template <typename Cat1, typename Cat2>
std::vector<Match<typename Cat1::Record, typename Cat2::Record> > matchRaDec(Cat1 const &cat1,
                                                                             Cat2 const &cat2, lsst::geom::Angle radius,
                                                                             MatchControl const &mc) {
    typedef Match<typename Cat1::Record, typename Cat2::Record> MatchT;
    std::vector<MatchT> matches;

    if (doSelfMatchIfSame(matches, cat1, cat2, radius)) return matches;

    if (radius < 0.0 || (radius > (45. * lsst::geom::degrees))) {
        throw LSST_EXCEPT(pex::exceptions::RangeError, "match radius out of range (0 to 45 degrees)");
    }
    if (cat1.size() == 0 || cat2.size() == 0) {
        return matches;
    }
    // setup match parameters
    double const d2Limit = toUnitSphereDistanceSquared(radius);

    // Build position lists
    size_t len1 = cat1.size();
    size_t len2 = cat2.size();

    typedef RecordPos<typename Cat1::Record> Pos1;
    typedef RecordPos<typename Cat2::Record> Pos2;
    std::unique_ptr<Pos1[]> pos1(new Pos1[len1]);
    std::unique_ptr<Pos2[]> pos2(new Pos2[len2]);
    len1 = makeRecordPositions(cat1, pos1.get());
    len2 = makeRecordPositions(cat2, pos2.get());
    std::shared_ptr<typename Cat2::Record> nullRecord = std::shared_ptr<typename Cat2::Record>();

    for (size_t i = 0, start = 0; i < len1; ++i) {
        double minDec = pos1[i].dec - radius.asRadians();
        while (start < len2 && pos2[start].dec < minDec) {
            ++start;
        }
        if (start == len2) {
            break;
        }
        double maxDec = pos1[i].dec + radius.asRadians();
        size_t closestIndex = -1;    // Index of closest match (if any)
        double d2Include = d2Limit;  // Squared distance for inclusion of match
        bool found = false;          // Found anything?
        size_t nMatches = 0;         // Number of matches
        for (size_t j = start; j < len2 && pos2[j].dec <= maxDec; ++j) {
            double dx = pos1[i].x - pos2[j].x;
            double dy = pos1[i].y - pos2[j].y;
            double dz = pos1[i].z - pos2[j].z;
            double d2 = dx * dx + dy * dy + dz * dz;
            if (d2 < d2Include) {
                if (mc.findOnlyClosest) {
                    d2Include = d2;
                    closestIndex = j;
                    found = true;
                } else {
                    matches.push_back(MatchT(pos1[i].src, pos2[j].src, fromUnitSphereDistanceSquared(d2)));
                }
                ++nMatches;
            }
        }
        if (mc.includeMismatches && nMatches == 0) {
            matches.push_back(MatchT(pos1[i].src, nullRecord, NAN));
        }
        if (mc.findOnlyClosest && found) {
            matches.push_back(
                    MatchT(pos1[i].src, pos2[closestIndex].src, fromUnitSphereDistanceSquared(d2Include)));
        }
    }
    return matches;
}

#define LSST_MATCH_RADEC(RTYPE, C1, C2)                             \
    template RTYPE matchRaDec(C1 const &, C2 const &, lsst::geom::Angle, bool); \
    template RTYPE matchRaDec(C1 const &, C2 const &, lsst::geom::Angle, MatchControl const &)

LSST_MATCH_RADEC(SimpleMatchVector, SimpleCatalog, SimpleCatalog);
LSST_MATCH_RADEC(ReferenceMatchVector, SimpleCatalog, SourceCatalog);
LSST_MATCH_RADEC(SourceMatchVector, SourceCatalog, SourceCatalog);

#undef LSST_MATCH_RADEC

template <typename Cat>
std::vector<Match<typename Cat::Record, typename Cat::Record> > matchRaDec(Cat const &cat,
                                                                           lsst::geom::Angle radius,
                                                                           bool symmetric) {
    MatchControl mc;
    mc.symmetricMatch = symmetric;

    return matchRaDec(cat, radius, mc);
}

template <typename Cat>
std::vector<Match<typename Cat::Record, typename Cat::Record> > matchRaDec(Cat const &cat,
                                                                           lsst::geom::Angle radius,
                                                                           MatchControl const &mc) {
    typedef Match<typename Cat::Record, typename Cat::Record> MatchT;
    std::vector<MatchT> matches;

    if (radius < 0.0 || radius > (45.0 * lsst::geom::degrees)) {
        throw LSST_EXCEPT(pex::exceptions::RangeError, "match radius out of range (0 to 45 degrees)");
    }
    if (cat.size() == 0) {
        return matches;
    }
    // setup match parameters
    double const d2Limit = toUnitSphereDistanceSquared(radius);

    // Build position list
    size_t len = cat.size();
    typedef RecordPos<typename Cat::Record> Pos;
    std::unique_ptr<Pos[]> pos(new Pos[len]);
    len = makeRecordPositions(cat, pos.get());

    for (size_t i = 0; i < len; ++i) {
        double maxDec = pos[i].dec + radius.asRadians();
        for (size_t j = i + 1; j < len && pos[j].dec <= maxDec; ++j) {
            double dx = pos[i].x - pos[j].x;
            double dy = pos[i].y - pos[j].y;
            double dz = pos[i].z - pos[j].z;
            double d2 = dx * dx + dy * dy + dz * dz;
            if (d2 < d2Limit) {
                lsst::geom::Angle d = fromUnitSphereDistanceSquared(d2);
                matches.push_back(MatchT(pos[i].src, pos[j].src, d));
                if (mc.symmetricMatch) {
                    matches.push_back(MatchT(pos[j].src, pos[i].src, d));
                }
            }
        }
    }
    return matches;
}

#define LSST_MATCH_RADEC(RTYPE, C)                     \
    template RTYPE matchRaDec(C const &, lsst::geom::Angle, bool); \
    template RTYPE matchRaDec(C const &, lsst::geom::Angle, MatchControl const &)

LSST_MATCH_RADEC(SimpleMatchVector, SimpleCatalog);
LSST_MATCH_RADEC(SourceMatchVector, SourceCatalog);

#undef LSST_MATCH_RADEC

SourceMatchVector matchXy(SourceCatalog const &cat1, SourceCatalog const &cat2, double radius, bool closest) {
    MatchControl mc;
    mc.findOnlyClosest = closest;

    return matchXy(cat1, cat2, radius, mc);
}

SourceMatchVector matchXy(SourceCatalog const &cat1, SourceCatalog const &cat2, double radius,
                          MatchControl const &mc) {
    if (&cat1 == &cat2) {
        return matchXy(cat1, radius);
    }
    // setup match parameters
    double const r2 = radius * radius;

    // copy and sort array of pointers on y
    size_t len1 = cat1.size();
    size_t len2 = cat2.size();
    std::unique_ptr<std::shared_ptr<SourceRecord>[]> pos1(new std::shared_ptr<SourceRecord>[len1]);
    std::unique_ptr<std::shared_ptr<SourceRecord>[]> pos2(new std::shared_ptr<SourceRecord>[len2]);
    std::shared_ptr<SourceRecord> nullRecord = std::shared_ptr<SourceRecord>();
    size_t n = 0;
    for (SourceCatalog::const_iterator i(cat1.begin()), e(cat1.end()); i != e; ++i) {
        if (std::isnan(i->getX()) || std::isnan(i->getY())) {
            continue;
        }
        pos1[n] = i;
        ++n;
    }
    len1 = n;
    n = 0;
    for (SourceCatalog::const_iterator i(cat2.begin()), e(cat2.end()); i != e; ++i) {
        if (std::isnan(i->getX()) || std::isnan(i->getY())) {
            continue;
        }
        pos2[n] = i;
        ++n;
    }
    len2 = n;

    std::sort(pos1.get(), pos1.get() + len1, CmpRecordPtr());
    std::sort(pos2.get(), pos2.get() + len2, CmpRecordPtr());

    SourceMatchVector matches;
    for (size_t i = 0, start = 0; i < len1; ++i) {
        double y = pos1[i]->getY();
        double minY = y - radius;
        while (start < len2 && pos2[start]->getY() < minY) {
            ++start;
        }
        if (start == len2) {
            break;
        }
        double x = pos1[i]->getX();
        double maxY = y + radius;
        double y2;
        size_t closestIndex = -1;  // Index of closest match (if any)
        double r2Include = r2;     // Squared radius for inclusion of match
        bool found = false;        // Found anything?
        size_t nMatches = 0;       // Number of matches
        for (size_t j = start; j < len2 && (y2 = pos2[j]->getY()) <= maxY; ++j) {
            double dx = x - pos2[j]->getX();
            double dy = y - y2;
            double d2 = dx * dx + dy * dy;
            if (d2 < r2Include) {
                if (mc.findOnlyClosest) {
                    r2Include = d2;
                    closestIndex = j;
                    found = true;
                } else {
                    matches.push_back(SourceMatch(pos1[i], pos2[j], std::sqrt(d2)));
                }
                ++nMatches;
            }
        }
        if (mc.includeMismatches && nMatches == 0) {
            matches.push_back(SourceMatch(pos1[i], nullRecord, NAN));
        }
        if (mc.findOnlyClosest && found) {
            matches.push_back(SourceMatch(pos1[i], pos2[closestIndex], std::sqrt(r2Include)));
        }
    }
    return matches;
}

SourceMatchVector matchXy(SourceCatalog const &cat, double radius, bool symmetric) {
    MatchControl mc;
    mc.symmetricMatch = symmetric;

    return matchXy(cat, radius, mc);
}

SourceMatchVector matchXy(SourceCatalog const &cat, double radius, MatchControl const &mc) {
    // setup match parameters
    double const r2 = radius * radius;

    // copy and sort array of pointers on y
    size_t len = cat.size();
    std::unique_ptr<std::shared_ptr<SourceRecord>[]> pos(new std::shared_ptr<SourceRecord>[len]);
    size_t n = 0;
    for (SourceCatalog::const_iterator i(cat.begin()), e(cat.end()); i != e; ++i) {
        if (std::isnan(i->getX()) || std::isnan(i->getY())) {
            continue;
        }
        pos[n] = i;
        ++n;
    }
    len = n;

    std::sort(pos.get(), pos.get() + len, CmpRecordPtr());

    SourceMatchVector matches;
    for (size_t i = 0; i < len; ++i) {
        double x = pos[i]->getX();
        double y = pos[i]->getY();
        double maxY = y + radius;
        double y2;
        for (size_t j = i + 1; j < len && (y2 = pos[j]->getY()) <= maxY; ++j) {
            double dx = x - pos[j]->getX();
            double dy = y - y2;
            double d2 = dx * dx + dy * dy;
            if (d2 < r2) {
                double d = std::sqrt(d2);
                matches.push_back(SourceMatch(pos[i], pos[j], d));
                if (mc.symmetricMatch) {
                    matches.push_back(SourceMatch(pos[j], pos[i], d));
                }
            }
        }
    }
    return matches;
}

template <typename Record1, typename Record2>
BaseCatalog packMatches(std::vector<Match<Record1, Record2> > const &matches) {
    Schema schema;
    Key<RecordId> outKey1 = schema.addField<RecordId>("first", "ID for first source record in match.");
    Key<RecordId> outKey2 = schema.addField<RecordId>("second", "ID for second source record in match.");
    Key<double> keyD = schema.addField<double>("distance", "Distance between matches sources.");
    BaseCatalog result(schema);
    result.getTable()->preallocate(matches.size());
    result.reserve(matches.size());
    typedef typename std::vector<Match<Record1, Record2> >::const_iterator Iter;
    for (Iter i = matches.begin(); i != matches.end(); ++i) {
        std::shared_ptr<BaseRecord> record = result.addNew();
        record->set(outKey1, i->first->getId());
        record->set(outKey2, i->second->getId());
        record->set(keyD, i->distance);
    }
    return result;
}

template BaseCatalog packMatches(SimpleMatchVector const &);
template BaseCatalog packMatches(ReferenceMatchVector const &);
template BaseCatalog packMatches(SourceMatchVector const &);

template <typename Cat1, typename Cat2>
std::vector<Match<typename Cat1::Record, typename Cat2::Record> > unpackMatches(BaseCatalog const &matches,
                                                                                Cat1 const &first,
                                                                                Cat2 const &second) {
    LOG_LOGGER tableLog = LOG_GET("afw.table");
    Key<RecordId> inKey1 = matches.getSchema()["first"];
    Key<RecordId> inKey2 = matches.getSchema()["second"];
    Key<double> keyD = matches.getSchema()["distance"];
    if (!first.isSorted() || !second.isSorted())
        throw LSST_EXCEPT(pex::exceptions::InvalidParameterError,
                          "Catalogs passed to unpackMatches must be sorted.");
    typedef Match<typename Cat1::Record, typename Cat2::Record> MatchT;
    std::vector<MatchT> result;
    result.resize(matches.size());
    typename std::vector<MatchT>::iterator j = result.begin();
    for (BaseCatalog::const_iterator i = matches.begin(); i != matches.end(); ++i, ++j) {
        typename Cat1::const_iterator k1 = first.find(i->get(inKey1));
        typename Cat2::const_iterator k2 = second.find(i->get(inKey2));
        if (k1 != first.end()) {
            j->first = k1;
        } else {
            LOGLS_WARN(tableLog, "Persisted match record with ID " << i->get(inKey1)
                                                                   << " not found in catalog 1.");
        }
        if (k2 != second.end()) {
            j->second = k2;
        } else {
            LOGLS_WARN(tableLog, "Persisted match record with ID " << i->get(inKey2)
                                                                   << " not found in catalog 2.");
        }
        j->distance = i->get(keyD);
    }
    return result;
}

template SimpleMatchVector unpackMatches(BaseCatalog const &, SimpleCatalog const &, SimpleCatalog const &);
template ReferenceMatchVector unpackMatches(BaseCatalog const &, SimpleCatalog const &,
                                            SourceCatalog const &);
template SourceMatchVector unpackMatches(BaseCatalog const &, SourceCatalog const &, SourceCatalog const &);
}
}
}  // namespace lsst::afw::table
