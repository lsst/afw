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
 
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE SourceMatch

#include <cmath>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
#include "boost/test/unit_test.hpp"
#pragma clang diagnostic pop
#include "boost/test/floating_point_comparison.hpp"

#include "lsst/afw/table/Source.h"
#include "lsst/afw/table/Match.h"
#include "lsst/afw/math/Random.h"
#include "lsst/afw/geom/Angle.h"

namespace math = lsst::afw::math;
namespace coord = lsst::afw::coord;
namespace afwGeom = lsst::afw::geom;
namespace afwTable = lsst::afw::table;

namespace {

math::Random & rng() {
    static math::Random * generator = 0;
    if (generator == 0) {
        generator = new math::Random(math::Random::MT19937);
    }
    return *generator;
}

PTR(afwTable::SourceTable) getGlobalTable() {
    static PTR(afwTable::SourceTable) table;
    if (!table) {
        afwTable::Schema schema = afwTable::SourceTable::makeMinimalSchema();
        schema.setVersion(0);
        afwTable::addCentroidFields(schema, "centroid", "dummy centroid");
        table = afwTable::SourceTable::make(schema);
        table->defineCentroid("centroid");
    }
    return table;
}

// Randomly generate a set of sources that are uniformly distributed across
// the unit sphere (ra/dec space) and the unit box (x,y space).
void makeSources(afwTable::SourceCatalog &set, int n) {
    for (int i = 0; i < n; ++i) {
        PTR(afwTable::SourceRecord) src = set.addNew();
        src->setId(i);
        src->set(set.getTable()->getCentroidKey(), afwGeom::Point2D(rng().uniform(), rng().uniform()));
        double z = rng().flat(-1.0, 1.0);
        src->set(afwTable::SourceTable::getCoordKey().getRa(), rng().flat(0.0, 360.) * afwGeom::degrees);
        src->set(afwTable::SourceTable::getCoordKey().getDec(), std::asin(z) * afwGeom::radians);
    }
}

struct CmpSourceMatch {
    bool operator()(afwTable::SourceMatch const &m1, afwTable::SourceMatch const &m2) {
        if (m1.first->getId() == m2.first->getId()) {
            return m1.second->getId() < m2.second->getId();
        }
        return m1.first->getId() < m2.first->getId(); 
    }
};

struct DistRaDec {
    double operator()(PTR(afwTable::SourceRecord) const &s1, PTR(afwTable::SourceRecord) const &s2) const {
        // halversine distance formula
        double sinDeltaRa = std::sin(0.5*(s2->getRa() - s1->getRa()));
        double sinDeltaDec = std::sin(0.5*(s2->getDec() - s1->getDec()));
        double cosDec1CosDec2 = std::cos(s1->getDec())*std::cos(s2->getDec());
        double a = sinDeltaDec*sinDeltaDec + cosDec1CosDec2*sinDeltaRa*sinDeltaRa;
        double b = std::sqrt(a);
        double c = b > 1 ? 1 : b;
        // radians
        return 2.0 * std::asin(c);
    }
};

struct DistXy {
    double operator()(PTR(afwTable::SourceRecord) const &s1, PTR(afwTable::SourceRecord) const &s2) const {
        double dx = s2->getX() - s1->getX();
        double dy = s2->getY() - s1->getY();
        return std::sqrt(dx*dx + dy*dy);
    }
};

template <typename DistFunctorT>
std::vector<afwTable::SourceMatch> bruteMatch(afwTable::SourceCatalog const &set,
                                         double radius,
                                         DistFunctorT const &distFun) {
    std::vector<afwTable::SourceMatch> matches;
    for (afwTable::SourceCatalog::const_iterator i1(set.begin()), e(set.end()); i1 != e; ++i1) {
        for (afwTable::SourceCatalog::const_iterator i2(i1); i2 != e; ++i2) {
            if (i1 == i2) {
                continue;
            }
            double d = distFun(i1, i2);
            if (d <= radius) {
               matches.push_back(afwTable::SourceMatch(i1, i2, d));
               matches.push_back(afwTable::SourceMatch(i2, i1, d));
            }
        }
    }
    return matches;
}

template <typename DistFunctorT>
std::vector<afwTable::SourceMatch> bruteMatch(afwTable::SourceCatalog const &set1,
                                         afwTable::SourceCatalog const &set2,
                                         double radius,
                                         DistFunctorT const &distFun) {
    if (&set1 == &set2) {
        return bruteMatch(set1, radius, distFun);
    }
    std::vector<afwTable::SourceMatch> matches;
    for (afwTable::SourceCatalog::const_iterator i1(set1.begin()), e1(set1.end()); i1 != e1; ++i1) {
        for (afwTable::SourceCatalog::const_iterator i2(set2.begin()), e2(set2.end()); i2 != e2; ++i2) {
            double d = distFun(i1, i2);
            if (d <= radius) {
                matches.push_back(afwTable::SourceMatch(i1, i2, d));
            }
        }
    }
    return matches;
}

// The distance computation for the brute-force and actual match algorithm is different,
// so we cannot naively test whether both match lists are identical. Instead,  check
// that any tuple in one result set but not the other has a match distance very close
// to the match radius.
void compareMatches(std::vector<afwTable::SourceMatch> &matches,
                    std::vector<afwTable::SourceMatch> &refMatches,
                    double radius) {
    double const tolerance = 1e-6; // 1 micro arcsecond
    CmpSourceMatch lessThan;

    std::sort(matches.begin(), matches.end(), lessThan);
    std::sort(refMatches.begin(), refMatches.end(), lessThan);
    std::vector<afwTable::SourceMatch>::const_iterator i(matches.begin());
    std::vector<afwTable::SourceMatch>::const_iterator j(refMatches.begin());
    std::vector<afwTable::SourceMatch>::const_iterator const iend(matches.end());
    std::vector<afwTable::SourceMatch>::const_iterator const jend(refMatches.end());

    while (i < iend && j < jend) {
        if (lessThan(*i, *j)) {
            BOOST_CHECK(std::fabs(i->distance - radius) <= tolerance);
            ++i;
        } else if (lessThan(*j, *i)) {
            BOOST_CHECK(std::fabs(j->distance - radius) <= tolerance);
            ++j;
        } else {
            BOOST_CHECK(std::fabs(i->distance - j->distance) <= tolerance);
            ++i;
            ++j;
        }
    }
    for (; i < iend; ++i) {
        BOOST_CHECK(std::fabs(i->distance - radius) <= tolerance);
    }
    for (; j < jend; ++j) {
        BOOST_CHECK(std::fabs(j->distance - radius) <= tolerance);
    }
}

} // namespace <anonymous>


BOOST_AUTO_TEST_CASE(matchRaDec) { /* parasoft-suppress  LsstDm-3-2a LsstDm-3-4a LsstDm-4-6 LsstDm-5-25 "Boost non-Std" */
    int const N = 500;    // # of points to generate
    double const M = 8.0; // avg. # of matches
    afwGeom::Angle radius = std::acos(1.0 - 2.0*M/N) * afwGeom::radians;

    afwTable::SourceCatalog set1(getGlobalTable()), set2(getGlobalTable());
    makeSources(set1, N);
    makeSources(set2, N);
    std::vector<afwTable::SourceMatch> matches = afwTable::matchRaDec(set1, set2, radius, false);
    std::vector<afwTable::SourceMatch> refMatches = bruteMatch(set1, set2, radius, DistRaDec()); 
    compareMatches(matches, refMatches, radius);
}

BOOST_AUTO_TEST_CASE(matchSelfRaDec) { /* parasoft-suppress  LsstDm-3-2a LsstDm-3-4a LsstDm-4-6 LsstDm-5-25 "Boost non-Std" */
    int const N = 500;    // # of points to generate
    double const M = 8.0; // avg. # of matches
    afwGeom::Angle radius = std::acos(1.0 - 2.0*M/N) * afwGeom::radians;

    afwTable::SourceCatalog set(getGlobalTable());
    makeSources(set, N);
    std::vector<afwTable::SourceMatch> matches = afwTable::matchRaDec(set, radius, true);
    std::vector<afwTable::SourceMatch> refMatches = bruteMatch(set, radius, DistRaDec());
    compareMatches(matches, refMatches, radius);
}

BOOST_AUTO_TEST_CASE(matchXy) { /* parasoft-suppress  LsstDm-3-2a LsstDm-3-4a LsstDm-4-6 LsstDm-5-25 "Boost non-Std" */
    int const N = 500;    // # of points to generate
    double const M = 8.0; // avg. # of matches
    double const radius = std::sqrt(M/(afwGeom::PI*static_cast<double>(N)));

    afwTable::SourceCatalog set1(getGlobalTable()), set2(getGlobalTable());
    makeSources(set1, N);
    makeSources(set2, N);
    std::vector<afwTable::SourceMatch> matches = afwTable::matchXy(set1, set2, radius, false);
    std::vector<afwTable::SourceMatch> refMatches = bruteMatch(set1, set2, radius, DistXy());
    compareMatches(matches, refMatches, radius);
}

BOOST_AUTO_TEST_CASE(matchSelfXy) { /* parasoft-suppress  LsstDm-3-2a LsstDm-3-4a LsstDm-4-6 LsstDm-5-25 "Boost non-Std" */
    int const N = 500;    // # of points to generate
    double const M = 8.0; // avg. # of matches
    double const radius = std::sqrt(M/(afwGeom::PI*static_cast<double>(N)));

    afwTable::SourceCatalog set(getGlobalTable());
    makeSources(set, N);
    std::vector<afwTable::SourceMatch> matches = afwTable::matchXy(set, radius, true);
    std::vector<afwTable::SourceMatch> refMatches = bruteMatch(set, radius, DistXy());
    compareMatches(matches, refMatches, radius);
}


static void normalizeRaDec(afwTable::SourceCatalog & ss) {
    for (size_t i=0; i<ss.size(); i++) {
        double r,d;
        r = ss[i].getRa().asRadians();
        d = ss[i].getDec().asRadians();
        // wrap Dec over the (north) pole
        if (d > afwGeom::HALFPI) {
            d = afwGeom::PI - d;
            r = r + afwGeom::PI;
        }
        ss[i].set(afwTable::SourceTable::getCoordKey().getRa(), r * afwGeom::radians);
        ss[i].set(afwTable::SourceTable::getCoordKey().getDec(), d * afwGeom::radians);
    }
}


BOOST_AUTO_TEST_CASE(matchNearPole) {

    afwTable::SourceCatalog set1(getGlobalTable());
    afwTable::SourceCatalog set2(getGlobalTable());

    // for each source, add a true match right on top, plus one within range
    // and one outside range in each direction.

    afwGeom::Angle rad = 0.1 * afwGeom::degrees;
    int id1 = 0;
    int id2 = 1000000;
    for (double  j=0.1; j<1; j+=0.1) {
        for (int i=0; i<360; i+=45) {
            afwGeom::Angle ra = i * afwGeom::degrees;
            afwGeom::Angle dec = (90 - j) * afwGeom::degrees;
            afwGeom::Angle ddec1 = rad;
            afwGeom::Angle dra1 = rad / cos(dec);
            afwGeom::Angle ddec2 = 2. * rad;
            afwGeom::Angle dra2 = 2. * rad / cos(dec);

            PTR(afwTable::SourceRecord) src1 = set1.addNew();
            src1->setId(id1);
            id1++;
            src1->set(afwTable::SourceTable::getCoordKey().getRa(), ra);
            src1->set(afwTable::SourceTable::getCoordKey().getDec(), dec);

            // right on top
            PTR(afwTable::SourceRecord) src2 = set2.addNew();
            src2->setId(id2);
            id2++;
            src2->set(afwTable::SourceTable::getCoordKey().getRa(), ra);
            src2->set(afwTable::SourceTable::getCoordKey().getDec(), dec);

            // +Dec 1
            src2 = set2.addNew();
            src2->setId(id2);
            id2++;
            src2->set(afwTable::SourceTable::getCoordKey().getRa(), ra);
            src2->set(afwTable::SourceTable::getCoordKey().getDec(), dec + ddec1);

            // +Dec 2
            src2 = set2.addNew();
            src2->setId(id2);
            id2++;
            src2->set(afwTable::SourceTable::getCoordKey().getRa(), ra);
            src2->set(afwTable::SourceTable::getCoordKey().getDec(), dec + ddec2);

            // -Dec 1
            src2 = set2.addNew();
            src2->setId(id2);
            id2++;
            src2->set(afwTable::SourceTable::getCoordKey().getRa(), ra);
            src2->set(afwTable::SourceTable::getCoordKey().getDec(), dec - ddec1);

            // -Dec 2
            src2 = set2.addNew();
            src2->setId(id2);
            id2++;
            src2->set(afwTable::SourceTable::getCoordKey().getRa(), ra);
            src2->set(afwTable::SourceTable::getCoordKey().getDec(), dec - ddec2);

            // +RA 1
            src2 = set2.addNew();
            src2->setId(id2);
            id2++;
            src2->set(afwTable::SourceTable::getCoordKey().getRa(), ra + dra1);
            src2->set(afwTable::SourceTable::getCoordKey().getDec(), dec);

            // +RA 2
            src2 = set2.addNew();
            src2->setId(id2);
            id2++;
            src2->set(afwTable::SourceTable::getCoordKey().getRa(), ra + dra2);
            src2->set(afwTable::SourceTable::getCoordKey().getDec(), dec);

            // -RA 1
            src2 = set2.addNew();
            src2->setId(id2);
            id2++;
            src2->set(afwTable::SourceTable::getCoordKey().getRa(), ra - dra1);
            src2->set(afwTable::SourceTable::getCoordKey().getDec(), dec);

            // -RA 2
            src2 = set2.addNew();
            src2->setId(id2);
            id2++;
            src2->set(afwTable::SourceTable::getCoordKey().getRa(), ra - dra2);
            src2->set(afwTable::SourceTable::getCoordKey().getDec(), dec);
        }
    }

    normalizeRaDec(set1);
    normalizeRaDec(set2);

    std::vector<afwTable::SourceMatch> matches = afwTable::matchRaDec(set1, set2, rad, false);
    std::vector<afwTable::SourceMatch> refMatches = bruteMatch(set1, set2, rad, DistRaDec()); 
    compareMatches(matches, refMatches, rad);

}
