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

#include "lsst/afw/table/BaseRecord.h"
#include "lsst/afw/table/Vector.h"
#include "lsst/afw/geom/Angle.h"

namespace lsst { namespace afw { namespace table {

/**
 *  @brief Lightweight representation of a geometric match between two records.
 *
 *  This is a template so it can hold derived record classes without a lot of 
 *  casting and properly use Angle for the distance when we do spherical coordinate
 *  matches.
 */
template <typename Record1, typename Record2, typename Distance>
struct Match {
    PTR(Record1) first;
    PTR(Record2) second;
    double distance;

    Match() : first(), second(), distance(0.0) {}
    
    Match(PTR(Record1) const & r1, PTR(Record2) const & r2, Distance dist)
        : first(r1), second(r2), distance(dist) {}

};

typedef Match<BaseRecord,BaseRecord,double> CartesianBaseMatch;
typedef Match<BaseRecord,BaseRecord,Angle> AngularBaseMatch;

typedef std::vector<CartesianBaseMatch> CartesianBaseMatchVector;
typedef std::vector<AngularBaseMatch> AngularBaseMatchVector;

/** 
 *  Compute all tuples (s1,s2,d) where s1 belings to @a set1, s2 belongs to @a set2 and
  * d, the distance between s1 and s2, is at most @a radius. If set1 and
  * set2 are identical, then this call is equivalent to @c matchRaDec(set1,radius,true).
  * The match is performed in ra, dec space.
  *
  * @param[in] set1     first set of sources
  * @param[in] set2     second set of sources
  * @param[in] radius   match radius
  * @param[in] closest  if true then just return the closest match
  */
CartesianBaseMatchVector matchXy(
    BaseVector const & v1, Key< Point<double> > const & key1,
    BaseVector const & v2, Key< Point<double> > const & key2,
    double dist, bool closest=true
);

CartesianBaseMatchVector matchXy(
    BaseVector const & v, Key< Point<double> > const & key,
    double dist, bool symmetric=true
);

AngularBaseMatchVector matchRaDec(
    BaseVector const & v1, Key<Coord> const & key1,
    BaseVector const & v2, Key<Coord> const & key2,
    Angle dist, bool closest=true
);

AngularBaseMatchVector matchRaDec(
    BaseVector const & v, Key<Coord> const & key,
    Angle dist, bool symmetric=true
);

}}} // namespace lsst::afw::table

#endif // #ifndef LSST_AFW_TABLE_MATCH_H
