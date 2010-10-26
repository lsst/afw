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
  * @author Serge Monkewitz
  * @ingroup afw
  */
#ifndef LSST_AFW_DETECTION_SOURCEMATCH_H
#define LSST_AFW_DETECTION_SOURCEMATCH_H

#include <vector>

#include "boost/tuple/tuple.hpp"

#include "lsst/afw/detection/Source.h"
#include "lsst/daf/base/Persistable.h"
#include "lsst/afw/formatters/SourceMatchVectorFormatter.h"

namespace lsst { namespace afw { namespace detection {

struct SourceMatch {
    Source::Ptr first;
    Source::Ptr second;
    double distance;

    SourceMatch() : first(), second(), distance(0.0) {}
    SourceMatch(Source::Ptr const & s1, Source::Ptr const & s2, double dist)
        : first(s1), second(s2), distance(dist) {}
    ~SourceMatch() {}
};

std::vector<SourceMatch> matchRaDec(SourceSet const &set1, SourceSet const &set2, double radius);
std::vector<SourceMatch> matchRaDec(SourceSet const &set, double radius, bool symmetric = true);
std::vector<SourceMatch> matchXy(SourceSet const &set1, SourceSet const &set2, double radius);
std::vector<SourceMatch> matchXy(SourceSet const &set, double radius, bool symmetric = true);


typedef std::vector<SourceMatch> SourceMatchVector;

class PersistableSourceMatchVector : public lsst::daf::base::Persistable {
public:
    typedef boost::shared_ptr<PersistableSourceMatchVector> Ptr;
    PersistableSourceMatchVector() {}
    PersistableSourceMatchVector(SourceMatchVector const & matches)
        : _matches(matches) {}
    ~PersistableSourceMatchVector(){_matches.clear();}
        
    SourceMatchVector getSourceMatches() const {return _matches; }
    void setSourceMatches(SourceMatchVector const & matches) {_matches = matches; }
    
private:

    LSST_PERSIST_FORMATTER(lsst::afw::formatters::SourceMatchVectorFormatter)
    SourceMatchVector _matches;
};

}}} // namespace lsst::afw::detection

#endif // #ifndef LSST_AFW_DETECTION_SOURCEMATCH_H

