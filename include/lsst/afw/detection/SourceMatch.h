// -*- lsst-c++ -*-
/** @file
  * @author Serge Monkewitz
  * @ingroup afw
  */
#ifndef LSST_AFW_DETECTION_SOURCEMATCH_H
#define LSST_AFW_DETECTION_SOURCEMATCH_H

#include <vector>

#include "boost/tuple/tuple.hpp"

#include "lsst/afw/detection/Source.h"


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

}}} // namespace lsst::afw::detection

#endif // #ifndef LSST_AFW_DETECTION_SOURCEMATCH_H

