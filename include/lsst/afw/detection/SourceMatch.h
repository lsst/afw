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

typedef boost::tuple<Source::Ptr, Source::Ptr, double> SourceMatch;

std::vector<SourceMatch> matchRaDec(SourceSet const &set1, SourceSet const &set2, double radius);
std::vector<SourceMatch> matchRaDec(SourceSet const &set, double radius, bool symmetric = true);
std::vector<SourceMatch> matchXy(SourceSet const &set1, SourceSet const &set2, double radius);
std::vector<SourceMatch> matchXy(SourceSet const &set, double radius, bool symmetric = true);

}}} // namespace lsst::afw::detection

#endif // #ifndef LSST_AFW_DETECTION_SOURCEMATCH_H

