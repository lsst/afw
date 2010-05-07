// -*- lsst-c++ -*-
/** @file
  * @author Steve Bickerton
  * @ingroup afw
  */

#include <iostream>
#include "lsst/afw/detection.h"
#include "lsst/afw/detection/Match.h"

namespace afwDet = lsst::afw::detection;

typedef std::vector<afwDet::MatchSet<afwDet::Source> > MatchSet;

int main() {

    int n = 100;
    double wid = 1024.0;
    double err = 1.0;
    double rLimit = 0.8;

    // make some phony sourceSets
    std::vector<afwDet::Source> ss1;
    std::vector<afwDet::Source> ss2;

    for (int i = 0; i < n; ++i) {
        afwDet::Source s1;
        s1.setXAstrom(wid*rand()/RAND_MAX);
        s1.setYAstrom(wid*rand()/RAND_MAX);
        s1.setId(i);
        ss1.push_back(s1);

        afwDet::Source s2;
        s2.setXAstrom(s1.getXAstrom() + err*(1.0*rand()/RAND_MAX - 0.5));
        s2.setYAstrom(s1.getYAstrom() + err*(1.0*rand()/RAND_MAX - 0.5));
        s2.setId(i);
        ss2.push_back(s2);

        //std::cout << s1.getXAstrom() << " " << s2.getXAstrom() << std::endl;
    }


    // try a match
    afwDet::MatchResult<afwDet::Source> match =
        afwDet::match(ss1, ss2, afwDet::MatchCircle(rLimit, afwDet::PIXELS));
    
    for (MatchSet::iterator it = match.getMatched()->begin(); it != match.getMatched()->end(); ++it) {
        std::cout << "Matched: " << it->first->getXAstrom() << " " <<
            it->first->getYAstrom() <<  " " << it->distance << std::endl;
    }
    for (std::vector<afwDet::Source>::iterator it = match.getUnmatched1()->begin();
         it != match.getUnmatched1()->end(); ++it) {
        std::cout << "Unmatched1: " << it->getXAstrom() << " " <<
            it->getYAstrom() << std::endl;
    }
    
    return 0;
}
