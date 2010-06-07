// -*- lsst-c++ -*-
/** @file
  * @author Steve Bickerton
  * @ingroup afw
  */

#include <iostream>
#include "lsst/afw/detection.h"
#include "lsst/afw/coord.h"
#include "lsst/afw/detection/Match.h"

namespace afwDet = lsst::afw::detection;
namespace afwCoord = lsst::afw::coord;

//typedef std::vector<afwDet::Match<afwDet::Source> > MatchSet;

int main() {

    int n = 20;
    double wid = 1024.0;
    double err = 1.0;
    double rLimit = 0.5;

    // make some phony sourceSets
    std::vector<afwDet::Source::Ptr> ss1;
    std::vector<afwDet::Source::Ptr> ss2;

    // make some phony coordSets
    std::vector<afwCoord::Coord::Ptr> cc1;
    std::vector<afwCoord::Coord::Ptr> cc2;
    
    for (int i = 0; i < n; ++i) {

        double x1 = wid*rand()/RAND_MAX;
        double y1 = wid*rand()/RAND_MAX;
        double x2 = x1 + err*(1.0*rand()/RAND_MAX - 0.5);
        double y2 = y1 + err*(1.0*rand()/RAND_MAX - 0.5);
        
        // use Sources
        afwDet::Source::Ptr s1(new afwDet::Source);
        s1->setXAstrom(x1);
        s1->setYAstrom(y1);
        s1->setId(i);
        ss1.push_back(s1);

        afwDet::Source::Ptr s2(new afwDet::Source);
        s2->setXAstrom(x2);
        s2->setYAstrom(y2);
        s2->setId(i);
        ss2.push_back(s2);

        // use Coords
        afwCoord::Coord::Ptr c1 = afwCoord::makeCoord(afwCoord::makeCoordEnum("FK5"),
                                                      x1/1000.0, y1/1000.0);
        cc1.push_back(c1);

        afwCoord::Coord::Ptr c2 = afwCoord::makeCoord(afwCoord::makeCoordEnum("FK5"),
                                                      x2/1000.0, y2/1000.0);
        cc2.push_back(c2);

    }


    /* =============================================================== */
    // try a match for Sources

    // put the source sets in a vector
    std::vector<std::vector<afwDet::Source::Ptr> > ss;
    ss.push_back(ss1);
    ss.push_back(ss2);

    // call 'match'
    afwDet::MatchResult<afwDet::Source> match =
        afwDet::match(ss, afwDet::MatchCircle(rLimit, afwDet::PIXELS));


    // see what we got
    std::vector<afwDet::Match<afwDet::Source> > matches = match.getMatches();
    for (std::vector<afwDet::Match<afwDet::Source> >::iterator it = matches.begin();
         it != matches.end(); ++it) {

        // If this match worked
        if ((*it)[0]->getYAstrom() != -1 && (*it)[1]->getYAstrom() != -1) {
            std::cout << "S_Matched: " <<
                (*it)[0]->getXAstrom() << " " <<
                (*it)[0]->getYAstrom() <<  " " <<
                (*it)[1]->getXAstrom() << " " <<
                (*it)[1]->getYAstrom() <<  " " <<
                it->getDistance(0) << std::endl;

            // if this match contains an s1 object unmatched in s2
        } else if ( (*it)[0]->getYAstrom() != -1 ) {
            std::cout << "S_Match_1: " <<
                (*it)[0]->getXAstrom() << " " <<
                (*it)[0]->getYAstrom() <<  " " <<
                0.0 << " " <<
                0.0 << " " <<
                0 << std::endl;
            
            // if this match contains an s2 object unmatched in s1
        } else if ( (*it)[1]->getYAstrom() != -1 ) {
            std::cout << "S_Match_2: " <<
                0.0 << " " <<
                0.0 << " " <<
                (*it)[1]->getXAstrom() << " " <<
                (*it)[1]->getYAstrom() <<  " " <<
                0 << std::endl;
        }
        
    }


    /* =================================================================== */
    // try with a Coord
    std::vector<std::vector<afwCoord::Coord::Ptr> > cc;
    cc.push_back(cc1);
    cc.push_back(cc2);

    // very sleezy, must fix this
    // I can't return a NULL Coord, so I used latitude=-1 to denote 'unmatched'
    double tol = 1.0e-9;

    // do the match ... try using an annulus this time
    // *********** must fix all this radians business *******************
    afwDet::MatchResult<afwCoord::Coord> matchC =
        afwDet::match(cc, afwDet::MatchAnnulus((M_PI/180.0)*0.5*rLimit/1000.0,
                                               (M_PI/180.0)*rLimit/1000.0,
                                               afwDet::PIXELS));

    std::vector<afwDet::Match<afwCoord::Coord> > matchesC = matchC.getMatches();
    for (std::vector<afwDet::Match<afwCoord::Coord> >::iterator it = matchesC.begin();
         it != matchesC.end(); ++it) {

        // if this match worked
        if ( fabs((*it)[0]->getLatitude(afwCoord::RADIANS) + 1) > tol &&
             fabs((*it)[1]->getLatitude(afwCoord::RADIANS) + 1) > tol) {
            std::cout << "C_Matched: " <<
                (*it)[0]->getLongitude(afwCoord::RADIANS) << " " <<
                (*it)[0]->getLatitude(afwCoord::RADIANS) <<  " " <<
                (*it)[1]->getLongitude(afwCoord::RADIANS) << " " <<
                (*it)[1]->getLatitude(afwCoord::RADIANS) <<  " " <<
                it->getDistance(1) << std::endl;


            // if this match contains a c1 object unmatched in c2
        } else if ( fabs((*it)[0]->getLatitude(afwCoord::RADIANS) + 1) > tol) {
            std::cout << "C_Match_1: " <<
                (*it)[0]->getLongitude(afwCoord::RADIANS) << " " <<
                (*it)[0]->getLatitude(afwCoord::RADIANS) <<  " " <<
                0.0 << " " <<
                0.0 << " " <<
                0 << std::endl;


            // if this match contains a c2 object unmatched in c1
        } else if ( fabs((*it)[1]->getLatitude(afwCoord::RADIANS) + 1) > tol) {
            std::cout << "C_Match_2: " <<
                0.0 << " " <<
                0.0 << " " <<
                (*it)[1]->getLongitude(afwCoord::RADIANS) << " " <<
                (*it)[1]->getLatitude(afwCoord::RADIANS) <<  " " <<
                0 << std::endl;
        }
    }

    
    return 0;
}
