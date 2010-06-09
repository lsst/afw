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

int main() {

    int n = 20;
    double wid = 1024.0;
    double err = 1.0;
    double rLimit = 0.8;

    int nSet = 3;
    std::vector<std::vector<afwDet::Source::Ptr> > ss(nSet, std::vector<afwDet::Source::Ptr>(n));
    std::vector<std::vector<afwCoord::Coord::Ptr> > cc(nSet, std::vector<afwCoord::Coord::Ptr>(n));

    std::vector<double> x0, y0;
    for (int i = 0; i < n; i++) {
        x0.push_back(wid*rand()/RAND_MAX);
        y0.push_back(wid*rand()/RAND_MAX);
    }
    
    for (int iS = 0; iS < nSet; iS++) {
        for (int i = 0; i < n; ++i) {

            double x = x0[i] + err*(1.0*rand()/RAND_MAX - 0.5);
            double y = y0[i] + err*(1.0*rand()/RAND_MAX - 0.5);

            // use Sources
            afwDet::Source::Ptr s(new afwDet::Source);
            s->setXAstrom(x);
            s->setYAstrom(y);
            s->setId(i);
            ss[iS][i] = s;

            // use Coords
            afwCoord::Coord::Ptr c = afwCoord::makeCoord(afwCoord::makeCoordEnum("FK5"),
                                                         x/1000.0+0.001, y/1000.0);
            cc[iS][i] = c;
        }
    }


    /* =============================================================== */
    // try matching

    // call 'match'
    afwDet::MatchCircle circle = afwDet::MatchCircle(rLimit, afwDet::PIXELS);
    afwDet::MatchEllipse ellipse = afwDet::MatchEllipse(rLimit, 0.5*rLimit, M_PI/6.0, afwDet::PIXELS);
    afwDet::MatchResult<afwDet::Source> match = afwDet::match(ss, ellipse);
    
    // see what we got
    std::vector<afwDet::Match<afwDet::Source>::Ptr> matches = match.getMatches();
    for (std::vector<afwDet::Match<afwDet::Source>::Ptr>::iterator it = matches.begin();
         it != matches.end(); ++it) {

        printf("S %1d  ", (*it)->getNullCount());
        for (size_t iS = 0; iS < (*it)->size(); iS++) {
            afwDet::Source::Ptr s = (*it)->getSource(iS);
            double x = s ? s->getXAstrom() : 0;
            double y = s ? s->getYAstrom() : 0;
            double d = (*it)->getDistance(iS);
            
            printf("%7.2f %7.2f %4.2f   ", x, y, d);
        }
        printf("\n");

    }

    printf("\nSource intersection\n");
    afwDet::SourceSet sInter = match.getIntersection();
    for (size_t i = 0; i < sInter.size(); i++) {
        printf("%7.2f %7.2f\n", sInter[i]->getXAstrom(), sInter[i]->getYAstrom());
    }
    printf("\n");
    
    printf("\nSource union\n");
    afwDet::SourceSet sUnion = match.getUnion();
    for (size_t i = 0; i < sUnion.size(); i++) {
        printf("%7.2f %7.2f\n", sUnion[i]->getXAstrom(), sUnion[i]->getYAstrom());
    }
    printf("\n");
    
    
    // do the match ... try using an annulus this time
    // Note:  this will reduce coords in intersection() set.
    
    // *********** must fix all this radians business *******************
    afwDet::MatchAnnulus annulus = afwDet::MatchAnnulus((M_PI/180.0)*0.5*rLimit/1000.0,
                                                        (M_PI/180.0)*rLimit/1000.0,
                                                        afwDet::PIXELS);
    afwDet::MatchResult<afwCoord::Coord> matchC = afwDet::match(cc, annulus);

    std::vector<afwDet::Match<afwCoord::Coord>::Ptr> matchesC = matchC.getMatches();
    for (std::vector<afwDet::Match<afwCoord::Coord>::Ptr>::iterator it = matchesC.begin();
         it != matchesC.end(); ++it) {


        printf("C %1d  ", (*it)->getNullCount());
        for (size_t iS = 0; iS < (*it)->size(); iS++) {
            afwCoord::Coord::Ptr c = (*it)->getSource(iS);
            double x = c ? c->getLongitude(afwCoord::DEGREES) : 0;
            double y = c ? c->getLatitude(afwCoord::DEGREES) : 0;
            double d = (*it)->getDistance(iS);
            
            printf("%7.2f %7.2f %4.2f   ", 1000.0*x, 1000.0*y, 1000.0*d);
        }
        printf("\n");

    }

    printf("\nCoord intersection\n");
    std::vector<afwCoord::Coord::Ptr> cInter = matchC.getIntersection();
    for (size_t i = 0; i < cInter.size(); i++) {
        printf("%7.2f %7.2f\n", 1000.0*cInter[i]->getLongitude(afwCoord::DEGREES),
               1000.0*cInter[i]->getLatitude(afwCoord::DEGREES));
    }

    printf("\nCoord Union\n");
    std::vector<afwCoord::Coord::Ptr> cUnion = matchC.getUnion();
    for (size_t i = 0; i < cUnion.size(); i++) {
        printf("%7.2f %7.2f\n", 1000.0*cUnion[i]->getLongitude(afwCoord::DEGREES),
               1000.0*cUnion[i]->getLatitude(afwCoord::DEGREES));
    }
    
    return 0;
}
