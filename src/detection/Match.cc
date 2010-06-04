// -*- lsst-c++ -*-
/** @file
  * @ingroup afw
  */

#include <cmath>

#include "lsst/pex/exceptions.h"
#include "lsst/afw/detection.h"
#include "lsst/afw/detection/Match.h"
#include "lsst/afw/coord/Coord.h"

namespace pexEx = lsst::pex::exceptions;
namespace afwDet = lsst::afw::detection;
namespace afwCoord = lsst::afw::coord;

afwDet::MatchRange::MatchRange(MatchUnit unit) : _unit(unit), _conversion(1.0) {
    
    // now adjust the distance to correct for units of the source
    // we'll leave RADIANS and PIXELS alone as they're the native units for sources
    switch (unit) {
      case RADIANS:
        break;
      case PIXELS:
        break;
      case DEGREES:
        setConversion(afwCoord::degToRad);
        break;
      case ARCMIN:
        setConversion(afwCoord::arcminToRad);
        break;
      case ARCSEC:
        setConversion(afwCoord::arcsecToRad);
        break;
    }
    
}

double afwDet::MatchRange::computeDistance(GenericSourceWrapper const &s1, GenericSourceWrapper const &s2) const {
    double dx = s1.getX() - s2.getX();
    double dy = s1.getY() - s2.getY();
    return std::sqrt(dx*dx + dy*dy);
}
    




            
double afwDet::MatchCircle::compare(GenericSourceWrapper const &s1, GenericSourceWrapper const &s2) const {
    double d = computeDistance(s1, s2);
    return (d < _radius) ? d : -1.0;
}

            
template<typename Src, typename Wrapper>
afwDet::MatchResult<Src> afwDet::matchEngine(std::vector<typename Src::Ptr> const &ss1,
                                             std::vector<typename Src::Ptr> const &ss2,
                                             MatchRange const &range) {
    
    std::vector<Match<Src> > matches;
    
    //boost::shared_ptr<std::vector<Src> > sourceUnion;
    //boost::shared_ptr<std::vector<Match::Ptr> > summary;

    std::vector<int> matched2(ss2.size(), 0);

    std::vector<typename Src::Ptr> v(2);
    std::vector<double> d(2);

    typename Src::Ptr nullSource(new Src);
    Wrapper(nullSource).setY(-1);
    
    // do the slow way first
    for (typename std::vector<typename Src::Ptr>::const_iterator s1 = ss1.begin(); s1 != ss1.end(); ++s1) {
        bool found = false;
        int iS2 = 0;

        for (typename std::vector<typename Src::Ptr>::const_iterator s2 = ss2.begin(); s2 != ss2.end(); ++s2, ++iS2) {
            double dist = range.compare(Wrapper(*s1), Wrapper(*s2));

            // if we found it, create a Match object
            if (dist >= 0) {
                found = true;

                v[0] = *s1;
                v[1] = *s2;

                d[0] = dist;
                d[1] = dist;
                matches.push_back(Match<Src>(v, d, 0));
                matched2[iS2] += 1;
            }
        }

        // if we didn't find it, create a match object with a NULL pointer
        if (! found) {
            v[0] = *s1;
            v[1] = nullSource;
            d[0] = -1;
            d[1] = -1;
            matches.push_back(Match<Src>(v, d, 1));
        }
    }

    // now get the ones in ss2 but not in ss1
    for (unsigned int i = 0; i < matched2.size(); ++i) {
        if (matched2[i] == 0) {
            v[0] = nullSource;
            v[1] = ss2[i];
            d[0] = -1;
            d[1] = -1;
            matches.push_back(Match<Src>(v, d, 1));
        }
    }
    
    return afwDet::MatchResult<Src>(matches);
}


afwDet::MatchResult<afwDet::Source> afwDet::match(std::vector<std::vector<afwDet::Source::Ptr> > const &ss,
                                                  afwDet::MatchRange const &range) {
    
    if ( range.getUnit() == PIXELS ) {
        return afwDet::matchEngine<Source, SourceXyWrapper>(ss[0], ss[1], range);
    } else {
        return afwDet::matchEngine<Source, SourceRaDecWrapper>(ss[0], ss[1], range);
    }
}


afwDet::MatchResult<afwCoord::Coord> afwDet::match(std::vector<std::vector<afwCoord::Coord::Ptr> > const &ss,
                                                  afwDet::MatchRange const &range) {
    
    return afwDet::matchEngine<afwCoord::Coord, CoordWrapper>(ss[0], ss[1], range);
}


#if 0            
std::vector<MatchSet<afwCoord::Coord> > match(std::vector<afwCoord::Coord> const &c1,
                                              std::vector<afwCoord::Coord> const &c2,
                                              MatchRange const &range) {

    if ( range.getUnit() == PIXELS ) {
        throw LSST_EXCEPT(pexEx::InvalidParameterException,
                          "Coord objects contain only celestial coordinates.  Can't match PIXELS.");
    } else {
        return matchEngine<afwCoord::Coord, CoordWrapper>(c1, c2, range);
    }
}
#endif            


//template afwDet::MatchResult<afwDet::Source> afwDet::match(std::vector<std::vector<afwDet::Source::Ptr> > const &ss, afwDet::MatchRange const &range);

//template afwDet::MatchResult<afwCoord::Coord> afwDet::match(std::vector<std::vector<afwCoord::Coord::Ptr> > const &ss, afwDet::MatchRange const &range);
