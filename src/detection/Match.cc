// -*- lsst-c++ -*-
/** @file
  * @ingroup afw
  */

#include <cmath>

#include "lsst/pex/exceptions.h"
#include "lsst/afw/detection.h"
#include "lsst/afw/detection/Match.h"

namespace pexEx = lsst::pex::exceptions;
namespace afwDet = lsst::afw::detection;


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

double afwDet::MatchRange::computeDistance(WrapperBase const &s1, WrapperBase const &s2) const {
    double dx = s1.getX() - s2.getX();
    double dy = s1.getY() - s2.getY();
    return std::sqrt(dx*dx + dy*dy);
}
    




            
double afwDet::MatchCircle::compare(WrapperBase const &s1, WrapperBase const &s2) const {
    double d = computeDistance(s1, s2);
    return (d < _radius) ? d : -1.0;
}

            
template<typename Src, typename Wrapper>
afwDet::MatchResult<Src> afwDet::matchEngine(std::vector<Src> const &ss1,
                                             std::vector<Src> const &ss2,
                                             MatchRange const &range) {
    
    typedef std::vector<MatchSet<Src> > MatchSetV;
    typedef std::vector<Src> UnmatchedV;
    typename boost::shared_ptr<MatchSetV> matches(new MatchSetV());
    typename boost::shared_ptr<UnmatchedV> unmatched1(new UnmatchedV());
    typename boost::shared_ptr<UnmatchedV> unmatched2(new UnmatchedV());

    MatchResult<Src> result(matches, unmatched1, unmatched2);
    
    std::vector<int> matched2(ss2.size(), 0);
    
    // do the slow way first
    for (typename std::vector<Src>::const_iterator s1 = ss1.begin(); s1 != ss1.end(); ++s1) {
        bool found = false;
        typename Src::Ptr ps1(new Src(*s1));
        int iS2 = 0;
        for (typename std::vector<Src>::const_iterator s2 = ss2.begin(); s2 != ss2.end(); ++s2, ++iS2) {
            double dist = range.compare(Wrapper(*s1), Wrapper(*s2));
            if (dist >= 0) {
                found = true;
                typename Src::Ptr ps2(new Src(*s2));
                matches->push_back(MatchSet<Src>(ps1, ps2, dist));
                matched2[iS2] += 1;
            }
        }
        if (found) {
            unmatched1->push_back(*s1);
        }
    }

    for (unsigned int i = 0; i < matched2.size(); ++i) {
        if (matched2[i] == 0) {
            (*unmatched2)[i] = ss2[i];
        }
    }
    
    return result;
}

afwDet::MatchResult<afwDet::Source> afwDet::match(std::vector<afwDet::Source> const &ss1,
                                                  std::vector<afwDet::Source> const &ss2,
                                                  afwDet::MatchRange const &range) {
    if ( range.getUnit() == PIXELS ) {
        return matchEngine<Source, SourceXyWrapper>(ss1, ss2, range);
    } else {
        return matchEngine<Source, SourceRaDecWrapper>(ss1, ss2, range);
    }
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
