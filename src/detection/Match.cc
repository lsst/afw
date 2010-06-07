// -*- lsst-c++ -*-
/** @file
  * @ingroup afw
  *
  * @todo Return NULL Sources for unmatched ones
  */

#include <cmath>

#include "lsst/pex/exceptions.h"
#include "lsst/afw/detection.h"
#include "lsst/afw/detection/Match.h"
#include "lsst/afw/coord/Coord.h"

namespace pexEx = lsst::pex::exceptions;
namespace afwDet = lsst::afw::detection;
namespace afwCoord = lsst::afw::coord;




/* =============================================================
 * The MatchRange methods 
 * ============================================================= */


/**
 *
 *
 */

afwDet::MatchRange::MatchRange(
                               MatchUnit unit
                              ) :
    _unit(unit), _conversion(1.0) {
    
    // now adjust the distance to correct for units of the source
    // we'll leave RADIANS and PIXELS alone as they're the native units for sources and coords
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

/**
 *
 */
double afwDet::MatchRange::computeDistance(
                                           GenericSourceWrapper const &s1,
                                           GenericSourceWrapper const &s2
                                          ) const {
    double dx = s1.getX() - s2.getX();
    double dy = s1.getY() - s2.getY();
    return std::sqrt(dx*dx + dy*dy);
}
    




/* =============================================================
 * Specific comparisons which inherit from MatchRange
 * ============================================================= */


/**
 * @brief If s1 is within _radius of s2, return d, otherwise -1
 */
double afwDet::MatchCircle::compare(
                                    GenericSourceWrapper const &s1,
                                    GenericSourceWrapper const &s2
                                   ) const {
    double d = computeDistance(s1, s2);
    return (d < _radius) ? d : -1.0;
}


/**
 * @brief If s2 is  between rInner,rOuter of s1, return d, otherwise -1
 */
double afwDet::MatchAnnulus::compare(
                                     GenericSourceWrapper const &s1,
                                     GenericSourceWrapper const &s2
                                    ) const {
    double d = computeDistance(s1, s2);
    return (d <= _rOuter && d >= _rInner) ? d : -1.0;
}






/* =============================================================
 * match engines 
 * ============================================================= */


/**
 * @note This matchEngine (Order n^2, very slow) is a place holder.  It will be replaced with Serge's
 *       existing match code when/if this design is worth pursuing.
 */
template<typename Src, typename Wrapper>
afwDet::MatchResult<Src> afwDet::matchEngine(
                                             std::vector<typename Src::Ptr> const &ss1,
                                             std::vector<typename Src::Ptr> const &ss2,
                                             MatchRange const &range
                                            ) {
    
    std::vector<Match<Src> > matches;
    
    std::vector<int> matched2(ss2.size(), 0);

    std::vector<typename Src::Ptr> v(2);
    std::vector<double> d(2);

    // ***** fix this ***** 
    // This should be NULL to denote 'no match', but I don't know how to return NULL for a Src::Ptr
    typename Src::Ptr nullSource(new Src);
    Wrapper(nullSource).setY(-1);
    
    // do the slow way first
    for (typename std::vector<typename Src::Ptr>::const_iterator s1 = ss1.begin(); s1 != ss1.end(); ++s1) {
        bool found = false;
        int iS2 = 0;

        for (typename std::vector<typename Src::Ptr>::const_iterator s2 = ss2.begin();
             s2 != ss2.end(); ++s2, ++iS2) {
            
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





/* =============================================================
 * match functions for each type we accept (Source, Coord)
 * ============================================================= */



/**
 *
 */
afwDet::MatchResult<afwDet::Source> afwDet::match(
                                                  std::vector<std::vector<afwDet::Source::Ptr> > const &ss,
                                                  afwDet::MatchRange const &range
                                                 ) {
    
    if ( range.getUnit() == PIXELS ) {
        return afwDet::matchEngine<Source, SourceXyWrapper>(ss[0], ss[1], range);
    } else {
        return afwDet::matchEngine<Source, SourceRaDecWrapper>(ss[0], ss[1], range);
    }
}



/**
 *
 */
afwDet::MatchResult<afwCoord::Coord> afwDet::match(
                                                   std::vector<std::vector<afwCoord::Coord::Ptr> > const &ss,
                                                   afwDet::MatchRange const &range
                                                  ) {
    
    return afwDet::matchEngine<afwCoord::Coord, CoordWrapper>(ss[0], ss[1], range);
}


