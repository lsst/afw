// -*- lsst-c++ -*-
/** @file
  * @ingroup afw
  *
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


/**
 * @brief If s2 is within ellipse a,b,theta centered on s1, return d, otherwise -1
 */
double afwDet::MatchEllipse::compare(
                                     GenericSourceWrapper const &s1,
                                     GenericSourceWrapper const &s2
                                    ) const {
    
    // rotate the coords so the ellipse is along x-axis
    double dx = s2.getX() - s1.getX();
    double dy = s2.getY() - s1.getY();
    double x = dx*std::cos(_theta) + dy*std::sin(_theta);
    double y = -dx*std::sin(_theta) + dy*std::cos(_theta);

    // eq is now  x^2/a^2 + y^2/b^2 = 1  ...  < 1 is within, >1 is outside
    double ee = x*x/(_a*_a) + y*y/(_b*_b);
    
    double d = computeDistance(s1, s2);
    return (ee <= 1) ? d : -1.0;
}






/**
 * @brief Count the null pointers in the Match
 */
template<typename Src>
int afwDet::Match<Src>::getNullCount() {
    int count = 0;
    for (size_t i = 0; i < _sources.size(); i++) {
        if (! _sources[i]) {
            count++;
        }
    }
    return count;
}

/**
 * @brief Get indices for Sources which are non-null
 */
template<typename Src>
std::vector<int> afwDet::Match<Src>::getValidIndices() {
    std::vector<int> v;
    for (size_t i = 0; i < _sources.size(); i++) {
        if (_sources[i]) {
            v.push_back(i);
        }
    }
    return v;
}


/**
 * @brief Constructor for a MatchResult
 */
template<typename Src>
afwDet::MatchResult<Src>::MatchResult(
    std::vector<typename Match<Src>::Ptr> matches ///< list of Match objects in the result
                                     ) :
    _haveUnion(false), _haveIntersection(false),
    _union(std::vector<typename Src::Ptr>(0)),
    _intersection(std::vector<typename Src::Ptr>(0)),
    _matches(matches) {
}



/**
 *
 */
template<typename Src>
std::vector<typename Src::Ptr> afwDet::MatchResult<Src>::getIntersection() {
    
    if (_haveIntersection) {
        return _intersection;
    } else {
        
        for (typename std::vector<typename afwDet::Match<Src>::Ptr>::iterator it = _matches.begin();
             it != _matches.end(); ++it) {
            if ((*it)->getNullCount() == 0) {
                _intersection.push_back( (*it)->getSource(0) );
            }
        }
        
        _haveIntersection = true;
        
        return _intersection;
    }
}

/**
 *
 */
template<typename Src>
std::vector<typename Src::Ptr> afwDet::MatchResult<Src>::getUnion() {

    if (_haveUnion) {
        return _union;
    } else {
        
        for (typename std::vector<typename Match<Src>::Ptr>::iterator it = _matches.begin();
             it != _matches.end(); ++it) {

            // go through until we find the first non-null
            typename Match<Src>::Ptr pMat = *it;
            std::vector<int> validInd = pMat->getValidIndices();
            int ind0 = validInd[0];
            _union.push_back(pMat->getSource(ind0));
            
        }
        _haveUnion = true;
        
        return _union;
    }
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
    
    std::vector<typename Match<Src>::Ptr> matches(0);
    
    std::vector<int> matched2(ss2.size(), 0);

    std::vector<typename Src::Ptr> v(2);
    std::vector<double> d(2);

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

                d[0] = 0.0;
                d[1] = dist;
                typename Match<Src>::Ptr m(new Match<Src>(v, d));
                matches.push_back(m);

                matched2[iS2] += 1;
            }
        }

        // if we didn't find it, create a match object with a NULL pointer
        if (! found) {
            v[0] = *s1;
            v[1] = typename Src::Ptr();
            d[0] = -1;
            d[1] = -1;
            typename Match<Src>::Ptr m(new Match<Src>(v, d));
            matches.push_back(m);
        }
    }

    // now get the ones in ss2 but not in ss1
    for (size_t i = 0; i < matched2.size(); ++i) {
        if (matched2[i] == 0) {
            v[0] = typename Src::Ptr();
            v[1] = ss2[i];
            d[0] = -1;
            d[1] = -1;
            typename Match<Src>::Ptr m(new Match<Src>(v, d));
            matches.push_back(m);
        }
    }
    
    return afwDet::MatchResult<Src>(matches);
}




/**
 *
 */

template<typename Src, typename Wrapper>
afwDet::MatchResult<Src> afwDet::matchChain(
                                            std::vector<std::vector<typename Src::Ptr> > const &ss,
                                            MatchRange const &range
                                           ) {
    int nSet = ss.size();

    std::vector<typename Src::Ptr> nulls(nSet, typename Src::Ptr());
    std::vector<double> zeros(nSet, 0);
    std::vector<int> indices(0);

    std::vector<typename Src::Ptr> ss0 = ss[0];

    std::map<typename Src::Ptr, typename Match<Src>::Ptr > masterList;

    // initialize the masterList with sources from ss[0]
    for (size_t i = 0; i < ss0.size(); i++) {
        masterList[ss0[i]] = typename Match<Src>::Ptr(new Match<Src>(nulls, zeros));
        masterList[ss0[i]]->setSource(0, ss0[i]);
    }
    
    // daisy chain through the sets
    for (int iS = 1; iS < nSet; iS++) {

        MatchResult<Src> result = matchEngine<Src, Wrapper>(ss0, ss[iS], range);
        ss0 = result.getUnion();

        // add the matches to the master list
        for (size_t iM = 0; iM < result.size(); iM++) {
            
            typename Match<Src>::Ptr thisMatch = result[iM];
            int firstValidIndex = (thisMatch->getValidIndices())[0];
            typename Src::Ptr sKey = thisMatch->getSource(firstValidIndex); // use as map key

            // if we don't already have it, create a new Match
            if (masterList.find(sKey) == masterList.end()) {
                masterList[sKey] = typename Match<Src>::Ptr(new Match<Src>(nulls, zeros));
            }

            typename Src::Ptr s0 = thisMatch->getSource(0);
            typename Src::Ptr s1 = thisMatch->getSource(1);
            
            // now put the two sources, in the Match ... in the right place!

            // if we found it in set iS
            if (s1) {
                masterList[sKey]->setSource(iS, s1);
                double dist = 0;
                if (s0) {
                    dist = thisMatch->getDistance(1);
                }
                masterList[sKey]->setDistance(iS, dist);
            }
            
        }
    }

    // now put the map entries into a vector
    std::vector<typename Match<Src>::Ptr> matches;

    // for each object
    for (size_t iM = 0; iM < ss0.size(); iM++) {
        matches.push_back( masterList[ss0[iM]] );
    }
    
    return MatchResult<Src>(matches);
    
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
        return matchChain<Source, SourceXyWrapper>(ss, range);
    } else {
        return matchChain<Source, SourceRaDecWrapper>(ss, range);
    }
}



/**
 *
 */
afwDet::MatchResult<afwCoord::Coord> afwDet::match(
                                                   std::vector<std::vector<afwCoord::Coord::Ptr> > const &ss,
                                                   afwDet::MatchRange const &range
                                                  ) {
    
    return matchChain<afwCoord::Coord, CoordWrapper>(ss, range);
}


template std::vector<afwDet::Source::Ptr> afwDet::MatchResult<afwDet::Source>::getIntersection();
template std::vector<afwCoord::Coord::Ptr> afwDet::MatchResult<afwCoord::Coord>::getIntersection();
