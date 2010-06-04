// -*- lsst-c++ -*-
/** @file
  * @author Steve Bickerton
  * @ingroup afw
  */
#ifndef LSST_AFW_DETECTION_MATCH_H
#define LSST_AFW_DETECTION_MATCH_H

#include <vector>

#include "boost/tuple/tuple.hpp"

#include "lsst/afw/detection/Source.h"

#include "lsst/afw/coord.h"

namespace afwCoord = lsst::afw::coord;

namespace lsst { namespace afw { namespace detection {

class GenericSourceWrapper {
public:
    virtual ~GenericSourceWrapper() {}
    virtual double getX() const = 0;
    virtual double getY() const = 0;
    virtual void setX(double x) const = 0;
    virtual void setY(double y) const = 0;
};

class SourceRaDecWrapper : public GenericSourceWrapper {
public:
    SourceRaDecWrapper(Source::Ptr s) : _s(s) {}
    double getX() const { return _s->getRaObject(); }
    double getY() const { return _s->getDecObject(); }
    void setX(double x) const { _s->setRaObject(x); }
    void setY(double y) const { _s->setDecObject(y); }
private:
    Source::Ptr _s;
};

class SourceXyWrapper : public GenericSourceWrapper {
public:
    SourceXyWrapper(Source::Ptr s) : _s(s) {}
    double getX() const { return _s->getXAstrom(); }
    double getY() const { return _s->getYAstrom(); }
    void setX(double x) const { _s->setXAstrom(x); }
    void setY(double y) const { _s->setYAstrom(y); }
private:
    Source::Ptr _s;
};
        
class CoordWrapper : public GenericSourceWrapper {
public:
    CoordWrapper(afwCoord::Coord::Ptr c) : _c(c) {}
    double getX() const { return (*_c)[0]; }
    double getY() const { return (*_c)[1]; }
    void setX(double x) const { _c->reset( (180.0/M_PI)*x, (180.0/M_PI)*(*_c)[1], _c->getEpoch() ); }
    void setY(double y) const { _c->reset( (180.0/M_PI)*(*_c)[0], (180.0/M_PI)*y, _c->getEpoch() ); }
private:
    afwCoord::Coord::Ptr _c;
};

    
enum MatchUnit {DEGREES, RADIANS, ARCMIN, ARCSEC, PIXELS};
            
class MatchRange {
public:
    MatchRange(MatchUnit unit);
    virtual ~MatchRange() {};
    virtual double compare(GenericSourceWrapper const &s1, GenericSourceWrapper const &s2) const = 0;
    
    double computeDistance(GenericSourceWrapper const &s1, GenericSourceWrapper const &s2) const;
    double getConversion() const { return _conversion; }
    void setConversion(double conversion) { _conversion = conversion; }

    MatchUnit getUnit() const { return _unit; }
    
private:
    MatchUnit _unit;
    double _conversion;
};


            
class MatchCircle : public MatchRange {
public:
    MatchCircle(double radius, MatchUnit unit) : MatchRange(unit), _radius(radius) {
        _radius *= getConversion();
    }
    double compare(GenericSourceWrapper const &s1, GenericSourceWrapper const &s2) const;
    
private:
    double _radius;
};



/** 
 *
 *
 */

template <typename Src>
class Match {
public:

    typedef boost::shared_ptr<Match<Src> > Ptr;
    
    Match(std::vector<typename Src::Ptr> sources,
          std::vector<double> distance, int nullCount) :
        _nullCount(nullCount), _sources(sources), _distance(distance) {}
    
    unsigned int getLength() { return _sources.size(); }
    typename Src::Ptr operator[](unsigned int i) { return _sources[i]; }

    double getDistance(unsigned int i) { return _distance[i]; }

    int getNullCount() { return _nullCount; }
private:
    int _nullCount;
    std::vector<typename Src::Ptr> _sources;
    std::vector<double> _distance;
};

            

template<typename Src>
class MatchResult {
public:
    
    MatchResult(
                std::vector<Match<Src> > matches
               ) :
        _matches(matches) {}
    
    std::vector<Match<Src> > getIntersection();
    //std::vector<Src> getUnmatched(unsigned int i) { return _unmatched[i]; }
    //std::vector<Src> getUnion() { return _union; }
    std::vector<Match<Src> > getMatches() { return _matches; }

private:
    typename std::vector<Match<Src> > _matches;
};


            
/**
 * @note We need to have overloaded match() functions to handle differences between Source and Coord
 *       ... can't just template and instantiate different ones.
 */
MatchResult<Source> match(std::vector<std::vector<Source::Ptr> > const &ss,
                          MatchRange const &range);


/**
 *
 */
MatchResult<afwCoord::Coord> match(std::vector<std::vector<afwCoord::Coord::Ptr> > const &ss,
                                   MatchRange const &range);
            
template<typename Src, typename Wrapper>
MatchResult<Src> matchEngine(std::vector<typename Src::Ptr> const &ss1,
                             std::vector<typename Src::Ptr> const &ss2,
                             MatchRange const &range);

            
#if 0
            MatchResult<Source> match(std::vector<Source::Ptr> const &ss1,
                          std::vector<Source::ptr> const &ss2,
                          MatchRange const &range);
            

std::vector<MatchSet<afwCoord::Coord> > match(std::vector<afwCoord::Coord> const &c1,
                                              std::vector<afwCoord::Coord> const &c2,
                                              MatchRange const &range);

#endif            

}}} // namespace lsst::afw::detection

#endif // #ifndef LSST_AFW_DETECTION_MATCH_H

