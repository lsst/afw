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

class WrapperBase {
public:
    virtual ~WrapperBase() {}
    virtual double getX() const = 0;
    virtual double getY() const = 0;
};

class SourceRaDecWrapper : public WrapperBase {
public:
    SourceRaDecWrapper(Source const &s) : _s(s) {}
    double getX() const { return _s.getRaObject(); }
    double getY() const { return _s.getDecObject(); }
private:
    Source const _s;
};

class SourceXyWrapper : public WrapperBase {
public:
    SourceXyWrapper(Source const &s) : _s(s) {}
    double getX() const { return _s.getXAstrom(); }
    double getY() const { return _s.getYAstrom(); }
private:
    Source const &_s;
};
        
class CoordWrapper : public WrapperBase {
public:
    CoordWrapper(afwCoord::Coord const &c) : _c(c) {}
    double getX() const { return _c[0]; }
    double getY() const { return _c[1]; }
private:
    afwCoord::Coord const &_c;
};

    
enum MatchUnit {DEGREES, RADIANS, ARCMIN, ARCSEC, PIXELS};
            
class MatchRange {
public:
    MatchRange(MatchUnit unit);
    virtual ~MatchRange() {};
    virtual double compare(WrapperBase const &s1, WrapperBase const &s2) const = 0;
    
    double computeDistance(WrapperBase const &s1, WrapperBase const &s2) const;
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
    double compare(WrapperBase const &s1, WrapperBase const &s2) const;
    
private:
    double _radius;
};


            
template<typename Src>
struct MatchSet {
    
    typename Src::Ptr first;
    typename Src::Ptr second;
    double distance;

    MatchSet() : first(), second(), distance(0.0) {}
    MatchSet(typename Src::Ptr const & s1, typename Src::Ptr const & s2, double dist)
        : first(s1), second(s2), distance(dist) {}
    ~MatchSet() {}
};


            
template<typename Src>
class MatchResult {
public:
    typedef boost::shared_ptr<std::vector<MatchSet<Src> > > MatchSetPtr;
    typedef boost::shared_ptr<std::vector<Src> > UnmatchedPtr;
    MatchResult(MatchSetPtr matches, UnmatchedPtr unmatched1, UnmatchedPtr unmatched2) :
        _matches(matches), _unmatched1(unmatched1), _unmatched2(unmatched2) { }
    MatchSetPtr getMatched() { return _matches; }
    UnmatchedPtr getUnmatched1() { return _unmatched1; }
    UnmatchedPtr getUnmatched2() { return _unmatched2; }
private:
    MatchSetPtr _matches;
    UnmatchedPtr _unmatched1;
    UnmatchedPtr _unmatched2;
};


            

template<typename Src, typename Wrapper>
MatchResult<Src> matchEngine(std::vector<Src> const &ss1,
                             std::vector<Src> const &ss2,
                             MatchRange const &range);

            
MatchResult<Source> match(std::vector<Source> const &ss1,
                          std::vector<Source> const &ss2,
                          MatchRange const &range);
            
#if 0            
std::vector<MatchSet<afwCoord::Coord> > match(std::vector<afwCoord::Coord> const &c1,
                                              std::vector<afwCoord::Coord> const &c2,
                                              MatchRange const &range);

#endif            

}}} // namespace lsst::afw::detection

#endif // #ifndef LSST_AFW_DETECTION_MATCH_H

