// -*- lsst-c++ -*-
#ifndef LSST_AFW_DETECTION_MATCH_H
#define LSST_AFW_DETECTION_MATCH_H
/** @file
  * @author Steve Bickerton
  * @ingroup afw
  *
  * @todo Making many copies.  Use pointers and refs when possible.
  * @todo use size_t for size() comparisons
  * @todo find a better way to deal with validindices
  */

#include <vector>

#include "boost/tuple/tuple.hpp"
#include "lsst/afw/detection/Source.h"
#include "lsst/afw/coord.h"

namespace afwCoord = lsst::afw::coord;


namespace lsst {
namespace afw {
namespace detection {


/* ==============================================================
 *
 * Wrappers/Adapters for the coordinate types we accept (Source, Coord)
 *
 * We need to access x,y but different containers have different names for them:
 *     eg. Source.getXAstrom() vs Coord.getLongitude()
 * We therefore wrap whatever we get so that we can call getX() or getY()
 *
 * ============================================================== */


/**
 * @brief A base class for a wrapper/adapter.
 *
 * We'll pass Source/Coord objects as objects of this type, but they'll actually be derived from it.
 */
class GenericSourceWrapper {
public:
    virtual ~GenericSourceWrapper() {}
    virtual double getX() const = 0;
    virtual double getY() const = 0;
    virtual void setX(double x) const = 0;
    virtual void setY(double y) const = 0;
};


/**
 * @brief Wrap a Source so getX/Y() access the getRaObject/DecObject() methods.
 */
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

            
/**
 * @brief Wrap a Source so getX/Y() access the getXAstrom/YAstrom() methods.
 */
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

/**
 * @brief Wrap a Coord so getX/Y() access the operator[] method to retrieve values in radians.
 */            
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

            
/**
 * @brief Base class for the match style (within r, within ellipse, etc)
 *
 */    
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



/**
 * @brief A class to define a circular match (ie. within radius r)
 *
 */
        
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
 * @brief A class to define an annular match (ie. between rInner and rOuter)
 *
 */
class MatchAnnulus : public MatchRange {
public:
    MatchAnnulus(double rInner, double rOuter, MatchUnit unit) :
        MatchRange(unit), _rInner(rInner), _rOuter(rOuter) {
        _rInner *= getConversion();
        _rOuter *= getConversion();
    }
    double compare(GenericSourceWrapper const &s1, GenericSourceWrapper const &s2) const;
    
private:
    double _rInner;
    double _rOuter;
};


/**
 * @brief A class to define an annular match (ie. between rInner and rOuter)
 *
 */
class MatchEllipse : public MatchRange {
public:
    MatchEllipse(double a, double b, double theta, MatchUnit unit) :
        MatchRange(unit), _a(a), _b(b), _theta(theta) {
        _a *= getConversion();
        _b *= getConversion();
    }
    double compare(GenericSourceWrapper const &s1, GenericSourceWrapper const &s2) const;
    
private:
    double _a;
    double _b;
    double _theta;
};

    
    

/** 
 * @brief A container to hold matched objects from N sets (Source, Coord, etc)
 *
 */

template <typename Src>
class Match {
public:

    typedef boost::shared_ptr<Match<Src> > Ptr;

     Match(std::vector<typename Src::Ptr> sources, std::vector<double> distances) :
        _sources(sources), _distances(distances) {}
    
    size_t size() { return _sources.size(); }
    
    typename Src::Ptr getSource(int i) { return _sources[i]; }
    void setSource(int i, typename Src::Ptr s) { _sources[i] = s; }
    typename Src::Ptr operator[](int i) { return _sources[i]; }

    double getDistance(int i) { return _distances[i]; }
    void setDistance(int i, double d) { _distances[i] = d; }
    
    int getNullCount();
    std::vector<int> getValidIndices();
    
private:
    std::vector<typename Src::Ptr> _sources;
    std::vector<double> _distances;
};

            

/**
 * @brief A class to contain the results of a match.  A full match will be performed and stored here
 *        Specific subsets (union, intersection, complement, etc) can get be computed as needed
 *        through methods in this class.
 */
template<typename Src>
class MatchResult {
public:
    
    MatchResult(std::vector<typename Match<Src>::Ptr> matches);
    
    typename Match<Src>::Ptr operator[](int i) { return _matches[i]; }
    size_t size() { return _matches.size(); }
    
    std::vector<typename Src::Ptr> getIntersection();
    std::vector<typename Src::Ptr> getUnion();
    std::vector<typename Match<Src>::Ptr> getMatches() { return _matches; }

private:
    bool _haveUnion;
    bool _haveIntersection;
    std::vector<typename Src::Ptr> _union;
    std::vector<typename Src::Ptr> _intersection;
    typename std::vector<typename Match<Src>::Ptr> _matches;
};


            
/**
 * @brief The function to perform a match on a vector of SourceSets
 *
 * @note We need to have overloaded match() functions to handle differences between Source and Coord
 *       ... can't just template and instantiate different ones.
 */
MatchResult<Source> match(std::vector<std::vector<Source::Ptr> > const &ss,
                          MatchRange const &range);


/**
 * @brief The function to perform a match on a vector of CoordSets
 */
MatchResult<afwCoord::Coord> match(std::vector<std::vector<afwCoord::Coord::Ptr> > const &ss,
                                   MatchRange const &range);



/**
 * @brief The brains of the operation.  An engine to compute matches between two sets
 *
 * @note To handle more than two sets, this will be called repeatedly in a chain
 */
template<typename Src, typename Wrapper>
MatchResult<Src> matchEngine(std::vector<typename Src::Ptr> const &ss1,
                             std::vector<typename Src::Ptr> const &ss2,
                             MatchRange const &range);


/**
 * @brief Function to match an arbitrary number of source sets
 *
 */
template<typename Src, typename Wrapper>
MatchResult<Src> matchChain(std::vector<std::vector<typename Src::Ptr> > const &ss,
                            MatchRange const &range);
            
            
}}} // namespace lsst::afw::detection

#endif // #ifndef LSST_AFW_DETECTION_MATCH_H

