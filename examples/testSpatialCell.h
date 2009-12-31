#if !defined(SPATIALCELL_H)
#define SPATIALCELL_H
#include "boost/shared_ptr.hpp"
#include "lsst/pex/policy.h"
#include "lsst/afw/math.h"

/*
 * Test class for SpatialCellCandidate
 */
class TestCandidate : public lsst::afw::math::SpatialCellCandidate {
public:
    typedef boost::shared_ptr<TestCandidate> Ptr;
    TestCandidate(float const xCenter, float const yCenter, float const flux);
    double getCandidateRating() const;
private:
    double _flux;
};

/// A class to pass around to all our TestCandidates
class TestCandidateVisitor : public lsst::afw::math::CandidateVisitor {
public:
    TestCandidateVisitor() : lsst::afw::math::CandidateVisitor(), _n(0) {}

    // Called by SpatialCellSet::visitCandidates before visiting any Candidates
    void reset() { _n = 0; }

    // Called by SpatialCellSet::visitCandidates for each Candidate
    void processCandidate(lsst::afw::math::SpatialCellCandidate *candidate) {
        ++_n;
    }

    int getN() const { return _n; }
private:
    int _n;                         // number of TestCandidates
};

/************************************************************************************************************/
/*
 * Test class for SpatialCellImageCandidate
 */
class TestImageCandidate : public lsst::afw::math::SpatialCellImageCandidate<lsst::afw::image::Image<float> > {
public:
    typedef boost::shared_ptr<TestImageCandidate> Ptr;
    typedef lsst::afw::image::Image<float> ImageT;

    TestImageCandidate(float const xCenter, float const yCenter, float const flux);

    double getCandidateRating() const;

    ImageT::ConstPtr getImage() const;
private:
    double _flux;
};

#endif
