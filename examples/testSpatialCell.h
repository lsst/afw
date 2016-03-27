/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */
 
#if !defined(TESTSPATIALCELL_H)
#define TESTSPATIALCELL_H
#include "boost/shared_ptr.hpp"
#include "lsst/pex/policy.h"
#include "lsst/afw/math.h"
#include "lsst/afw/geom.h"
#include "lsst/afw/image/Image.h"

/************************************************************************************************************/
/*
 * Test class for SpatialCellImageCandidate
 */
class ExampleCandidate : public lsst::afw::math::SpatialCellMaskedImageCandidate<float> {
public:
    typedef boost::shared_ptr<ExampleCandidate> Ptr;
    typedef float PixelT;
    typedef lsst::afw::image::MaskedImage<PixelT> MaskedImageT;

    ExampleCandidate(float const xCenter, float const yCenter,
                     MaskedImageT::ConstPtr parent, lsst::afw::geom::Box2I bbox);

    lsst::afw::geom::Box2I getBBox() const { return _bbox; }

    double getCandidateRating() const;

    MaskedImageT::ConstPtr getMaskedImage() const;
private:
    ExampleCandidate::MaskedImageT::ConstPtr _parent;
    lsst::afw::geom::Box2I _bbox;
};

/************************************************************************************************************/
/*
 * Class to pass around to all our ExampleCandidates.  All this one does is count acceptable candidates
 */
class ExampleCandidateVisitor : public lsst::afw::math::CandidateVisitor {
public:
    ExampleCandidateVisitor() : lsst::afw::math::CandidateVisitor(), _n(0), _npix(0) {}

    // Called by SpatialCellSet::visitCandidates before visiting any Candidates
    void reset() { _n = _npix = 0; }

    // Called by SpatialCellSet::visitCandidates for each Candidate
    void processCandidate(lsst::afw::math::SpatialCellCandidate *candidate) {
        ++_n;

        lsst::afw::geom::Box2I box = dynamic_cast<ExampleCandidate *>(candidate)->getBBox();
        _npix += box.getArea();
    }

    int getN() const { return _n; }
    int getNPix() const { return _npix; }
private:
    int _n;                         // number of ExampleCandidates
    int _npix;                      // number of pixels in ExampleCandidates's bounding boxes
};

#endif
