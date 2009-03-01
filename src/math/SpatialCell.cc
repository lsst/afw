// -*- lsst-c++ -*-
/**
 * @file
 *
 * @brief Implementation of SpatialCell class
 *
 * @ingroup afw
 */
#include <algorithm>

#include "lsst/afw/image/Mask.h"
#include "lsst/afw/image/Image.h"

#include "lsst/pex/exceptions/Exception.h"
#include "lsst/pex/logging/Trace.h"
#include "lsst/afw/math/SpatialCell.h"

namespace lsst {
namespace afw {
namespace math {

namespace {
    struct CandidatePtrMore : public std::binary_function<SpatialCellCandidate::Ptr,
                                                          SpatialCellCandidate::Ptr,
                                                          bool> {
        bool operator()(SpatialCellCandidate::Ptr a, SpatialCellCandidate::Ptr b) {
            return a->getCandidateRating() > b->getCandidateRating();
        }
    };
}

/// Unique identifier for candidates; useful for preserving current candidate following insertion
int SpatialCellCandidate::_CandidateId = 0;

/**
 * Ctor
 */
SpatialCell::SpatialCell(std::string const& label, ///< string representing "name" of cell
                         lsst::afw::image::BBox const& bbox, ///< Bounding box of cell in overall image
                         CandidateList const& candidateList  ///< list of candidates to represent this cell
                        ) :
    _label(label),
    _bbox(bbox),
    _candidateList(candidateList),
    _currentCandidate()
{
    lsst::pex::logging::TTrace<3>("lsst.afw.math.SpatialCell", 
                                  "Cell %s : created with %d candidates",
                                  this->_label.c_str(), this->_candidateList.size());
    //
    // Sort the list to have the Larger things at the beginning.
    //
    sort(_candidateList.begin(), _candidateList.end(), CandidatePtrMore());

    _currentCandidate = _candidateList.begin();
}

/************************************************************************************************************/
namespace {
    struct equal_to_id : public std::unary_function<SpatialCellCandidate::Ptr, bool> {
        equal_to_id(int id2) : _id2(id2) { }
        bool operator()(SpatialCellCandidate::Ptr const& candidate) const {
            return candidate->getId() == _id2;
        }
    private:
        int _id2;
    };
}
/**
 * Add a candidate to the list, preserving ranking
 *
 * @note doesn't invalid currentCandidate
 */
void SpatialCell::insertCandidate(SpatialCellCandidate::Ptr candidate) {
    CandidateList::iterator pos = std::lower_bound(_candidateList.begin(), _candidateList.end(),
                                                   candidate, CandidatePtrMore());
    int currentId = -1;
    if (_currentCandidate != _candidateList.end()) { // remember current candidate
        currentId = (*_currentCandidate)->getId();    
    }

    _candidateList.insert(pos, candidate);

    if (currentId >= 0) {
        _currentCandidate = std::find_if(_candidateList.begin(), _candidateList.end(), equal_to_id(currentId));        
    } else {
        _currentCandidate = _candidateList.begin();
    }
}
    
/**
 * Select best candidate, returning true if an acceptable candidate in available
 *
 * @note Currently this does *not* use Sdqa objects, but it will.  It
 * only selects the first "good" candidate.
 * 
 * @note For now, we just give up
 */
bool SpatialCell::selectBestCandidate(bool) {
    lsst::pex::logging::TTrace<4>("lsst.ip.diffim.SpatialModelCell.selectBestModel", 
                                  "Cell %s : Locking with no good candidates", this->_label.c_str());

    return false;
}

/**
 * Determine if cell has a usable candidate
 */
bool SpatialCell::isUsable() const {
    return _currentCandidate != _candidateList.end();
}

/**
 * Return the current candidate (if there is one)
 *
 * @throw lsst::pex::exceptions::NotFoundErrorException if no candidate is available
 */
SpatialCellCandidate::Ptr SpatialCell::getCurrentCandidate() {
    if (_currentCandidate == _candidateList.end()) {
        throw LSST_EXCEPT(lsst::pex::exceptions::NotFoundException,
                          (boost::format("SpatialCell %s has no candidate") % _label.c_str()).str());
    }

    return *_currentCandidate;
}
    
/**
 * Move to the previous (or first) candidate in the list
 */
bool SpatialCell::prevCandidate(bool first ///< If true, rewind to the beginning of the list
                               ) {
    if (first) {
        _currentCandidate = _candidateList.begin();
    } else if (_currentCandidate == _candidateList.begin()) {
        /* You are at the beginning; return the best we saw (if any) */
        return selectBestCandidate(true);
    } else {
        if (_candidateList.empty()) {
            return false;
        }

        --_currentCandidate;
    }

    if (!(*_currentCandidate)->instantiate()) { // candidate may need to do something, e.g. build a model
        return prevCandidate();
    }
    
    return true;
}

/**
 * Move to the next candidate in the list
 */
bool SpatialCell::nextCandidate() {
    if (_currentCandidate != _candidateList.end()) {
        ++_currentCandidate;
    }

    if (_currentCandidate == _candidateList.end()) {
        /* You are at the last one; go back and choose the best we saw (if any) */
        return selectBestCandidate(true);
    } else {
        if (!(*_currentCandidate)->instantiate()) { // candidate may need to do something, e.g. build a model
            return nextCandidate();
        }

        return true;
    }
}

}}}
