// -*- lsst-c++ -*-
/**
 * @file
 *
 * @brief Implementation of SpatialCell class
 *
 * @ingroup afw
 */
#include <algorithm>

#include "lsst/afw/image/ImageUtils.h"
#include "lsst/afw/image/Utils.h"

#include "lsst/pex/exceptions/Exception.h"
#include "lsst/pex/logging/Trace.h"
#include "lsst/afw/math/SpatialCell.h"

namespace image = lsst::afw::image;

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
                         image::BBox const& bbox,  ///< Bounding box of cell in overall image
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
        if (_candidateList.empty()) {
            return false;
        }

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

    if ((*_currentCandidate)->getStatus() == SpatialCellCandidate::BAD ||
        !(*_currentCandidate)->instantiate()) { // candidate may need to do something, e.g. build a model
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
        if ((*_currentCandidate)->getStatus() == SpatialCellCandidate::BAD ||
            !(*_currentCandidate)->instantiate()) { // candidate may need to do something, e.g. build a model
            return nextCandidate();
        }

        return true;
    }
}

/************************************************************************************************************/

/**
 * Constructor
 *
 * @throw lsst::pex::exceptions::LengthErrorException if nx or ny is non-positive
 */
SpatialCellSet::SpatialCellSet(image::BBox const& region, ///< Bounding box for %image
                               int xSize,                 ///< size of cells in the column direction
                               int ySize                  ///< size of cells in the row direction (0: == xSize)
                              ) :
    _region(region), _cellList(CellList()) {
    if (ySize == 0) {
        ySize = xSize;
    }
    
    if (xSize <= 0 || ySize <= 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::LengthErrorException,
                          (boost::format("Please specify cells that contain pixels, not %dx%d") %
                           xSize % ySize).str());
    }
    
    int nx = region.getWidth()/xSize;
    if (nx*xSize != region.getWidth()) {
        nx++;
    }

    int ny = region.getHeight()/ySize;
    if (ny*ySize != region.getHeight()) {
        ny++;
    }
    //
    // N.b. the SpatialCells will be sorted in y at the end of this
    //
    int y0 = region.getY0();
    for (int y = 0; y < ny; ++y) {
        int const y1 = (y == ny - 1) ? region.getY1() : (y + 1)*ySize; // ny may not be a factor of height
        int x0 = region.getX0();
        for (int x = 0; x < nx; ++x) {
            int const x1 = (x == nx - 1) ? region.getX1() : (x + 1)*xSize; // nx may not be a factor of width
            image::BBox bbox(image::PointI(x0, y0), image::PointI(x1, y1));
            std::string label = (boost::format("Cell %dx%d") % x % y).str();

            _cellList.push_back(SpatialCell::Ptr(new SpatialCell(label, bbox)));        

            x0 = x1 + 1;
        }
        y0 = y1 + 1;
    }
}

/************************************************************************************************************/

namespace {
    struct CellContains : public std::unary_function<SpatialCell::Ptr,
                                                     bool> {
        CellContains(SpatialCellCandidate::Ptr candidate) : _candidate(candidate) {}

        bool operator()(SpatialCell::Ptr cell) {
            return cell->getBBox().contains(image::PointI(image::positionToIndex(_candidate->getXCenter()),
                                                          image::positionToIndex(_candidate->getYCenter())));
        }
    private:
        SpatialCellCandidate::Ptr _candidate;
    };
}

/**
 * Insert a candidate into the correct cell
 */
void SpatialCellSet::insertCandidate(SpatialCellCandidate::Ptr candidate) {
    CellList::iterator pos = std::find_if(_cellList.begin(), _cellList.end(), CellContains(candidate));

    if (pos == _cellList.end()) {
        throw LSST_EXCEPT(lsst::pex::exceptions::OutOfRangeException,
                          (boost::format("Unable to insert a candidate at (%.2f, %.2f)") %
                           candidate->getXCenter() % candidate->getYCenter()).str());
    }

    (*pos)->insertCandidate(candidate);
}

}}}
