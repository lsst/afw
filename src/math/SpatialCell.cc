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

/// Set the candidate's status
void SpatialCellCandidate::setStatus(Status status) {
    switch (status) {
      case GOOD:
      case UNKNOWN:
        _status = status;
        return;
      case BAD:
        _status = status;
        return;
    }
    
    throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException,
                      (boost::format("Saw unknown status %d") % status).str());
}

/************************************************************************************************************/
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
    _ignoreBad(true)
{
    lsst::pex::logging::TTrace<3>("lsst.afw.math.SpatialCell", 
                                  "Cell %s : created with %d candidates",
                                  this->_label.c_str(), this->_candidateList.size());
    //
    // Sort the list to have the Larger things at the beginning.
    //
    sort(_candidateList.begin(), _candidateList.end(), CandidatePtrMore());
}

/************************************************************************************************************/
/**
 * Add a candidate to the list, preserving ranking
 */
void SpatialCell::insertCandidate(SpatialCellCandidate::Ptr candidate) {
    CandidateList::iterator pos = std::lower_bound(_candidateList.begin(), _candidateList.end(),
                                                   candidate, CandidatePtrMore());
    _candidateList.insert(pos, candidate);
}

/**
 * Determine if cell has no usable candidates
 */
bool SpatialCell::empty() const {
    // Cast away const;  end is only non-const as it provides access to the Candidates
    // and we don't (yet) have SpatialCellCandidateConstIterator
    SpatialCell *mthis = const_cast<SpatialCell *>(this);

    for (SpatialCellCandidateIterator ptr = mthis->begin(), end = mthis->end(); ptr != end; ++ptr) {
        if (!(_ignoreBad && (*ptr)->isBad())) { // found a good candidate, or don't care
            return false;
        }
    }

    return true;
}

/**
 * Return number of usable candidates in Cell
 */
size_t SpatialCell::size() const {
    // Cast away const; begin/end is only non-const as they provide access to the Candidates
    // and we don't (yet) have SpatialCellCandidateConstIterator
    SpatialCell *mthis = const_cast<SpatialCell *>(this);

    return mthis->end() - mthis->begin();
}

/************************************************************************************************************/
/// ctor; designed to be used to pass begin to SpatialCellCandidateIterator
SpatialCellCandidateIterator::SpatialCellCandidateIterator(
        CandidateList::iterator iterator, ///< Where this iterator should start
        CandidateList::iterator end,      ///< One-past-the-end of iterator's range
        bool ignoreBad                      ///< Should we pass over bad Candidates?
                                                          )
    : _iterator(iterator), _end(end), _ignoreBad(ignoreBad) {
    for (; _iterator != _end; ++_iterator) {
        (*_iterator)->instantiate();
        
        if (!(_ignoreBad && (*_iterator)->isBad())) { // found a good candidate, or don't care
            return;
        }
    }
}

/// ctor; designed to be used to pass end to SpatialCellCandidateIterator
SpatialCellCandidateIterator::SpatialCellCandidateIterator(
        CandidateList::iterator,          ///< start of of iterator's range; not used
        CandidateList::iterator end,      ///< Where this iterator should start
        bool ignoreBad,                     ///< Should we pass over bad Candidates?
        bool
                                                          )
    : _iterator(end), _end(end), _ignoreBad(ignoreBad) {
    if (ignoreBad) {
        // We could decrement end if there are bad Candidates at the end of the list, but it's probably
        // not worth the trouble
    }
}

/**
 * Advance the iterator, maybe skipping over candidates labelled BAD
 */
void SpatialCellCandidateIterator::operator++() {
    if (_iterator != _end) {
        ++_iterator;
    }
    
    for (; _iterator != _end; ++_iterator) {
        (*_iterator)->instantiate();
        
        if (!(_ignoreBad && (*_iterator)->isBad())) { // found a good candidate, or don't care
            return;
        }
    }
}

/**
 * Return the number of candidate between this and rhs
 */
size_t SpatialCellCandidateIterator::operator-(SpatialCellCandidateIterator const& rhs) const {
    size_t n = 0;
    for (SpatialCellCandidateIterator ptr = rhs; ptr != *this; ++ptr) {
        if (!(_ignoreBad && (*ptr)->isBad())) { // found a good candidate, or don't care
            ++n;
        }
    }

    return n;
}

/**
 * Dereference the iterator to return the Candidate (if there is one)
 *
 * @throw lsst::pex::exceptions::NotFoundErrorException if no candidate is available
 */
SpatialCellCandidate::ConstPtr SpatialCellCandidateIterator::operator*() const {
    if (_iterator == _end) {
        throw LSST_EXCEPT(lsst::pex::exceptions::NotFoundException, "Iterator points to end");
    }

    return *_iterator;
}

/// Return the CellCandidate::Ptr
SpatialCellCandidate::Ptr SpatialCellCandidateIterator::operator*() {
    if (_iterator == _end) {
        throw LSST_EXCEPT(lsst::pex::exceptions::NotFoundException, "Iterator points to end");
    }

    return *_iterator;
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
        int const y1 = (y == ny - 1) ? region.getY1() : (y + 1)*ySize - 1; // ny may not be a factor of height
        int x0 = region.getX0();
        for (int x = 0; x < nx; ++x) {
            int const x1 = (x == nx - 1) ? region.getX1() : (x + 1)*xSize - 1; // nx may not be a factor of width
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

/************************************************************************************************************/
/**
 * Call the visitor's processCandidate method for each Candidate in the SpatialSetCell
 *
 * @note This is obviously similar to the Design Patterns (Go4) Visitor pattern, but we've simplified the
 * double dispatch (i.e. we don't call a virtual method on SpatialCellCandidate that in turn calls
 * processCandidate(*this), but can be re-defined)
 */
void SpatialCellSet::visitCandidates(CandidateVisitor *visitor, ///< Pass this object to every Candidate
                                     int const nMaxPerCell      ///< Visit no more than this many Candidates (<= 0: all)
                                    ) {
    visitor->reset();
    
    for (CellList::iterator cell = _cellList.begin(), end = _cellList.end(); cell != end; ++cell) {
        int i = 0;
        for (SpatialCell::iterator candidate = (*cell)->begin(), candidateEnd = (*cell)->end();
             candidate != candidateEnd; ++candidate, ++i) {
            if (nMaxPerCell > 0 && i == nMaxPerCell) { // we've processed all the candidates we want
                break;
            }
            
            visitor->processCandidate((*candidate).get());
        }
    }
}
    
/**
 * Call the visitor's processCandidate method for each Candidate in the SpatialSetCell (const version)
 *
 * This is the const version of SpatialCellSet::visitCandidates
 *
 * @todo This is currently implemented via a const_cast (arghhh). The problem is that
 * SpatialCell::begin() const isn't yet implemented
 */
void SpatialCellSet::visitCandidates(CandidateVisitor const* visitor, ///< Pass this object to every Candidate
                                     int const nMaxPerCell      ///< Visit no more than this many Candidates (-ve: all)
                                    ) const {
#if 1
    //
    // These const_cast must go!
    //
    SpatialCellSet *mthis = const_cast<SpatialCellSet *>(this);
    CandidateVisitor *mvisitor = const_cast<CandidateVisitor *>(visitor);
    mthis->visitCandidates(mvisitor, nMaxPerCell);
#else
    visitor->reset();

    for (CellList::const_iterator cell = _cellList.begin(), end = _cellList.end(); cell != end; ++cell) {
        int i = 0;
        for (SpatialCell::const_iterator candidate = (*cell)->begin(), candidateEnd = (*cell)->end();
             candidate != candidateEnd; ++candidate, ++i) {
            if (i == nMaxPerCell) {   // we've processed all the candidates we want
                break;
            }
            
            visitor->processCandidate((*candidate).get());
        }
    }
#endif
}

}}}
