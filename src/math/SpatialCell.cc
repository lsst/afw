// -*- lsst-c++ -*-

/*
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
 *
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the LSST License Statement and
 * the GNU General Public License along with this program.  If not,
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */

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
#include "lsst/log/Log.h"
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

    throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError,
                      (boost::format("Saw unknown status %d") % status).str());
}

/************************************************************************************************************/
/**
 * Ctor
 */
SpatialCell::SpatialCell(std::string const& label, ///< string representing "name" of cell
                         geom::Box2I const& bbox,  ///< Bounding box of cell in overall image
                         CandidateList const& candidateList  ///< list of candidates to represent this cell
                        ) :
    _label(label),
    _bbox(bbox),
    _candidateList(candidateList),
    _ignoreBad(true)
{
    LOGF_TRACE3("lsst.afw.math.SpatialCell", "Cell %s : created with %d candidates",
                                  this->_label.c_str(), this->_candidateList.size());
    sortCandidates();
}

/************************************************************************************************************/
///
/// Rearrange the candidates to reflect their current ratings
///
void SpatialCell::sortCandidates()
{
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

void SpatialCell::removeCandidate(SpatialCellCandidate::Ptr candidate)
{
    CandidateList::iterator pos = std::find(_candidateList.begin(), _candidateList.end(), candidate);
    if (pos == _candidateList.end()) {
        throw LSST_EXCEPT(lsst::pex::exceptions::NotFoundError,
                          (boost::format("Unable to find candidate with ID == %d") %
                           candidate->getId()).str());
    }
    _candidateList.erase(pos);
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
/**
 * Return the SpatialCellCandidate with the specified id
 *
 * @throw lsst::pex::exceptions::NotFoundError if no candidate matches the id
 */
SpatialCellCandidate::Ptr SpatialCell::getCandidateById(int id, ///< The desired ID
                                                        bool noThrow ///< Return NULL in case of error
                                                       ) {
    for (SpatialCellCandidateIterator ptr = begin(), end = this->end(); ptr != end; ++ptr) {
        if ((*ptr)->getId() == id) {
            return *ptr;
        }
    }

    if (noThrow) {
        return SpatialCellCandidate::Ptr();
    } else {
        throw LSST_EXCEPT(lsst::pex::exceptions::NotFoundError,
                          (boost::format("Unable to find object with ID == %d") % id).str());
    }
}

/**
 * Call the visitor's processCandidate method for each Candidate in the SpatialCell
 *
 * @note This is obviously similar to the Design Patterns (Go4) Visitor pattern, but we've simplified the
 * double dispatch (i.e. we don't call a virtual method on SpatialCellCandidate that in turn calls
 * processCandidate(*this), but can be re-defined)
 */
void SpatialCell::visitCandidates(CandidateVisitor *visitor, ///< Pass this object to every Candidate
                                  int const nMaxPerCell, ///< Visit no more than
                                                         ///<   this many Candidates (<= 0: all)
                                  bool const ignoreExceptions, ///< Ignore any exceptions thrown by
                                                               ///<  the processing
                                  bool const reset             ///< Reset visitor before passing it around
                                 ) {
    if (reset) {
        visitor->reset();
    }

    int i = 0;
    for (SpatialCell::iterator candidate = begin(), candidateEnd = end();
         candidate != candidateEnd; ++candidate, ++i) {
        if (nMaxPerCell > 0 && i == nMaxPerCell) { // we've processed all the candidates we want
            return;
        }

        try {
            visitor->processCandidate((*candidate).get());
        } catch(lsst::pex::exceptions::Exception &e) {
            if (ignoreExceptions) {
                ;
            } else {
                LSST_EXCEPT_ADD(e, "Visiting candidate");
                throw e;
            }
        }
    }
}

/**
 * Call the visitor's processCandidate method for each Candidate in the SpatialCell (const version)
 *
 * This is the const version of SpatialCellSet::visitCandidates
 *
 * @todo This is currently implemented via a const_cast (arghhh). The problem is that
 * SpatialCell::begin() const isn't yet implemented
 */
void SpatialCell::visitCandidates(
        CandidateVisitor * visitor, ///< Pass this object to every Candidate
        int const nMaxPerCell,           ///< Visit no more than this many Candidates (-ve: all)
        bool const ignoreExceptions,     ///< Ignore any exceptions thrown by the processing
        bool const reset                 ///< Reset visitor before passing it around
                                 ) const {
#if 1
    //
    // This const_cast must go!
    //
    SpatialCell *mthis = const_cast<SpatialCell *>(this);
    mthis->visitCandidates(visitor, nMaxPerCell, ignoreExceptions, reset);
#else
    int i = 0;
    for (SpatialCell::const_iterator candidate = (*cell)->begin(), candidateEnd = (*cell)->end();
         candidate != candidateEnd; ++candidate, ++i) {
        if (i == nMaxPerCell) {   // we've processed all the candidates we want
            return;
        }

        try {
            visitor->processCandidate((*candidate).get());
        } catch(lsst::pex::exceptions::LengthError &e) {
            if (ignoreExceptions) {
                ;
            } else {
                LSST_EXCEPT_ADD(e, "Visiting candidate");
                throw e;
            }
        }
    }
#endif
}

/**
 * Call the visitor's processCandidate method for every Candidate in the SpatialCell
 *
 * @sa visitCandidates
 */
void SpatialCell::visitAllCandidates(CandidateVisitor *visitor, ///< Pass this object to every Candidate
                                     bool const ignoreExceptions, ///< Ignore any exceptions thrown by
                                     bool const reset             ///< Reset visitor before passing it around
                                    ) {
    if (reset) {
        visitor->reset();
    }

    int i = 0;
    for (SpatialCell::iterator candidate = begin(false), candidateEnd = end(false);
         candidate != candidateEnd; ++candidate, ++i) {
        try {
            visitor->processCandidate((*candidate).get());
        } catch(lsst::pex::exceptions::LengthError &e) {
            if (ignoreExceptions) {
                ;
            } else {
                LSST_EXCEPT_ADD(e, "Visiting candidate");
                throw e;
            }
        }
    }
}

/**
 * Call the visitor's processCandidate method for each Candidate in the SpatialCell (const version)
 *
 * This is the const version of SpatialCellSet::visitAllCandidates
 *
 * @todo This is currently implemented via a const_cast (arghhh). The problem is that
 * SpatialCell::begin() const isn't yet implemented
 */
void SpatialCell::visitAllCandidates(
        CandidateVisitor * visitor, ///< Pass this object to every Candidate
        bool const ignoreExceptions,     ///< Ignore any exceptions thrown by the processing
        bool const reset                 ///< Reset visitor before passing it around
                                 ) const {
#if 1
    //
    // This const_cast must go!
    //
    SpatialCell *mthis = const_cast<SpatialCell *>(this);
    mthis->visitAllCandidates(visitor, ignoreExceptions, reset);
#else
    int i = 0;
    for (SpatialCell::const_iterator candidate = (*cell)->begin(false), candidateEnd = (*cell)->end(false);
         candidate != candidateEnd; ++candidate, ++i) {
        try {
            visitor->processCandidate((*candidate).get());
        } catch(lsst::pex::exceptions::LengthError &e) {
            if (ignoreExceptions) {
                ;
            } else {
                LSST_EXCEPT_ADD(e, "Visiting candidate");
                throw e;
            }
        }
    }
#endif
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
 * @throw lsst::pex::exceptions::NotFoundError if no candidate is available
 */
SpatialCellCandidate::ConstPtr SpatialCellCandidateIterator::operator*() const {
    if (_iterator == _end) {
        throw LSST_EXCEPT(lsst::pex::exceptions::NotFoundError, "Iterator points to end");
    }

    return *_iterator;
}

/// Return the CellCandidate::Ptr
SpatialCellCandidate::Ptr SpatialCellCandidateIterator::operator*() {
    if (_iterator == _end) {
        throw LSST_EXCEPT(lsst::pex::exceptions::NotFoundError, "Iterator points to end");
    }

    return *_iterator;
}

/************************************************************************************************************/

/**
 * Constructor
 *
 * @throw lsst::pex::exceptions::LengthError if nx or ny is non-positive
 */
SpatialCellSet::SpatialCellSet(geom::Box2I const& region, ///< Bounding box for %image
                               int xSize,              ///< size of cells in the column direction
                               int ySize               ///< size of cells in the row direction (0: == xSize)
                              ) :
    _region(region), _cellList(CellList()) {
    if (ySize == 0) {
        ySize = xSize;
    }

    if (xSize <= 0 || ySize <= 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::LengthError,
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
    int y0 = region.getMinY();
    for (int y = 0; y < ny; ++y) {
        // ny may not be a factor of height
        int const y1 = (y == ny - 1) ? region.getMaxY() : y0 + ySize - 1;
        int x0 = region.getMinX();
        for (int x = 0; x < nx; ++x) {
            // nx may not be a factor of width
            int const x1 = (x == nx - 1) ? region.getMaxX() : x0 + xSize - 1;
            geom::Box2I bbox(geom::Point2I(x0, y0), geom::Point2I(x1, y1));
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
            return cell->getBBox().contains(geom::Point2I(image::positionToIndex(_candidate->getXCenter()),
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
        throw LSST_EXCEPT(lsst::pex::exceptions::OutOfRangeError,
                          (boost::format("Unable to insert a candidate at (%.2f, %.2f)") %
                           candidate->getXCenter() % candidate->getYCenter()).str());
    }

    (*pos)->insertCandidate(candidate);
}


/************************************************************************************************************/
///
/// Rearrange the Candidates in all SpatialCells to reflect their current ratings
///
void SpatialCellSet::sortCandidates()
{
    for (CellList::iterator cell = _cellList.begin(), end = _cellList.end(); cell != end; ++cell) {
        (*cell)->sortCandidates();
    }
}

/************************************************************************************************************/
/**
 * Call the visitor's processCandidate method for each Candidate in the SpatialCellSet
 *
 * @note This is obviously similar to the Design Patterns (Go4) Visitor pattern, but we've simplified the
 * double dispatch (i.e. we don't call a virtual method on SpatialCellCandidate that in turn calls
 * processCandidate(*this), but can be re-defined)
 */
void SpatialCellSet::visitCandidates(
        CandidateVisitor *visitor,      ///< Pass this object to every Candidate
        int const nMaxPerCell,          ///< Visit no more than this many Candidates (<= 0: all)
        bool const ignoreExceptions     ///< Ignore any exceptions thrown by the processing
                                    ) {
    visitor->reset();

    for (CellList::iterator cell = _cellList.begin(), end = _cellList.end(); cell != end; ++cell) {
        (*cell)->visitCandidates(visitor, nMaxPerCell, ignoreExceptions, false);
    }
}

/**
 * Call the visitor's processCandidate method for each Candidate in the SpatialCellSet (const version)
 *
 * This is the const version of SpatialCellSet::visitCandidates
 */
void SpatialCellSet::visitCandidates(
        CandidateVisitor *visitor, ///< Pass this object to every Candidate
        int const nMaxPerCell,          ///< Visit no more than this many Candidates (-ve: all)
        bool const ignoreExceptions ///< Ignore any exceptions thrown by the processing
                                    ) const {
    visitor->reset();

    for (CellList::const_iterator cell = _cellList.begin(), end = _cellList.end(); cell != end; ++cell) {
        SpatialCell const *ccell = cell->get(); // the SpatialCellSet's SpatialCells should be const too
        ccell->visitCandidates(visitor, nMaxPerCell, ignoreExceptions, false);
    }
}

/************************************************************************************************************/
/**
 * Call the visitor's processCandidate method for every Candidate in the SpatialCellSet
 *
 * @sa visitCandidates
 */
void SpatialCellSet::visitAllCandidates(
        CandidateVisitor *visitor,      ///< Pass this object to every Candidate
        bool const ignoreExceptions     ///< Ignore any exceptions thrown by the processing
                                    ) {
    visitor->reset();

    for (CellList::iterator cell = _cellList.begin(), end = _cellList.end(); cell != end; ++cell) {
        (*cell)->visitAllCandidates(visitor, ignoreExceptions, false);
    }
}

/**
 * Call the visitor's processCandidate method for every Candidate in the SpatialCellSet (const version)
 *
 * This is the const version of SpatialCellSet::visitAllCandidates
 */
void SpatialCellSet::visitAllCandidates(
        CandidateVisitor *visitor, ///< Pass this object to every Candidate
        bool const ignoreExceptions ///< Ignore any exceptions thrown by the processing
                                    ) const {
    visitor->reset();

    for (CellList::const_iterator cell = _cellList.begin(), end = _cellList.end(); cell != end; ++cell) {
        SpatialCell const *ccell = cell->get(); // the SpatialCellSet's SpatialCells should be const too
        ccell->visitAllCandidates(visitor, ignoreExceptions, false);
    }
}

/************************************************************************************************************/
/**
 * Return the SpatialCellCandidate with the specified id
 *
 * @throw lsst::pex::exceptions::NotFoundError if no candidate matches the id (unless noThrow
 * is true, in which case a Ptr(NULL) is returned
 */
SpatialCellCandidate::Ptr SpatialCellSet::getCandidateById(int id, ///< The desired ID
                                                           bool noThrow ///< Return NULL in case of error
                                                       ) {
    for (CellList::iterator cell = _cellList.begin(), end = _cellList.end(); cell != end; ++cell) {
        SpatialCellCandidate::Ptr cand = (*cell)->getCandidateById(id, true);

        if (cand) {
            return cand;
        }
    }

    if (noThrow) {
        return SpatialCellCandidate::Ptr();
    } else {
        throw LSST_EXCEPT(lsst::pex::exceptions::NotFoundError,
                          (boost::format("Unable to find object with ID == %d") % id).str());
    }
}

/// Set whether we should omit BAD candidates from candidate list when traversing
void SpatialCellSet::setIgnoreBad(bool ignoreBad) {
    for (CellList::iterator cell = _cellList.begin(), end = _cellList.end(); cell != end; ++cell) {
        (*cell)->setIgnoreBad(ignoreBad);
    }
}

}}}
