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

/*
 * Implementation of SpatialCell class
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
struct CandidatePtrMore : public std::binary_function<std::shared_ptr<SpatialCellCandidate>,
                                                      std::shared_ptr<SpatialCellCandidate>, bool> {
    bool operator()(std::shared_ptr<SpatialCellCandidate> a, std::shared_ptr<SpatialCellCandidate> b) {
        return a->getCandidateRating() > b->getCandidateRating();
    }
};
}

int SpatialCellCandidate::_CandidateId = 0;

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

int SpatialCellImageCandidate::_width = 0;

int SpatialCellImageCandidate::_height = 0;

SpatialCell::SpatialCell(std::string const &label, lsst::geom::Box2I const &bbox,
                         CandidateList const &candidateList)
        : _label(label), _bbox(bbox), _candidateList(candidateList), _ignoreBad(true) {
    LOGL_DEBUG("afw.math.SpatialCell", "Cell %s : created with %d candidates", this->_label.c_str(),
               this->_candidateList.size());
    sortCandidates();
}

void SpatialCell::sortCandidates() { sort(_candidateList.begin(), _candidateList.end(), CandidatePtrMore()); }

void SpatialCell::insertCandidate(std::shared_ptr<SpatialCellCandidate> candidate) {
    CandidateList::iterator pos =
            std::lower_bound(_candidateList.begin(), _candidateList.end(), candidate, CandidatePtrMore());
    _candidateList.insert(pos, candidate);
}

void SpatialCell::removeCandidate(std::shared_ptr<SpatialCellCandidate> candidate) {
    CandidateList::iterator pos = std::find(_candidateList.begin(), _candidateList.end(), candidate);
    if (pos == _candidateList.end()) {
        throw LSST_EXCEPT(
                lsst::pex::exceptions::NotFoundError,
                (boost::format("Unable to find candidate with ID == %d") % candidate->getId()).str());
    }
    _candidateList.erase(pos);
}

bool SpatialCell::empty() const {
    // Cast away const;  end is only non-const as it provides access to the Candidates
    // and we don't (yet) have SpatialCellCandidateConstIterator
    SpatialCell *mthis = const_cast<SpatialCell *>(this);

    for (SpatialCellCandidateIterator ptr = mthis->begin(), end = mthis->end(); ptr != end; ++ptr) {
        if (!(_ignoreBad && (*ptr)->isBad())) {  // found a good candidate, or don't care
            return false;
        }
    }

    return true;
}

size_t SpatialCell::size() const {
    // Cast away const; begin/end is only non-const as they provide access to the Candidates
    // and we don't (yet) have SpatialCellCandidateConstIterator
    SpatialCell *mthis = const_cast<SpatialCell *>(this);

    return mthis->end() - mthis->begin();
}

std::shared_ptr<SpatialCellCandidate> SpatialCell::getCandidateById(int id, bool noThrow) {
    for (SpatialCellCandidateIterator ptr = begin(), end = this->end(); ptr != end; ++ptr) {
        if ((*ptr)->getId() == id) {
            return *ptr;
        }
    }

    if (noThrow) {
        return std::shared_ptr<SpatialCellCandidate>();
    } else {
        throw LSST_EXCEPT(lsst::pex::exceptions::NotFoundError,
                          (boost::format("Unable to find object with ID == %d") % id).str());
    }
}

void SpatialCell::visitCandidates(CandidateVisitor *visitor, int const nMaxPerCell,

                                  bool const ignoreExceptions, bool const reset) {
    if (reset) {
        visitor->reset();
    }

    int i = 0;
    for (SpatialCell::iterator candidate = begin(), candidateEnd = end(); candidate != candidateEnd;
         ++candidate, ++i) {
        if (nMaxPerCell > 0 && i == nMaxPerCell) {  // we've processed all the candidates we want
            return;
        }

        try {
            visitor->processCandidate((*candidate).get());
        } catch (lsst::pex::exceptions::Exception &e) {
            if (ignoreExceptions) {
                ;
            } else {
                LSST_EXCEPT_ADD(e, "Visiting candidate");
                throw e;
            }
        }
    }
}

void SpatialCell::visitCandidates(CandidateVisitor *visitor, int const nMaxPerCell,
                                  bool const ignoreExceptions, bool const reset) const {
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
        if (i == nMaxPerCell) {  // we've processed all the candidates we want
            return;
        }

        try {
            visitor->processCandidate((*candidate).get());
        } catch (lsst::pex::exceptions::LengthError &e) {
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

void SpatialCell::visitAllCandidates(CandidateVisitor *visitor, bool const ignoreExceptions,
                                     bool const reset) {
    if (reset) {
        visitor->reset();
    }

    int i = 0;
    for (SpatialCell::iterator candidate = begin(false), candidateEnd = end(false); candidate != candidateEnd;
         ++candidate, ++i) {
        try {
            visitor->processCandidate((*candidate).get());
        } catch (lsst::pex::exceptions::LengthError &e) {
            if (ignoreExceptions) {
                ;
            } else {
                LSST_EXCEPT_ADD(e, "Visiting candidate");
                throw e;
            }
        }
    }
}

void SpatialCell::visitAllCandidates(CandidateVisitor *visitor, bool const ignoreExceptions,
                                     bool const reset) const {
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
        } catch (lsst::pex::exceptions::LengthError &e) {
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

SpatialCellCandidateIterator::SpatialCellCandidateIterator(CandidateList::iterator iterator,
                                                           CandidateList::iterator end, bool ignoreBad)
        : _iterator(iterator), _end(end), _ignoreBad(ignoreBad) {
    for (; _iterator != _end; ++_iterator) {
        (*_iterator)->instantiate();

        if (!(_ignoreBad && (*_iterator)->isBad())) {  // found a good candidate, or don't care
            return;
        }
    }
}

SpatialCellCandidateIterator::SpatialCellCandidateIterator(CandidateList::iterator,
                                                           CandidateList::iterator end, bool ignoreBad, bool)
        : _iterator(end), _end(end), _ignoreBad(ignoreBad) {
    if (ignoreBad) {
        // We could decrement end if there are bad Candidates at the end of the list, but it's probably
        // not worth the trouble
    }
}

void SpatialCellCandidateIterator::operator++() {
    if (_iterator != _end) {
        ++_iterator;
    }

    for (; _iterator != _end; ++_iterator) {
        (*_iterator)->instantiate();

        if (!(_ignoreBad && (*_iterator)->isBad())) {  // found a good candidate, or don't care
            return;
        }
    }
}

size_t SpatialCellCandidateIterator::operator-(SpatialCellCandidateIterator const &rhs) const {
    size_t n = 0;
    for (SpatialCellCandidateIterator ptr = rhs; ptr != *this; ++ptr) {
        if (!(_ignoreBad && (*ptr)->isBad())) {  // found a good candidate, or don't care
            ++n;
        }
    }

    return n;
}

std::shared_ptr<SpatialCellCandidate const> SpatialCellCandidateIterator::operator*() const {
    if (_iterator == _end) {
        throw LSST_EXCEPT(lsst::pex::exceptions::NotFoundError, "Iterator points to end");
    }

    return *_iterator;
}

std::shared_ptr<SpatialCellCandidate> SpatialCellCandidateIterator::operator*() {
    if (_iterator == _end) {
        throw LSST_EXCEPT(lsst::pex::exceptions::NotFoundError, "Iterator points to end");
    }

    return *_iterator;
}

SpatialCellSet::SpatialCellSet(lsst::geom::Box2I const &region, int xSize, int ySize)
        : _region(region), _cellList(CellList()) {
    if (ySize == 0) {
        ySize = xSize;
    }

    if (xSize <= 0 || ySize <= 0) {
        throw LSST_EXCEPT(
                lsst::pex::exceptions::LengthError,
                (boost::format("Please specify cells that contain pixels, not %dx%d") % xSize % ySize).str());
    }

    int nx = region.getWidth() / xSize;
    if (nx * xSize != region.getWidth()) {
        nx++;
    }

    int ny = region.getHeight() / ySize;
    if (ny * ySize != region.getHeight()) {
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
            lsst::geom::Box2I bbox(lsst::geom::Point2I(x0, y0), lsst::geom::Point2I(x1, y1));
            std::string label = (boost::format("Cell %dx%d") % x % y).str();

            _cellList.push_back(std::shared_ptr<SpatialCell>(new SpatialCell(label, bbox)));

            x0 = x1 + 1;
        }
        y0 = y1 + 1;
    }
}

namespace {
struct CellContains : public std::unary_function<std::shared_ptr<SpatialCell>, bool> {
    CellContains(std::shared_ptr<SpatialCellCandidate> candidate) : _candidate(candidate) {}

    bool operator()(std::shared_ptr<SpatialCell> cell) {
        return cell->getBBox().contains(lsst::geom::Point2I(image::positionToIndex(_candidate->getXCenter()),
                                                      image::positionToIndex(_candidate->getYCenter())));
    }

private:
    std::shared_ptr<SpatialCellCandidate> _candidate;
};
}

void SpatialCellSet::insertCandidate(std::shared_ptr<SpatialCellCandidate> candidate) {
    CellList::iterator pos = std::find_if(_cellList.begin(), _cellList.end(), CellContains(candidate));

    if (pos == _cellList.end()) {
        throw LSST_EXCEPT(lsst::pex::exceptions::OutOfRangeError,
                          (boost::format("Unable to insert a candidate at (%.2f, %.2f)") %
                           candidate->getXCenter() % candidate->getYCenter())
                                  .str());
    }

    (*pos)->insertCandidate(candidate);
}

void SpatialCellSet::sortCandidates() {
    for (CellList::iterator cell = _cellList.begin(), end = _cellList.end(); cell != end; ++cell) {
        (*cell)->sortCandidates();
    }
}

void SpatialCellSet::visitCandidates(CandidateVisitor *visitor, int const nMaxPerCell,
                                     bool const ignoreExceptions) {
    visitor->reset();

    for (CellList::iterator cell = _cellList.begin(), end = _cellList.end(); cell != end; ++cell) {
        (*cell)->visitCandidates(visitor, nMaxPerCell, ignoreExceptions, false);
    }
}

void SpatialCellSet::visitCandidates(CandidateVisitor *visitor, int const nMaxPerCell,
                                     bool const ignoreExceptions) const {
    visitor->reset();

    for (CellList::const_iterator cell = _cellList.begin(), end = _cellList.end(); cell != end; ++cell) {
        SpatialCell const *ccell = cell->get();  // the SpatialCellSet's SpatialCells should be const too
        ccell->visitCandidates(visitor, nMaxPerCell, ignoreExceptions, false);
    }
}

void SpatialCellSet::visitAllCandidates(CandidateVisitor *visitor, bool const ignoreExceptions) {
    visitor->reset();

    for (CellList::iterator cell = _cellList.begin(), end = _cellList.end(); cell != end; ++cell) {
        (*cell)->visitAllCandidates(visitor, ignoreExceptions, false);
    }
}

void SpatialCellSet::visitAllCandidates(CandidateVisitor *visitor, bool const ignoreExceptions) const {
    visitor->reset();

    for (CellList::const_iterator cell = _cellList.begin(), end = _cellList.end(); cell != end; ++cell) {
        SpatialCell const *ccell = cell->get();  // the SpatialCellSet's SpatialCells should be const too
        ccell->visitAllCandidates(visitor, ignoreExceptions, false);
    }
}

std::shared_ptr<SpatialCellCandidate> SpatialCellSet::getCandidateById(int id, bool noThrow) {
    for (CellList::iterator cell = _cellList.begin(), end = _cellList.end(); cell != end; ++cell) {
        std::shared_ptr<SpatialCellCandidate> cand = (*cell)->getCandidateById(id, true);

        if (cand) {
            return cand;
        }
    }

    if (noThrow) {
        return std::shared_ptr<SpatialCellCandidate>();
    } else {
        throw LSST_EXCEPT(lsst::pex::exceptions::NotFoundError,
                          (boost::format("Unable to find object with ID == %d") % id).str());
    }
}

void SpatialCellSet::setIgnoreBad(bool ignoreBad) {
    for (CellList::iterator cell = _cellList.begin(), end = _cellList.end(); cell != end; ++cell) {
        (*cell)->setIgnoreBad(ignoreBad);
    }
}
}
}
}
