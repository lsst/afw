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
 * Class to ensure constraints for spatial modeling
 */

#ifndef LSST_AFW_MATH_SPATIALCELL_H
#define LSST_AFW_MATH_SPATIALCELL_H

#include <limits>
#include <vector>
#include <string>

#include <memory>
#include "lsst/base.h"
#include "lsst/pex/exceptions.h"
#include "lsst/geom.h"
#include "lsst/afw/image/LsstImageTypes.h"

namespace lsst {
namespace afw {

// forward declarations
namespace image {
template <typename ImagePixelT>
class Image;
template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
class MaskedImage;
}  // namespace image

namespace math {

/// A class to pass around to all our Candidates
class SpatialCellCandidate;

class CandidateVisitor {
public:
    CandidateVisitor() {}
    virtual ~CandidateVisitor() {}

    virtual void reset() {}
    virtual void processCandidate(SpatialCellCandidate*) {}
};

/**
 * Base class for candidate objects in a SpatialCell
 */
class SpatialCellCandidate {
public:
    enum Status { BAD = 0, GOOD = 1, UNKNOWN = 2 };

    SpatialCellCandidate(float const xCenter,  ///< The object's column-centre
                         float const yCenter   ///< The object's row-centre
                         )
            : _id(++_CandidateId), _status(UNKNOWN), _xCenter(xCenter), _yCenter(yCenter) {}

    SpatialCellCandidate(SpatialCellCandidate const&) = default;
    SpatialCellCandidate(SpatialCellCandidate&&) = default;
    SpatialCellCandidate& operator=(SpatialCellCandidate const&) = default;
    SpatialCellCandidate& operator=(SpatialCellCandidate&&) = default;

    /**
     * (virtual) destructor -- this is a base class you know
     */
    virtual ~SpatialCellCandidate() = default;

    /// Return the object's column-centre
    float getXCenter() const { return _xCenter; }

    /// Return the object's row-centre
    float getYCenter() const { return _yCenter; }

    /// Do anything needed to make this candidate usable
    virtual bool instantiate() { return true; }

    /// Return candidate's rating
    virtual double getCandidateRating() const = 0;
    /// Set the candidate's rating
    virtual void setCandidateRating(double) {}

    /// Return the candidate's unique ID
    int getId() const { return _id; }
    /// Return the candidate's status
    Status getStatus() const { return _status; }
    /// Set the candidate's status
    void setStatus(Status status);
    /// Is this candidate unacceptable?
    virtual bool isBad() const { return (_status == BAD); }

private:
    int _id;               // Unique ID for object
    Status _status;        // Is this Candidate good?
    float const _xCenter;  // The object's column-centre
    float const _yCenter;  // The object's row-centre

    /// Unique identifier for candidates; useful for preserving current candidate following insertion
    static int _CandidateId;
};

/**
 * Base class for candidate objects in a SpatialCell that are able to return an Image of some sort
 * (e.g. a PSF or a DIA kernel)
 */
class SpatialCellImageCandidate : public SpatialCellCandidate {
public:
    /// ctor
    SpatialCellImageCandidate(float const xCenter,  ///< The object's column-centre
                              float const yCenter   ///< The object's row-centre
                              )
            : SpatialCellCandidate(xCenter, yCenter), _chi2(std::numeric_limits<double>::max()) {}
    SpatialCellImageCandidate(SpatialCellImageCandidate const&) = default;
    SpatialCellImageCandidate(SpatialCellImageCandidate&&) = default;
    SpatialCellImageCandidate& operator=(SpatialCellImageCandidate const&) = default;
    SpatialCellImageCandidate& operator=(SpatialCellImageCandidate&&) = default;
    ~SpatialCellImageCandidate() override = default;

    /// Set the width of the image that getImage should return
    static void setWidth(int width) { _width = width; }
    /// Return the width of the image that getImage should return
    static int getWidth() { return _width; }

    /// Set the height of the image that getImage should return
    static void setHeight(int height) { _height = height; }
    /// Return the height of the image that getImage should return
    static int getHeight() { return _height; }

    /// Return the candidate's chi^2
    double getChi2() const { return _chi2; }
    /// Set the candidate's chi^2
    void setChi2(double chi2) { _chi2 = chi2; }

private:
    static int _width;   // the width of images to return; may be ignored by subclasses
    static int _height;  // the height of images to return; may be ignored by subclasses
    double _chi2;        // chi^2 for fit
};

/**
 * An iterator that only returns usable members of the SpatialCell
 */
class SpatialCellCandidateIterator {
    friend class SpatialCell;
    typedef std::vector<std::shared_ptr<SpatialCellCandidate>> CandidateList;

public:
    // ctors are protected
    /**
     * Advance the iterator, maybe skipping over candidates labelled BAD
     */
    void operator++();
    /**
     * Return the number of candidate between this and rhs
     */
    size_t operator-(SpatialCellCandidateIterator const& rhs) const;

    /**
     * Dereference the iterator to return the Candidate (if there is one)
     *
     * @throws lsst::pex::exceptions::NotFoundError if no candidate is available
     */
    std::shared_ptr<SpatialCellCandidate const> operator*() const;
    /// Return the std::shared_ptr<CellCandidate>
    std::shared_ptr<SpatialCellCandidate> operator*();

    /// Are two SpatialCellCandidateIterator%s equal?
    bool operator==(SpatialCellCandidateIterator const& rhs) const { return _iterator == rhs._iterator; }
    /// Are two SpatialCellCandidateIterator%s unequal?
    bool operator!=(SpatialCellCandidateIterator const& rhs) const { return _iterator != rhs._iterator; }

protected:
    /** ctor; designed to be used to pass begin to SpatialCellCandidateIterator
     *
     * @param iterator Where this iterator should start
     * @param end One-past-the-end of iterator's range
     * @param ignoreBad Should we pass over bad Candidates?
     */
    SpatialCellCandidateIterator(CandidateList::iterator iterator, CandidateList::iterator end,
                                 bool ignoreBad);
    /** ctor; designed to be used to pass end to SpatialCellCandidateIterator
     *
     * @param iterator start of of iterator's range; not used
     * @param end Where this iterator should start
     * @param ignoreBad Should we pass over bad Candidates?
     */
    SpatialCellCandidateIterator(CandidateList::iterator iterator, CandidateList::iterator end,
                                 bool ignoreBad, bool);

private:
    CandidateList::iterator _iterator;
    CandidateList::iterator _end;
    bool _ignoreBad;
};

/**
 * Class to ensure constraints for spatial modeling
 *
 * A given %image is divided up into cells, with each cell represented by an instance of this class.
 * Each cell itself contains a list of instances of classes derived from SpatialCellCandidate.  One class
 * member from each cell will be chosen to fit to a spatial model.  In case of a poor fit, the next class
 * instance in the list will be fit for.  If all instances in a list are rejected from the spatial model,
 * the best one will be used.
 *
 * @see @link SpatialCellSetExample@endlink
 */
class SpatialCell {
public:
    typedef std::vector<std::shared_ptr<SpatialCellCandidate>> CandidateList;
    typedef SpatialCellCandidateIterator iterator;
    /**
     * Constructor
     *
     * @param label string representing "name" of cell
     * @param bbox Bounding box of cell in overall image
     * @param candidateList list of candidates to represent this cell
     */
    SpatialCell(std::string const& label, lsst::geom::Box2I const& bbox = lsst::geom::Box2I(),
                CandidateList const& candidateList = CandidateList());

    SpatialCell(SpatialCell const&) = default;
    SpatialCell(SpatialCell&&) = default;
    SpatialCell& operator=(SpatialCell const&) = default;
    SpatialCell& operator=(SpatialCell&&) = default;

    /**
     * Destructor
     */
    virtual ~SpatialCell() = default;

    /**
     * Determine if cell has no usable candidates
     */
    bool empty() const;
    /**
     * Return number of usable candidates in Cell
     */
    size_t size() const;

    /**
     * Rearrange the candidates to reflect their current ratings
     */
    void sortCandidates();
    /**
     * Return an iterator to the beginning of the Candidates
     */
    SpatialCellCandidateIterator begin() {
        return SpatialCellCandidateIterator(_candidateList.begin(), _candidateList.end(), _ignoreBad);
    }
    SpatialCellCandidateIterator begin(bool ignoreBad  ///< If true, ignore BAD candidates
    ) {
        return SpatialCellCandidateIterator(_candidateList.begin(), _candidateList.end(), ignoreBad);
    }
    /**
     * Return an iterator to (one after) the end of the Candidates
     */
    SpatialCellCandidateIterator end() {
        return SpatialCellCandidateIterator(_candidateList.begin(), _candidateList.end(), _ignoreBad, true);
    }
    SpatialCellCandidateIterator end(bool ignoreBad  ///< If true, ignore BAD candidates
    ) {
        return SpatialCellCandidateIterator(_candidateList.begin(), _candidateList.end(), ignoreBad, true);
    }
    /**
     * Add a candidate to the list, preserving ranking
     */
    void insertCandidate(std::shared_ptr<SpatialCellCandidate> candidate);

    /** Remove a candidate from the list
     *
     * This is not a particularly efficient operation, since we're
     * using a std::vector, but should not hurt too much if the number
     * of candidates in a cell is small.
     */
    void removeCandidate(std::shared_ptr<SpatialCellCandidate> candidate);

    /// Set whether we should omit BAD candidates from candidate list when traversing
    void setIgnoreBad(bool ignoreBad) { _ignoreBad = ignoreBad; }
    /// Get whether we are omitting BAD candidates from candidate list when traversing
    bool getIgnoreBad() const { return _ignoreBad; }

    /**
     * Return the SpatialCellCandidate with the specified id
     *
     * @param id The desired ID
     * @param noThrow Return NULL in case of error
     *
     * @throws lsst::pex::exceptions::NotFoundError if no candidate matches the id
     */
    std::shared_ptr<SpatialCellCandidate> getCandidateById(int id, bool noThrow = false);
    /**
     * Get SpatialCell's label
     */
    std::string const& getLabel() const { return _label; }
    /**
     * Get SpatialCell's BBox
     */
    lsst::geom::Box2I const& getBBox() const { return _bbox; }
    /*
     * Visit our candidates
     */
    /**
     * Call the visitor's processCandidate method for each Candidate in the SpatialCell
     *
     * @param visitor Pass this object to every Candidate
     * @param nMaxPerCell Visit no more than this many Candidates (<= 0: all)
     * @param ignoreExceptions Ignore any exceptions thrown by the processing
     * @param reset Reset visitor before passing it around
     *
     * @note This is obviously similar to the Design Patterns (Go4) Visitor pattern, but we've simplified the
     * double dispatch (i.e. we don't call a virtual method on SpatialCellCandidate that in turn calls
     * processCandidate(*this), but can be re-defined)
     */
    void visitCandidates(CandidateVisitor* visitor, int const nMaxPerCell = -1,
                         bool const ignoreExceptions = false, bool const reset = true);
    /**
     * Call the visitor's processCandidate method for each Candidate in the SpatialCell (const version)
     *
     * This is the const version of SpatialCellSet::visitCandidates
     *
     * @param visitor Pass this object to every Candidate
     * @param nMaxPerCell Visit no more than this many Candidates (-ve: all)
     * @param ignoreExceptions Ignore any exceptions thrown by the processing
     * @param reset Reset visitor before passing it around
     *
     * @todo This is currently implemented via a const_cast (arghhh). The problem is that
     * SpatialCell::begin() const isn't yet implemented
     */
    void visitCandidates(CandidateVisitor* visitor, int const nMaxPerCell = -1,
                         bool const ignoreExceptions = false, bool const reset = true) const;
    /**
     * Call the visitor's processCandidate method for every Candidate in the SpatialCell
     *
     * @param visitor Pass this object to every Candidate
     * @param ignoreExceptions Ignore any exceptions thrown by
     * @param reset Reset visitor before passing it around
     *
     * @see visitCandidates
     */
    void visitAllCandidates(CandidateVisitor* visitor, bool const ignoreExceptions = false,
                            bool const reset = true);
    /**
     * Call the visitor's processCandidate method for each Candidate in the SpatialCell (const version)
     *
     * This is the const version of SpatialCellSet::visitAllCandidates
     *
     * @param visitor Pass this object to every Candidate
     * @param ignoreExceptions Ignore any exceptions thrown by the processing
     * @param reset Reset visitor before passing it around
     *
     * @todo This is currently implemented via a const_cast (arghhh). The problem is that
     * SpatialCell::begin() const isn't yet implemented
     */
    void visitAllCandidates(CandidateVisitor* visitor, bool const ignoreExceptions = false,
                            bool const reset = true) const;

private:
    std::string _label;            // Name of cell for logging/trace
    lsst::geom::Box2I _bbox;       // Bounding box of cell in overall image
    CandidateList _candidateList;  // List of all candidates in the cell
    bool _ignoreBad;               // Don't include BAD candidates when traversing the list
};

/**
 * A collection of SpatialCells covering an entire %image
 */
class SpatialCellSet {
public:
    typedef std::vector<std::shared_ptr<SpatialCell>> CellList;

    /**
     * Constructor
     *
     * @param region Bounding box for %image
     * @param xSize size of cells in the column direction
     * @param ySize size of cells in the row direction (0: == xSize)
     *
     * @throws lsst::pex::exceptions::LengthError if nx or ny is non-positive
     */
    SpatialCellSet(lsst::geom::Box2I const& region, int xSize, int ySize = 0);

    SpatialCellSet(SpatialCellSet const&) = default;
    SpatialCellSet(SpatialCellSet&&) = default;
    SpatialCellSet& operator=(SpatialCellSet const&) = default;
    SpatialCellSet& operator=(SpatialCellSet&&) = default;

    /**
     * Destructor
     */
    virtual ~SpatialCellSet() = default;

    /**
     * Return our SpatialCells
     */
    CellList& getCellList() { return _cellList; }

    /**
     * Return the bounding box of the %image
     */
    lsst::geom::Box2I getBBox() const { return _region; };

    /**
     * Insert a candidate into the correct cell
     */
    void insertCandidate(std::shared_ptr<SpatialCellCandidate> candidate);

    /// Rearrange the Candidates in all SpatialCells to reflect their current ratings
    void sortCandidates();

    /**
     * Call the visitor's processCandidate method for each Candidate in the SpatialCellSet
     *
     * @param visitor Pass this object to every Candidate
     * @param nMaxPerCell Visit no more than this many Candidates (<= 0: all)
     * @param ignoreExceptions Ignore any exceptions thrown by the processing
     *
     * @note This is obviously similar to the Design Patterns (Go4) Visitor pattern, but we've simplified the
     * double dispatch (i.e. we don't call a virtual method on SpatialCellCandidate that in turn calls
     * processCandidate(*this), but can be re-defined)
     */
    void visitCandidates(CandidateVisitor* visitor, int const nMaxPerCell = -1,
                         bool const ignoreExceptions = false);
    /**
     * Call the visitor's processCandidate method for each Candidate in the SpatialCellSet (const version)
     *
     * This is the const version of SpatialCellSet::visitCandidates
     *
     * @param visitor Pass this object to every Candidate
     * @param nMaxPerCell Visit no more than this many Candidates (-ve: all)
     * @param ignoreExceptions Ignore any exceptions thrown by the processing
     */
    void visitCandidates(CandidateVisitor* visitor, int const nMaxPerCell = -1,
                         bool const ignoreExceptions = false) const;
    /**
     * Call the visitor's processCandidate method for every Candidate in the SpatialCellSet
     *
     * @param visitor Pass this object to every Candidate
     * @param ignoreExceptions Ignore any exceptions thrown by the processing
     *
     * @see visitCandidates
     */
    void visitAllCandidates(CandidateVisitor* visitor, bool const ignoreExceptions = false);
    /**
     * Call the visitor's processCandidate method for every Candidate in the SpatialCellSet (const version)
     *
     * This is the const version of SpatialCellSet::visitAllCandidates
     *
     * @param visitor Pass this object to every Candidate
     * @param ignoreExceptions Ignore any exceptions thrown by the processing
     */
    void visitAllCandidates(CandidateVisitor* visitor, bool const ignoreExceptions = false) const;

    /**
     * Return the SpatialCellCandidate with the specified id
     *
     * @param id The desired ID
     * @param noThrow Return NULL in case of error
     *
     * @throws lsst::pex::exceptions::NotFoundError if no candidate matches the id (unless noThrow
     * is true, in which case a null pointer is returned
     */
    std::shared_ptr<SpatialCellCandidate> getCandidateById(int id, bool noThrow = false);

    /// Set whether we should omit BAD candidates from candidate list when traversing
    void setIgnoreBad(bool ignoreBad);

private:
    lsst::geom::Box2I _region;  // Bounding box of overall image
    CellList _cellList;         // List of SpatialCells
};
}  // namespace math
}  // namespace afw
}  // namespace lsst

#endif
