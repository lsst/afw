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
 * @brief Class to ensure constraints for spatial modeling
 *
 * @ingroup afw
 */

#ifndef LSST_AFW_MATH_SPATIALCELL_H
#define LSST_AFW_MATH_SPATIALCELL_H

#include <limits>
#include <vector>
#include <string>

#include <memory>
#include "lsst/base.h"
#include "lsst/pex/exceptions.h"
#include "lsst/afw/geom.h"
#include "lsst/afw/image/LsstImageTypes.h"

namespace lsst {
namespace afw {

// forward declarations
namespace image {
    template<typename ImagePixelT> class Image;
    template<typename ImagePixelT, typename MaskPixelT, typename VariancePixelT> class MaskedImage;
}

namespace math {

    /********************************************************************************************************/
    /// A class to pass around to all our Candidates
    class SpatialCellCandidate;

    class CandidateVisitor {
    public:
        CandidateVisitor() {}
        virtual ~CandidateVisitor() {}

        virtual void reset() {}
        virtual void processCandidate(SpatialCellCandidate *) {}
    };

    /************************************************************************************************************/
    /**
     * Base class for candidate objects in a SpatialCell
     */
    class SpatialCellCandidate {
    public:
        typedef std::shared_ptr<SpatialCellCandidate> Ptr;
        typedef std::shared_ptr<const SpatialCellCandidate> ConstPtr;

        enum Status {BAD = 0, GOOD = 1, UNKNOWN = 2};

        SpatialCellCandidate(float const xCenter, ///< The object's column-centre
                             float const yCenter  ///< The object's row-centre
                    ) :
            _id(++_CandidateId),
            _status(UNKNOWN),
            _xCenter(xCenter), _yCenter(yCenter) {
        }

        /**
         * (virtual) destructor -- this is a base class you know
         */
        virtual ~SpatialCellCandidate() {}

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
        void setStatus(Status status);
        /// Is this candidate unacceptable?
        virtual bool isBad() const {
            return (_status == BAD);
        }
    private:
        int _id;                        // Unique ID for object
        Status _status;                 // Is this Candidate good?
        float const _xCenter;           // The object's column-centre
        float const _yCenter;           // The object's row-centre

        static int _CandidateId;        // Unique identifier for candidates; useful for preserving current candidate
                                        // following insertion
    };

    /************************************************************************************************************/
    /**
     * Base class for candidate objects in a SpatialCell that are able to return an Image of some sort
     * (e.g. a PSF or a DIA kernel)
     */
    template<typename PixelT>
    class SpatialCellImageCandidate : public SpatialCellCandidate {
    public:
        typedef std::shared_ptr<SpatialCellImageCandidate> Ptr;
        typedef std::shared_ptr<const SpatialCellImageCandidate> ConstPtr;

        /// ctor
        SpatialCellImageCandidate(float const xCenter, ///< The object's column-centre
                                  float const yCenter  ///< The object's row-centre
                                 ) : SpatialCellCandidate(xCenter, yCenter),
                                     _image(PTR(lsst::afw::image::Image<PixelT>)()),
                                     _chi2(std::numeric_limits<double>::max()) {
        }
        virtual ~SpatialCellImageCandidate() {}

        /// Return the Candidate's Image
        virtual CONST_PTR(lsst::afw::image::Image<PixelT>) getImage() const = 0;

        /// Set the width of the image that getImage should return
        static void setWidth(int width) {
            _width = width;
        }
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

    protected:
        PTR(lsst::afw::image::Image<PixelT>) mutable _image; ///< a pointer to the Image, for the use of the base class
    private:
        static int _width;              // the width of images to return; may be ignored by subclasses
        static int _height;             // the height of images to return; may be ignored by subclasses
        double _chi2;                   // chi^2 for fit
    };

    /// The width of images that SpatialCellImageCandidate should return; may be ignored by subclasses
    template<typename PixelT>
    int SpatialCellImageCandidate<PixelT>::_width = 0;

    /// The height of images that SpatialCellImageCandidate should return; may be ignored by subclasses
    template<typename PixelT>
    int SpatialCellImageCandidate<PixelT>::_height = 0;

    /************************************************************************************************************/
    /**
     * Base class for candidate objects in a SpatialCell that are able to return a MaskedImage of some sort
     * (e.g. a PSF or a DIA kernel)
     */
    template<typename PixelT>
    class SpatialCellMaskedImageCandidate : public SpatialCellCandidate {
    public:
        typedef std::shared_ptr<SpatialCellMaskedImageCandidate> Ptr;
        typedef std::shared_ptr<const SpatialCellMaskedImageCandidate> ConstPtr;

        /// ctor
        SpatialCellMaskedImageCandidate(float const xCenter, ///< The object's column-centre
                                        float const yCenter  ///< The object's row-centre
                                       ) : SpatialCellCandidate(xCenter, yCenter),
                                           _image(PTR(lsst::afw::image::MaskedImage<PixelT,
                                                      lsst::afw::image::MaskPixel,
                                                      lsst::afw::image::VariancePixel>)()),
                                           _chi2(std::numeric_limits<double>::max()) {
        }
        virtual ~SpatialCellMaskedImageCandidate() {}

        /// Return the Candidate's Image
        virtual CONST_PTR(lsst::afw::image::MaskedImage<PixelT,lsst::afw::image::MaskPixel,
                          lsst::afw::image::VariancePixel>) getMaskedImage() const = 0;

        /// Set the width of the image that getImage should return
        static void setWidth(int width) {
            _width = width;
        }
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

    protected:
        PTR(lsst::afw::image::MaskedImage<PixelT,lsst::afw::image::MaskPixel,
            lsst::afw::image::VariancePixel>) mutable _image; ///< a pointer to the MaskedImage, for the use of the base class
    private:
        static int _width;              // the width of images to return; may be ignored by subclasses
        static int _height;             // the height of images to return; may be ignored by subclasses
        double _chi2;                   // chi^2 for fit
    };

    /// The width of images that SpatialCellMaskedImageCandidate should return; may be ignored by subclasses
    template<typename PixelT>
    int SpatialCellMaskedImageCandidate<PixelT>::_width = 0;

    /// The height of images that SpatialCellMaskedImageCandidate should return; may be ignored by subclasses
    template<typename PixelT>
    int SpatialCellMaskedImageCandidate<PixelT>::_height = 0;

    /************************************************************************************************************/
    /**
     * @brief An iterator that only returns usable members of the SpatialCell
     */
    class SpatialCellCandidateIterator {
        friend class SpatialCell;
        typedef std::vector<PTR(SpatialCellCandidate)> CandidateList;

    public:
        // ctors are protected
        void operator++();
        size_t operator-(SpatialCellCandidateIterator const& rhs) const;

        CONST_PTR(SpatialCellCandidate) operator*() const;
        PTR(SpatialCellCandidate)       operator*();

        /// Are two SpatialCellCandidateIterator%s equal?
        bool operator==(SpatialCellCandidateIterator const& rhs) const {
            return _iterator == rhs._iterator;
        }
        /// Are two SpatialCellCandidateIterator%s unequal?
        bool operator!=(SpatialCellCandidateIterator const& rhs) const {
            return _iterator != rhs._iterator;
        }

    protected:
        SpatialCellCandidateIterator(CandidateList::iterator iterator, CandidateList::iterator end, bool ignoreBad);
        SpatialCellCandidateIterator(CandidateList::iterator iterator, CandidateList::iterator end, bool ignoreBad, bool);

    private:
        CandidateList::iterator _iterator;
        CandidateList::iterator _end;
        bool _ignoreBad;
    };

    /************************************************************************************************************/
    /**
     * @brief Class to ensure constraints for spatial modeling
     *
     * A given %image is divided up into cells, with each cell represented by an instance of this class.
     * Each cell itself contains a list of instances of classes derived from SpatialCellCandidate.  One class
     * member from each cell will be chosen to fit to a spatial model.  In case of a poor fit, the next class
     * instance in the list will be fit for.  If all instances in a list are rejected from the spatial model,
     * the best one will be used.
     *
     * \sa \link SpatialCellSetExample\endlink
     */
    class SpatialCell {
    public:
        typedef std::shared_ptr<SpatialCell> Ptr;
        typedef std::shared_ptr<const SpatialCell> ConstPtr;
        typedef std::vector<PTR(SpatialCellCandidate)> CandidateList;
        typedef SpatialCellCandidateIterator iterator;
        /**
         * Constructor
         */
        SpatialCell(std::string const& label,
                    lsst::afw::geom::Box2I const& bbox=lsst::afw::geom::Box2I(),
                    CandidateList const& candidateList=CandidateList());

        /**
         * Destructor
         */
        virtual ~SpatialCell() {;};

        bool empty() const;
        size_t size() const;

        void sortCandidates();
        /**
         * Return an iterator to the beginning of the Candidates
         */
        SpatialCellCandidateIterator begin() {
            return SpatialCellCandidateIterator(_candidateList.begin(), _candidateList.end(), _ignoreBad);
        }
        SpatialCellCandidateIterator begin(bool ignoreBad ///< If true, ignore BAD candidates
                                          ) {
            return SpatialCellCandidateIterator(_candidateList.begin(), _candidateList.end(), ignoreBad);
        }
        /**
         * Return an iterator to (one after) the end of the Candidates
         */
        SpatialCellCandidateIterator end() {
            return SpatialCellCandidateIterator(_candidateList.begin(), _candidateList.end(), _ignoreBad, true);
        }
        SpatialCellCandidateIterator end(bool ignoreBad ///< If true, ignore BAD candidates
                                        ) {
            return SpatialCellCandidateIterator(_candidateList.begin(), _candidateList.end(), ignoreBad, true);
        }
        //
        void insertCandidate(PTR(SpatialCellCandidate) candidate);

        /// Remove a candidate from the list
        ///
        /// This is not a particularly efficient operation, since we're
        /// using a std::vector, but should not hurt too much if the number
        /// of candidates in a cell is small.
        void removeCandidate(PTR(SpatialCellCandidate) candidate);

        /// Set whether we should omit BAD candidates from candidate list when traversing
        void setIgnoreBad(bool ignoreBad) { _ignoreBad = ignoreBad; }
        /// Get whether we are omitting BAD candidates from candidate list when traversing
        bool getIgnoreBad() const { return _ignoreBad; }

        PTR(SpatialCellCandidate) getCandidateById(int id, bool noThrow=false);
        /**
         * Get SpatialCell's label
         */
        std::string const& getLabel() const { return _label; }
        /**
         * Get SpatialCell's BBox
         */
        lsst::afw::geom::Box2I const& getBBox() const { return _bbox; }
        /*
         * Visit our candidates
         */
        void visitCandidates(CandidateVisitor * visitor, int const nMaxPerCell=-1,
                             bool const ignoreExceptions=false, bool const reset=true);
        void visitCandidates(CandidateVisitor * visitor, int const nMaxPerCell=-1,
                             bool const ignoreExceptions=false, bool const reset=true) const;
        void visitAllCandidates(CandidateVisitor * visitor,
                                bool const ignoreExceptions=false, bool const reset=true);
        void visitAllCandidates(CandidateVisitor * visitor,
                                bool const ignoreExceptions=false, bool const reset=true) const;

    private:
        std::string _label;             // Name of cell for logging/trace
        lsst::afw::geom::Box2I _bbox;   // Bounding box of cell in overall image
        CandidateList _candidateList;   // List of all candidates in the cell
        bool _ignoreBad;                // Don't include BAD candidates when traversing the list
    };

    /**
     * @brief A collection of SpatialCells covering an entire %image
     */
    class SpatialCellSet {
    public:
        typedef std::shared_ptr<SpatialCellSet> Ptr;
        typedef std::shared_ptr<const SpatialCellSet> ConstPtr;

        typedef std::vector<PTR(SpatialCell)> CellList;

        SpatialCellSet(lsst::afw::geom::Box2I const& region, int xSize, int ySize=0);

        /**
         * Destructor
         */
        virtual ~SpatialCellSet() {;};

        /**
         * Return our SpatialCells
         */
        CellList& getCellList() { return _cellList; }

        /**
         * Return the bounding box of the %image
         */
        lsst::afw::geom::Box2I getBBox() const { return _region; };

        void insertCandidate(PTR(SpatialCellCandidate) candidate);

        void sortCandidates();

        void visitCandidates(CandidateVisitor * visitor, int const nMaxPerCell=-1,
                             bool const ignoreExceptions=false);
        void visitCandidates(CandidateVisitor * visitor, int const nMaxPerCell=-1,
                             bool const ignoreExceptions=false) const;
        void visitAllCandidates(CandidateVisitor * visitor, bool const ignoreExceptions=false);
        void visitAllCandidates(CandidateVisitor * visitor, bool const ignoreExceptions=false) const;

        PTR(SpatialCellCandidate) getCandidateById(int id, bool noThrow=false);

        void setIgnoreBad(bool ignoreBad);

    private:
        lsst::afw::geom::Box2I _region;   // Bounding box of overall image
        CellList _cellList;               // List of SpatialCells
    };
}}}

#endif
