// -*- lsst-c++ -*-
/**
 * @file
 *
 * @brief Class to ensure constraints for spatial modeling
 *
 * @author Andrew Becker, University of Washington
 *
 * @ingroup afw
 */

#ifndef LSST_AFW_MATH_SPATIALCELL_H
#define LSST_AFW_MATH_SPATIALCELL_H

#include <limits>
#include <vector>
#include <string>

#include "boost/shared_ptr.hpp"
#include "lsst/pex/exceptions.h"

namespace lsst {
namespace afw {
namespace math {

    /************************************************************************************************************/
    /**
     * Base class for candidate objects in a SpatialCell
     */
    class SpatialCellCandidate {
    public:
        typedef boost::shared_ptr<SpatialCellCandidate> Ptr;
        typedef boost::shared_ptr<const SpatialCellCandidate> ConstPtr;

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

        /// Return candidates rating
        virtual double getCandidateRating() const = 0;

        /// Return the candidate's unique ID
        int getId() const { return _id; }
        /// Return the candidate's status
        Status getStatus() const { return _status; }
        /// Set the candidate's status
        void setStatus(Status status) {
            switch (status) {
              case GOOD:
              case BAD:
              case UNKNOWN:
                _status = status;
                return;
            }

            throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException,
                              (boost::format("Saw unknown status %d") % status).str());
        }
        /// Is this candidate acceptable?
        virtual operator bool() const {
            return (_status != BAD);
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
     * Base class for candidate objects in a SpatialCell that are able to return an %image of some sort
     * (e.g. a PSF or a DIA kernel)
     */
    template<typename ImageT>
    class SpatialCellImageCandidate : public SpatialCellCandidate {
    public:
        typedef boost::shared_ptr<SpatialCellImageCandidate> Ptr;
        typedef boost::shared_ptr<const SpatialCellImageCandidate> ConstPtr;

        SpatialCellImageCandidate(float const xCenter, ///< The object's column-centre
                                  float const yCenter  ///< The object's row-centre
                                 ) : SpatialCellCandidate(xCenter, yCenter),
                                     _image(typename ImageT::Ptr()) {
        }

        /** Return the Candidate's Image */
        virtual typename ImageT::ConstPtr getImage() const = 0;

        /** Set the width of the image that getImage should return */
        void setWidth(int width) { _width = width; }
        /** Return the width of the image that getImage should return */
        int getWidth() const { return _width; }

        /** Set the height of the image that getImage should return */
        void setHeight(int height) { _height = height; }
        /** Return the height of the image that getImage should return */
        int getHeight() const { return _height; }

    protected:
        typename ImageT::Ptr mutable _image; ///< a pointer to the %image, for the use of the base class
    private:
        static int _width;              // the width of images to return; may be ignored by subclasses
        static int _height;             // the height of images to return; may be ignored by subclasses
    };

    /// The width of images that SpatialCellImageCandidate should return; may be ignored by subclasses
    template<typename ImageT>
    int SpatialCellImageCandidate<ImageT>::_width = 0;

    /// The height of images that SpatialCellImageCandidate should return; may be ignored by subclasses
    template<typename ImageT>
    int SpatialCellImageCandidate<ImageT>::_height = 0;

    /************************************************************************************************************/
    /**
     * @brief An iterator that only returns usable members of the SpatialCell
     */
    class SpatialCellCandidateIterator {
        friend class SpatialCell;
        typedef std::vector<SpatialCellCandidate::Ptr> CandidateList;

    public:
        // ctors are protected
        void operator++();
        size_t operator-(SpatialCellCandidateIterator const& rhs);

        SpatialCellCandidate::ConstPtr operator*() const;
        SpatialCellCandidate::Ptr      operator*();

        /// Are two SpatialCellCandidateIterator%s equal?
        bool operator==(SpatialCellCandidateIterator const& rhs) {
            return _iterator == rhs._iterator;
        }
        /// Are two SpatialCellCandidateIterator%s unequal?
        bool operator!=(SpatialCellCandidateIterator const& rhs) {
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
     * A given %image is be divided up into cells, with each cell represented by an instance of this class.
     * Each cell itself contains a list of instances of classes derived from SpatialCellCandidate.  One class
     * member from each cell will be chosen to fit to a spatial model.  In case of a poor fit, the next class
     * instance in the list will be fit for.  If all instances in a list are rejected from the spatial model,
     * the best one will be used.
     */
    class SpatialCell {
    public:
        typedef boost::shared_ptr<SpatialCell> Ptr;
        typedef boost::shared_ptr<const SpatialCell> ConstPtr;
        typedef std::vector<SpatialCellCandidate::Ptr> CandidateList;
        typedef SpatialCellCandidateIterator iterator;
        /**
         * Constructor
         */
        SpatialCell(std::string const& label,
                    lsst::afw::image::BBox const& bbox=lsst::afw::image::BBox(),
                    CandidateList const& candidateList=CandidateList());
        
        /**
         * Destructor
         */
        virtual ~SpatialCell() {;};

        bool empty() const;
        size_t size() const;

        /**
         * Return an iterator to the beginning of the Candidates
         */
        SpatialCellCandidateIterator begin() {
            return SpatialCellCandidateIterator(_candidateList.begin(), _candidateList.end(), _ignoreBad);
        }
        SpatialCellCandidateIterator begin(bool ignoreBad) {
            return SpatialCellCandidateIterator(_candidateList.begin(), _candidateList.end(), ignoreBad);
        }
        SpatialCellCandidateIterator end() {
            return SpatialCellCandidateIterator(_candidateList.begin(), _candidateList.end(), _ignoreBad, true);
        }
        SpatialCellCandidateIterator end(bool ignoreBad) {
            return SpatialCellCandidateIterator(_candidateList.begin(), _candidateList.end(), ignoreBad, true);
        }
        //
        void insertCandidate(SpatialCellCandidate::Ptr candidate);
        /// Set whether we should omit BAD candidates from candidate list when traversing
        void setIgnoreBad(bool ignoreBad) { _ignoreBad = ignoreBad; }
        /// Get whether we are omitting BAD candidates from candidate list when traversing
        bool getIgnoreBad() const { return _ignoreBad; }
        /**
         * Get SpatialCell's label
         */
        std::string const& getLabel() const { return _label; }
        /**
         * Get SpatialCell's BBox
         */
        lsst::afw::image::BBox const& getBBox() const { return _bbox; }
    private:
        std::string _label;             // Name of cell for logging/trace
        lsst::afw::image::BBox _bbox;   // Bounding box of cell in overall image
        CandidateList _candidateList;   // List of candidates in the cell
        bool _ignoreBad;                // Don't include BAD candidates when traversing the list
    };

    /** 
     * @brief A collection of SpatialCells covering an entire %image
     */
    class SpatialCellSet {
    public:
        typedef boost::shared_ptr<SpatialCellSet> Ptr;
        typedef boost::shared_ptr<const SpatialCellSet> ConstPtr;
        
        typedef std::vector<SpatialCell::Ptr> CellList;

        SpatialCellSet(lsst::afw::image::BBox const& region, int xSize, int ySize=0);
        
        /**
         * Destructor
         */
        virtual ~SpatialCellSet() {;};

        /**
         * Return our SpatialCells
         */
        CellList& getCellList() { return _cellList; }

        void insertCandidate(SpatialCellCandidate::Ptr candidate);
    private:
        lsst::afw::image::BBox _region;   // Dimensions of overall image
        CellList _cellList;               // List of SpatialCells
    };
}}}

#endif
