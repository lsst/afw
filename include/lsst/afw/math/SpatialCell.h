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

        SpatialCellCandidate(float const xCenter, ///< The object's column-centre
                             float const yCenter  ///< The object's row-centre
                    ) :
            _id(++_CandidateId),
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
    private:
        int _id;                        // Unique ID for object
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

        void insertCandidate(SpatialCellCandidate::Ptr candidate);

        bool nextCandidate();
        bool prevCandidate(bool first=false);
        SpatialCellCandidate::Ptr getCurrentCandidate();
        bool selectBestCandidate(bool fix=false);
        /**
         * Get SpatialCell's label
         */
        std::string const& getLabel() const { return _label; }
        /**
         * Get SpatialCell's BBox
         */
        lsst::afw::image::BBox const& getBBox() const { return _bbox; }

        bool isUsable() const;
    private:
        std::string _label;             // Name of cell for logging/trace
        lsst::afw::image::BBox _bbox;   // Bounding box of cell in overall image
        CandidateList _candidateList;   // List of candidates in the cell

        CandidateList::iterator _currentCandidate; // The current candidate in this Cell
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
