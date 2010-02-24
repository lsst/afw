// -*- LSST-C++ -*- // fixed format comment for emacs
/**
* @file
*
* @brief Declaration of the sky pixelization classes.
*
* Basic concepts:
* * SkyMapScheme: describes a sky pixelization scheme, such as Healpix.
* * PixelId: the unique ID of a pixel. The ID type is a template parameter and depends on
*   the pixelization scheme. PixelId must support <, > and == operators.
* * PixelData: the data for (value of) a sky map pixel. The data type is a template parameter.
* * SkyMapPixel: a pair <Pixel ID, Pixel Data> describing the location and value of a pixel in the sky map.
* * SkyMapIdSet: a collection of pixel IDs.
* * SkyMapImage: a collection of Pixels (Pixel ID, Pixel Data pairs).
*
* @todo 
* * move implementation to src file
* * Refactor to avoid having healpix names pollute the namespace; one solution is use
*   pointer-to-implementation for the HealPixMapScheme class.
* * Add clone methods to SkyMapIdSet and SkyMapImage.
*
* @ingroup afw
*/

#ifndef LSST_AFW_IMAGE_SKYMAP_H
#define LSST_AFW_IMAGE_SKYMAP_H

#include <algorithm>
#include <utility>
#include <vector>
#include <cmath>

#include "boost/shared_ptr.hpp"
#include "boost/cstdint.hpp"
#include "healpix_base2.h"

#include "lsst/pex/exceptions.h"
#include "lsst/afw/geom.h"

namespace lsst {
namespace afw {

namespace formatters {
}

namespace image {

    // forward declarations
    template <typename PixelIdT>
    class SkyMapScheme;
    template <typename PixelIdT, typename PixelDataT>
    class SkyMapImage;
    
    /**
    * @brief A sky map pixel, consisting of a const pixel Id and mutable pixel data.
    */
    template<typename PixelIdT, typename PixelDataT>
    class SkyMapPixel {
    public:
        typedef PixelIdT PixelId;
        typedef PixelDataT PixelData;
        
        SkyMapPixel(
                PixelIdT const &pixelId     ///< pixel ID
        ) :
            _id(pixelId),
            _data()
        { }

        SkyMapPixel(
                PixelIdT const &pixelId,    ///< pixel ID
                PixelDataT const &pixelData ///< pixel data
        ) :
            _id(pixelId),
            _data(pixelData)
        { }
        
        /**
        * @brief Get the pixel ID
        */
        PixelIdT getId() const { return _id; };
        
        /**
        * @brief Get the pixel data
        */
        PixelDataT getData() const { return _data; };

        /**
        * @brief Set the pixel data
        */
        void setData(PixelDataT const &pixelData) { _data = pixelData; };

    private:
        PixelIdT const _id;
        PixelDataT _data;
    };

    /**
    * @brief A collection of sky map pixels IDs
    *
    * @note uses the default copy constructor
    */
    template<typename PixelIdT>
    class SkyMapIdSet {
    public:
        typedef typename SkyMapScheme<PixelIdT>::ConstPtr SchemeConstPtr;
        typedef PixelIdT PixelId;
        typedef boost::shared_ptr<SkyMapIdSet> Ptr;
        typedef boost::shared_ptr<SkyMapIdSet const> ConstPtr;
        typedef typename std::vector<PixelIdT>::const_iterator ConstIterator;

        /**
        * @brief Construct a SkyMapIdSet from a scheme and list of pixel IDs
        */
        explicit SkyMapIdSet(
                SkyMapScheme<PixelIdT> const &scheme, ///< sky pixelization scheme
                std::vector<PixelIdT> const &idList ///< list of sky map pixel IDs
        ) :
            _schemePtr(scheme.clone()),
            _idList(idList)
        {
            _normalize();
        }
        
        virtual ~SkyMapIdSet() {};

        /**
        * @brief Is the pixel present?
        */
        inline bool contains(
                PixelId const &pixelId ///< pixel ID
        ) const {
            return find(pixelId) != end();
        }

        /**
        * @brief Find the specified pixel ID; return end() if not found.
        */
        // Use binary search since data is sorted          
        ConstIterator find(PixelId const &pixelId) {
            ConstIterator firstIter = begin();
            ConstIterator lastIter = end() - 1;
            while (firstIter <= lastIter) {
                ConstIterator ctrIter = firstIter + ((lastIter - firstIter) / 2);
                if (pixelId > *ctrIter) {
                    firstIter = ctrIter + 1;
                } else if (pixelId < *ctrIter) {
                    lastIter = ctrIter - 1;
                } else {
                    return ctrIter;
                }
            }
            return end();
        }

        SchemeConstPtr getScheme() const { return _schemePtr; };

        /**
        * @brief Return the number of elements
        */
        boost::int64_t getSize() const {
            return _idList.size();
        };

        ConstIterator begin() const { return _idList.begin(); };

        ConstIterator end() const { return _idList.end(); };
        
        template <typename A, typename B> friend class SkyMapImage;

    private:
        /**
        * @brief Sort the data and remove duplicates
        */
        void _normalize() {
            std::sort(_idList.begin(), _idList.end());
            std::unique(_idList.begin(), _idList.end());
        }
        SchemeConstPtr _schemePtr;
        std::vector<PixelId> _idList;
    };


    typedef boost::int64_t HealPixId; ///< use as a template parameter PixelId for HealPix maps

    /**
    * @brief Describes a sky pixelization scheme and provides basic methods for accessing pixel IDs
    *
    * An abstract base class. Subclasses provide all of the implementation.
    */
    template <typename PixelIdT>
    class SkyMapScheme {
    public:
        virtual ~SkyMapScheme() {};

        typedef typename boost::shared_ptr<SkyMapScheme const> ConstPtr;
        typedef PixelIdT PixelId;
        typedef SkyMapIdSet<PixelIdT> IdSet;
        
        /**
        * @brief return a shared pointer to a copy
        *
        * Using a shared pointer prevents type slicing.
        */
        virtual ConstPtr clone() const = 0;

        /**
        * @brief find all indices whose centers lie within the specified polygon
        */
        virtual SkyMapIdSet<PixelIdT> findIndicesInPolygon(
            std::vector<lsst::afw::geom::Point2D> const &vertexList ///< list of RA/Dec vertices (radians)
        ) const = 0;

        /**
        * @brief find the indices of all pixels whose centers lie with a given radius of a point on the sky
        */
        virtual SkyMapIdSet<PixelIdT> findIndicesInDisc(
            lsst::afw::geom::Point2D raDec, ///< RA/Dec center point (radians)
            double radius ///< radius (radians)
        ) const = 0;

        /**
        * @brief return the area of the specified pixel (in steradians)
        *
        * For sky pixelization schemes with equal-area pixels the argument is ignored.
        */
        virtual double getPixelArea(
                PixelIdT const &pixelId ///< pixel ID
        ) const = 0;
        
        /**
        * @brief return the total number of pixels in an all-sky map
        */
        virtual boost::int64_t getTotalPixels() const = 0;

        /**
        * @brief return on-sky position of center of pixel
        */
        virtual lsst::afw::geom::Point2D getSkyPosition(
                PixelIdT const &pixelId ///< pixel ID
        ) const = 0;

        /**
        * @brief get on-sky position of pixel corners
        */
        virtual PixelIdT getPixelId(
                lsst::afw::geom::Point2D const &raDec ///< RA/Dec (radians)
        ) const = 0;

        /**
        * @brief Are the schemes equal?
        *
        * Schemes are considered equal if and only if they are the same type and have the same parameters.
        * (It is not sufficient for one to be a subclass of the other because == must commute).
        */
        virtual bool operator==(
                SkyMapScheme const &rhs ///< right-hand-side sky map scheme
        ) const = 0;

        /**
        * @brief Are the schemes not equal?
        *
        * See operator== for the definition of equal.
        */
        virtual bool operator!=(
                SkyMapScheme const &rhs ///< right-hand-side sky map scheme
        ) const { return !(*this == rhs); };
    };


    class HealPixMapScheme: public SkyMapScheme<HealPixId> {
    public:
        typedef SkyMapScheme<HealPixId>::PixelId PixelId;
        typedef SkyMapScheme<HealPixId>::IdSet IdSet;

        explicit HealPixMapScheme(
                boost::int64_t nSides ///< number of sides for HEALPix map
        ) :
            _healPixBase(nSides, RING, SET_NSIDE)
        { }

        virtual ~HealPixMapScheme() {};

        virtual SkyMapScheme<HealPixId>::ConstPtr clone() const {
            return SkyMapScheme<HealPixId>::ConstPtr(new HealPixMapScheme(getNSides()));
        }

        virtual SkyMapIdSet<HealPixId> findIndicesInPolygon(
                std::vector<lsst::afw::geom::Point2D> const &vertexList
        ) const {
            // adapt the HEALPix FORTRAN code since query_polygon is not available in C++
            throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException, "Not yet implemented");
        }

        virtual SkyMapIdSet<HealPixId> findIndicesInDisc(
                lsst::afw::geom::Point2D raDec, ///< RA/Dec center point (radians)
                double radius   ///< radius (radians)
        ) const {
            std::vector<PixelId> pixelList;
            throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException, "Not yet implemented");
            // the following code should work, but query_disc is missing from the healpix library
//             this->_healPixBase.query_disc(_pointingFromRaDec(raDec), radius, pixelList);
//             IdSet retIdSet(*this);
//             for (std::vector<PixelId>::const_iterator pixPtr = pixelList.begin();
//                 pixPtr != pixelList.end(); ++pixPtr) {
//                 retIdSet += *pixPtr;
//             }
//             return retIdSet;
        }

        virtual double getPixelArea(HealPixId const & /* pixelId */) const {
            return (4.0 * M_PI) / double(_healPixBase.Npix());
        }

        virtual boost::int64_t getTotalPixels() const { return _healPixBase.Npix(); };

        virtual lsst::afw::geom::Point2D getSkyPosition(HealPixId const &pixelId) const {
            return _raDecFromPointing(_healPixBase.pix2ang(pixelId));
        }

        virtual HealPixId getPixelId(lsst::afw::geom::Point2D const &raDec) const {
            return _healPixBase.ang2pix(_pointingFromRaDec(raDec));
        }
        
        /**
        * @brief return the HEALPix-specific nSides parameter
        */
        boost::int64_t getNSides() const { return _healPixBase.Nside(); };

        virtual bool operator==(SkyMapScheme<PixelId> const &rhs) const {
            /* Test typeid because a subclass will not do. Alternatively, one could trap an exception
            in the dynamic_cast of a reference, but exceptions are potentially expensive.
            */
            if (typeid(*this) != typeid(rhs)) return false;
            /* The types match so cast the reference. If there is a bug such that the cast fails, that throws
            * an exception, which is better than dereferencing a null pointer from failing to cast a pointer.
            */
            HealPixMapScheme const &rhsCastRef = dynamic_cast<HealPixMapScheme const &>(rhs);
            return ((this->getNSides() == rhsCastRef.getNSides()) &&
                (this->_healPixBase.Scheme() == rhsCastRef._healPixBase.Scheme()));
        }
    
    private:
        Healpix_Base2 _healPixBase;
        
        /**
        * @brief convert an RA/Dec to a HEALPix pointing object
        */
        inline static pointing _pointingFromRaDec(lsst::afw::geom::Point2D const &raDec) {
            return pointing((M_PI / 2.0) - raDec[1], raDec[0]);
        }

        /**
        * @brief convert a HEALPix pointing object to an RA/Dec
        */
        inline static lsst::afw::geom::Point2D _raDecFromPointing(pointing const &ptg) {
            return lsst::afw::geom::makePointD(ptg.phi, (M_PI / 2.0) - ptg.theta);
        }
    };

    /**
    * @brief A collection of sky map pixels (pixel ID/data pairs)
    *
    * @note uses the default copy constructor
    */
    template <typename PixelIdT, typename PixelDataT>
    class SkyMapImage {
    public:
        typedef typename SkyMapScheme<PixelIdT>::ConstPtr SchemeConstPtr;
        typedef SkyMapIdSet<PixelIdT> IdSet;
        typedef PixelIdT PixelId;
        typedef PixelDataT PixelData;
        typedef SkyMapPixel<PixelIdT, PixelDataT> Pixel;
        typedef boost::shared_ptr<SkyMapImage> Ptr;
        typedef boost::shared_ptr<SkyMapImage const> ConstPtr;
        typedef typename std::vector<Pixel>::const_iterator ConstIterator;
        typedef typename std::vector<Pixel>::iterator Iterator;
        
        explicit SkyMapImage(
                SkyMapIdSet<PixelIdT> const &idSet ///< sky map ID set
        ) :
            _schemePtr(idSet.getScheme())
        {
            PixelData const NullPixel(0);
            for (typename IdSet::ConstIterator idIter = idSet.begin(); idIter != idSet.end(); ++idIter) {
                _pixelSet.push_back(Pixel(*idIter, NullPixel));
            }
        }

        virtual ~SkyMapImage() {};

        SchemeConstPtr getScheme() const { return _schemePtr; };

        /**
        * @brief Is the pixel present?
        */
        inline bool contains(
                PixelId const &pixelId ///< pixel ID
        ) const {
            return find(pixelId) != end();
        }

        /**
        * @brief Find specified pixel by pixelId; return end() if not found.
        */
        Iterator find(
                PixelId const &pixelId ///< pixel ID
        ) {
            Iterator firstIter = _pixelSet.begin();
            Iterator lastIter = _pixelSet.end() - 1;
            while (firstIter <= lastIter) {
                Iterator ctrIter = firstIter + ((lastIter - firstIter) / 2);
                if (pixelId > ctrIter->getId()) {
                    firstIter = ctrIter + 1;
                } else if (pixelId > ctrIter->getId()) {
                    lastIter = ctrIter - 1;
                } else {
                    return ctrIter;
                }
            }
            return end();
        }
        
        /**
        * @brief Return the value of the specified pixel
        *
        * @raise lsst::pex::exceptions::NotFoundException if pixel is not present
        */
        PixelDataT get(
                PixelId const &pixelId ///< pixel ID
        ) const {
            // Use binary search because pixel indices are sorted
            ConstIterator pixelIter = find(pixelId);
            if (pixelIter == end()) {
                throw LSST_EXCEPT(lsst::pex::exceptions::NotFoundException, "Not found");
            }
            return pixelIter->getData();
        }
        
        /**
        * @brief Set the value of a specified pixel
        *
        * @raise lsst::pex::exceptions::NotFoundException if pixel is not present
        */
        void set(
                PixelId const &pixelId, ///< pixel ID
                PixelData const &pixelData ///< pixel value
        ) {
            Iterator pixelIter = find(pixelId);
            if (pixelIter == end()) {
                throw LSST_EXCEPT(lsst::pex::exceptions::NotFoundException, "Not found");
            }
            pixelIter->setData(pixelData);
        }

        /**
        * @brief Return the IDSet for the pixels
        */
        IdSet getIdSet() const {
            // no need to normalize the result since SkyMapImage is already normalized
            IdSet retIdSet(_schemePtr);
            for (ConstIterator pixelPtr = begin(); pixelPtr != end(); ++pixelPtr) {
                retIdSet._idList.push_back(pixelPtr->getId());
            }
            return retIdSet;
        }
        
        /**
        * @brief Return the number of elements
        */
        boost::int64_t getSize() const {
            return _pixelSet.size();
        };

        Iterator begin() const { return _pixelSet.begin(); };

        Iterator end() const { return _pixelSet.end(); };
        
    private:
        SchemeConstPtr _schemePtr;
        std::vector<Pixel> _pixelSet;
    };

}}} // namespace lsst::afw::image

#endif // LSST_AFW_IMAGE_SKYMAP_H
