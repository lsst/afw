// -*- LSST-C++ -*- // fixed format comment for emacs
/**
* @file
*
* @brief Declaration of the sky pixelization classes.
*
* Basic concepts:
* * SkyMapScheme: describes a sky pixelization scheme, such as Healpix.
* * Pixel Id: the unique ID of a pixel. The ID type is a template parameter and depends on
*   the pixelization scheme.
* * Pixel Data: the data for (value of) a pixel. The data type is a template parameter.
* * Pixel: a pair <Pixel ID, Pixel Data> describing the location and value of a pixel in the sky map.
* * SkyMapIdSet: a collection of pixel IDs.
* * SkyMapImage: a collection of Pixels (Pixel ID, Pixel Data pairs).
*
* @todo 
* * Add operator== and operator!= to SkyMapIdSet and perhaps SkyMapImage.
* * move implementation to src file
* * Refactor to avoid having healpix names pollute the namespace; one solution is use
*   pointer-to-implementation for the HealPixMapScheme class.
* * Add clone methods to SkyMapIdSet and SkyMapImage.
* * Figure out why invalid nside causes an abort, not just an ordinary exception,
*   and do what you can to work around the problem and make the healpix interface more robust.
*
* @ingroup afw
*/

#ifndef LSST_AFW_IMAGE_SKYMAP_H
#define LSST_AFW_IMAGE_SKYMAP_H

#include <map>
#include <set>
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

    // forward declaration
    template <typename PixelIdT>
    class SkyMapScheme;

    /**
    * @brief A collection of sky map pixels IDs
    *
    * @note uses the default copy constructor
    */
    template<typename PixelIdT>
    class SkyMapIdSet {
    public:
        typedef SkyMapScheme<PixelIdT> Scheme;
        typedef PixelIdT PixelId;
        typedef typename std::set<PixelIdT>::const_iterator const_iterator;
        typedef typename std::set<PixelIdT>::iterator iterator;

        explicit SkyMapIdSet(
            SkyMapScheme<PixelIdT> const &scheme ///< sky pixelization scheme
        ) :
            _schemePtr(scheme.clone()),
            _idList()
        { }
        
        virtual ~SkyMapIdSet() {};

        typename SkyMapScheme<PixelIdT>::Ptr const getScheme() const { return _schemePtr; };

        /**
        * @brief Add the specified pixel; ignored if a already present
        */
        inline void operator+=(PixelId const &pixelId) { _idList.insert(pixelId); }

        /**
        * @brief Add the specified list of pixels; pixels that are already present are ignored
        */
        void operator+=(SkyMapIdSet const &idList) { _idList.insert(idList.begin(), idList.end()); }

        /**
        * @brief Remove the specified pixel; a no-op if the pixel is absent.
        */
        inline void operator-=(PixelId const &pixelId) { _idList.erase(pixelId); }

        /**
        * @brief Remove the specified list of pixels; pixels that are absent are ignored.
        */
        void operator-=(SkyMapIdSet const &idList) {
            for (const_iterator inPtr = idList.begin(); inPtr != idList.end(); ++inPtr) {
                _idList.erase(*inPtr);
            }
        }

        /**
        * @brief Is the pixel present?
        */
        inline bool contains(PixelId const &pixelId) const { return _idList.count(pixelId) > 0; }
        
        /**
        * @brief Return the number of elements
        */
        boost::int64_t getSize() const {
            return _idList.size();
        };
        
        void clear() { _idList.clear(); };

        const_iterator begin() const { return _idList.begin(); };

        const_iterator end() const { return _idList.end(); };

        iterator begin() { return _idList.begin(); };

        iterator end() { return _idList.end(); };

    private:
        typename Scheme::Ptr const _schemePtr;
        std::set<PixelId> _idList;
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

        typedef typename boost::shared_ptr<SkyMapScheme<PixelIdT> > Ptr;
        typedef PixelIdT PixelId;
        typedef SkyMapIdSet<PixelIdT> IdSet;
        
        /**
        * @brief return a shared pointer to a copy
        *
        * Using a shared pointer prevents type slicing.
        */
        virtual typename SkyMapScheme<PixelIdT>::Ptr clone() const = 0;

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
        virtual double getPixelArea(PixelIdT const &pixelId) const = 0;
        
        /**
        * @brief return the total number of pixels in an all-sky map
        */
        virtual boost::int64_t getTotalPixels() const = 0;

        /**
        * @brief return on-sky position of center of pixel
        */
        virtual lsst::afw::geom::Point2D getPixelPosition(PixelIdT const &pixelId) const = 0;

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
        virtual bool operator==(SkyMapScheme const &rhs) const = 0;

        /**
        * @brief Are the schemes not equal?
        *
        * See operator== for the definition of equal.
        */
        virtual bool operator!=(SkyMapScheme const &rhs) const { return !(*this == rhs); };
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

        virtual SkyMapScheme<HealPixId>::Ptr clone() const {
            return SkyMapScheme<HealPixId>::Ptr(new HealPixMapScheme(getNSides()));
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

        virtual lsst::afw::geom::Point2D getPixelPosition(HealPixId const &pixelId) const {
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
        typedef SkyMapScheme<PixelIdT> Scheme;
        typedef SkyMapIdSet<PixelIdT> IdSet;
        typedef PixelIdT PixelId;
        typedef PixelDataT PixelData;
        typedef std::pair<PixelIdT, PixelDataT> Pixel;
        typedef typename std::map<PixelIdT, PixelDataT>::const_iterator const_iterator;
        typedef typename std::map<PixelIdT, PixelDataT>::iterator iterator;
        
        explicit SkyMapImage(
                SkyMapScheme<PixelIdT> const &scheme ///< sky map scheme
        ) :
            _schemePtr(scheme.clone()),
            _pixelMap()
        { }

        virtual ~SkyMapImage() {};

        typename SkyMapScheme<PixelIdT>::Ptr const getScheme() const { return _schemePtr; };
        
        /**
        * @brief Return a the IDs of the contained pixels
        */
        SkyMapIdSet<PixelIdT> getIdSet() const {
            IdSet retIdSet;
            for (const_iterator pixelPtr = begin(); pixelPtr != end(); ++pixelPtr) {
                retIdSet += pixelPtr->first;
            }
            return retIdSet;
        }

        /**
        * @brief Add the specified pixel; ignored if a already present
        */
        inline void operator+=(Pixel const &pixel) { _pixelMap.insert(pixel); }

        /**
        * @brief Add the specified list of pixels; pixels that are already present are ignored
        *
        * @throw lsst::pex::exceptions::InvalidParameterException if the schemes do not match
        */
        void operator+=(SkyMapImage const &skyMapImage) {
            if (skyMapImage.getScheme() != this->getScheme()) {
               throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException, "Schemes do not match");
            }
            _pixelMap.insert(skyMapImage.begin(), skyMapImage.end());
        }

        /**
        * @brief Remove the specified pixel; a no-op if the pixel is absent.
        */
        inline void operator-=(PixelId const &pixelId) { _pixelMap.erase(pixelId); }

        /**
        * @brief Remove the specified list of pixels; pixels that are absent are ignored.
        *
        * @throw lsst::pex::exceptions::InvalidParameterException if the schemes do not match
        */
        void operator-=(SkyMapIdSet<PixelIdT> const &idList) {
            if (idList.getScheme() != this->getScheme()) {
               throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException, "Schemes do not match");
            }
            for (typename IdSet::const_iterator inPtr = idList.begin(); inPtr != idList.end(); ++inPtr) {
                _pixelMap.erase(*inPtr);
            }
        }

        /**
        * @brief Is the pixel present?
        */
        inline bool contains(PixelId const &pixelId) const { return _pixelMap.count(pixelId) > 0; }
        
        /**
        * @brief Return the value of the specified pixel
        *
        * @raise lsst::pex::exceptions::NotFoundException if pixel is not present
        */
        PixelDataT get(PixelId const &pixelId) const {
            Pixel *pixelPtr = _pixelMap.find(pixelId);
            if (pixelPtr == end()) {
                throw LSST_EXCEPT(lsst::pex::exceptions::NotFoundException, "Not found");
            }
            return pixelPtr->second;
        };
        
        /**
        * @brief Set the value of a specified pixel
        *
        * @raise lsst::pex::exceptions::NotFoundException if pixel is not present
        */
        void set(PixelId const &pixelId, PixelData const &pixelData) {
            Pixel *pixelIter = _pixelMap.find(pixelId);
            if (pixelIter == end()) {
                throw LSST_EXCEPT(lsst::pex::exceptions::NotFoundException, "Not found");
            }
            pixelIter->second = pixelData;
        }
        
        /**
        * @brief Return the number of elements
        */
        boost::int64_t getSize() const {
            return _pixelMap.size();
        };
        
        void clear() { _pixelMap.clear(); };

        const_iterator begin() const { return _pixelMap.begin(); };

        const_iterator end() const { return _pixelMap.end(); };

        iterator begin() { return _pixelMap.begin(); };

        iterator end() { return _pixelMap.end(); };
        
        /**
        * @brief Make a sky map pixel
        */
        Pixel static makePixel(PixelId const &pixelId, PixelDataT const &pixelData) {
            return Pixel(pixelId, pixelData);
        }
        
    private:
        typename Scheme::Ptr const _schemePtr;
        std::map<PixelId, PixelDataT> _pixelMap;
    };

}}} // namespace lsst::afw::image

#endif // LSST_AFW_IMAGE_SKYMAP_H
