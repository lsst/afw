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
* * SkyMapIdList: a collection of pixel IDs.
* * SkyMapImage: a collection of Pixels (Pixel ID, Pixel Data pairs).
*
* \todo 
* * Add operator== and operator!= to scheme; any sort of comparing or combining SkyMapIdLists
*   and/or SkyMapDataLists requires that the schemes be the same.
*   To implement this first test that the scheme classes are the same,
*   then that the parameters are the same (which may require a separate implementation in each subclass).
* * Add operator== and operator!= to SkyMapIdList and perhaps SkyMapImage.
* * move implementation to src file
* * Try to figure out how to avoid having healpix names pollute the namespace;
*   I'm not sure this is practical since HealPixMapScheme lists it as a member variable
* * Add clone methods to SkyMapIdList and SkyMapImage (once the current discussion calms down)
* * Handle invalid nside without letting healpix abort (sigh).
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
    * \brief A collection of sky map pixels IDs
    */
    template<typename SkyMapSchemeT>
    class SkyMapIdList {
    public:
        typedef typename SkyMapSchemeT::PixelId PixelId;
        typedef typename std::set<PixelId>::const_iterator const_iterator;
        typedef typename std::set<PixelId>::iterator iterator;

        explicit SkyMapIdList(
            SkyMapSchemeT const &scheme ///< sky pixelization scheme
        ) :
            _schemePtr(scheme.clone()),
            _idList()
        { }
        
        virtual ~SkyMapIdList() {};

        typename SkyMapSchemeT::Ptr getScheme() { return _schemePtr; };

        /**
        * \brief Add the specified pixel; ignored if a already present
        */
        inline void add(PixelId const &pixelId) { _idList.insert(pixelId); }

        /**
        * \brief Add the specified list of pixels; pixels that are already present are ignored
        */
        void add(SkyMapIdList const &idList) { _idList.insert(idList.begin(), idList.end()); }

        /**
        * \brief Is the pixel present?
        */
        inline bool contains(PixelId const &pixelId) const { return _idList.count(pixelId) > 0; }

        /**
        * \brief Remove the specified pixel; a no-op if the pixel is absent.
        */
        inline void remove(PixelId const &pixelId) { _idList.erase(pixelId); }

        /**
        * \brief Remove the specified list of pixels; pixels that are absent are ignored.
        */
        void remove(SkyMapIdList const &idList) {
            for (const_iterator inPtr = idList.begin(); inPtr != idList.end(); ++inPtr) {
                _idList.erase(*inPtr);
            }
        }

        const_iterator begin() const { return _idList.begin(); };

        const_iterator end() const { return _idList.end(); };

        iterator begin() { return _idList.begin(); };

        iterator end() { return _idList.end(); };

//         SkyMapIdList union(SkyMapIdList)
//         SkyMapIdList intersection(SkyMapIdList)
//         SkyMapIdList difference(SkyMapIdList) # elements not in dest, not in src
//         SkyMapIdList isSubSet(SkyMapIdList)
//         SkyMapIdList isSuperSet(SkyMapIdList)

    private:
        typename SkyMapSchemeT::Ptr _schemePtr;
        std::set<PixelId> _idList;
    };


    typedef boost::int64_t HealPixId; ///< use as a template parameter PixelId for HealPix maps

    /**
    * \brief Describes a sky pixelization scheme and provides basic methods for accessing pixel IDs
    *
    * An abstract base class. Subclasses provide all of the implementation.
    */
    template <typename PixelIdT>
    class SkyMapScheme {
    public:
        virtual ~SkyMapScheme() {};

        typedef typename boost::shared_ptr<SkyMapScheme<PixelIdT> > Ptr;
        typedef PixelIdT PixelId;
        typedef SkyMapIdList<SkyMapScheme> IdList;
        
        /**
        * \brief return a shared pointer to a copy
        *
        * Using a shared pointer prevents type slicing.
        */
        virtual typename SkyMapScheme<PixelIdT>::Ptr clone() const = 0;

        /**
        * \brief find all indices whose centers lie within the specified polygon
        */
        virtual SkyMapIdList<SkyMapScheme> findIndicesInPolygon(
            std::vector<lsst::afw::geom::Point2D> const &vertexList ///< list of RA/Dec vertices (radians)
        ) const = 0;

        /**
        * \brief find the indices of all pixels whose centers lie with a given radius of a point on the sky
        */
        virtual SkyMapIdList<SkyMapScheme> findIndicesInDisc(
            lsst::afw::geom::Point2D raDec, ///< RA/Dec center point (radians)
            double radius ///< radius (radians)
        ) const = 0;

        /**
        * \brief return the area of the specified pixel (in steradians)
        *
        * For sky pixelization schemes with equal-area pixels the argument is ignored.
        */
        virtual double getPixelArea(PixelIdT const &pixelId) const = 0;
        
        /**
        * \brief return the total number of pixels in an all-sky map
        */
        virtual boost::int64_t getTotalPixels() const = 0;

        /**
        * \brief return on-sky position of center of pixel
        */
        virtual lsst::afw::geom::Point2D getPixelPosition(PixelIdT const &pixelId) const = 0;

        /**
        * \brief get on-sky position of pixel corners
        */
        virtual PixelIdT getPixelId(
                lsst::afw::geom::Point2D const &raDec ///< RA/Dec (radians)
        ) const = 0;
    };


    class HealPixMapScheme: public SkyMapScheme<HealPixId> {
    public:
        typedef SkyMapScheme<HealPixId>::PixelId PixelId;
        typedef SkyMapScheme<HealPixId>::IdList IdList;

        explicit HealPixMapScheme(
                boost::int64_t nSides ///< number of sides for HEALPix map
        ) :
            _healPixBase(nSides, RING)
        { }

        virtual ~HealPixMapScheme() {};

        virtual SkyMapScheme<HealPixId>::Ptr clone() const {
            return SkyMapScheme<HealPixId>::Ptr(new HealPixMapScheme(getNSides()));
        }

        virtual SkyMapIdList<SkyMapScheme<HealPixId> > findIndicesInPolygon(
                std::vector<lsst::afw::geom::Point2D> const &vertexList
        ) const {
            // adapt the HEALPix FORTRAN code since it's not available in C++
            throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException, "Not yet implemented");
        }

        virtual SkyMapIdList<SkyMapScheme<HealPixId> > findIndicesInDisc(
                lsst::afw::geom::Point2D raDec, ///< RA/Dec center point (radians)
                double radius   ///< radius (radians)
        ) const {
            std::vector<PixelId> pixelList;
            this->_healPixBase.query_disc(_pointingFromRaDec(raDec), radius, pixelList);
            IdList retIdList(*this);
            for (std::vector<PixelId>::const_iterator pixPtr = pixelList.begin();
                pixPtr != pixelList.end(); ++pixPtr) {
                retIdList.add(*pixPtr);
            }
            return retIdList;
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
        * \brief return the HEALPix-specific nSides parameter
        */
        boost::int64_t getNSides() const { return _healPixBase.Nside(); };
    
    private:
        Healpix_Base2 _healPixBase;
        
        /**
        * \brief convert an RA/Dec to a HEALPix pointing object
        */
        inline static pointing _pointingFromRaDec(lsst::afw::geom::Point2D const &raDec) {
            return pointing((M_PI / 2.0) - raDec[1], raDec[2]);
        }

        /**
        * \brief convert a HEALPix pointing object to an RA/Dec
        */
        inline static lsst::afw::geom::Point2D _raDecFromPointing(pointing const &ptg) {
            return lsst::afw::geom::makePointD(ptg.phi, (M_PI / 2.0) - ptg.theta);
        }
    };

    /**
    * \brief A collection of sky map pixels (pixel ID/data pairs)
    */
    template <typename SkyMapSchemeT, typename PixelDataT>
    class SkyMapImage {
    public:
        typedef typename SkyMapSchemeT::PixelId PixelId;
        typedef PixelDataT PixelData;
        typedef std::pair<PixelId, PixelDataT> Pixel;
        typedef SkyMapIdList<SkyMapSchemeT> IdList;
        typedef typename std::map<PixelId, PixelDataT>::const_iterator const_iterator;
        typedef typename std::map<PixelId, PixelDataT>::iterator iterator;
        
        explicit SkyMapImage(
                SkyMapSchemeT const &scheme ///< sky map scheme
        ) : _schemePtr(scheme.clone()), _pixelList() {}

        explicit SkyMapImage(
                SkyMapSchemeT const &scheme, ///< sky map scheme
                SkyMapImage<SkyMapSchemeT, PixelDataT> const &pixelPtr ///< sky map pixel list
        ) : _schemePtr(scheme.clone()), _pixelList(pixelPtr) {}

        virtual ~SkyMapImage() {};

        typename SkyMapSchemeT::Ptr getScheme() { return _schemePtr; };
        
        /**
        * \brief Return a the IDs of the contained pixels
        */
        SkyMapIdList<SkyMapSchemeT> getIdList() const {
            IdList retIdList;
            for (const_iterator pixelPtr = begin(); pixelPtr != end(); ++pixelPtr) {
                retIdList.add(pixelPtr->first);
            }
            return retIdList;
        }

        /**
        * \brief Add the specified pixel; ignored if a already present
        */
        inline void add(Pixel const &pixel) { _pixelList.insert(pixel); }

        /**
        * \brief Add the specified list of pixels; pixels that are already present are ignored
        */
        void add(SkyMapImage const &pixelPtr) { _pixelList.insert(pixelPtr.begin(), pixelPtr.end()); }

        /**
        * \brief Is the pixel present?
        */
        inline bool contains(PixelId const &pixelId) const { return _pixelList.count(pixelId) > 0; }

        /**
        * \brief Remove the specified pixel; a no-op if the pixel is absent.
        */
        inline void remove(PixelId const &pixelId) { _pixelList.erase(pixelId); }

        /**
        * \brief Remove the specified list of pixels; pixels that are absent are ignored.
        */
        void remove(SkyMapIdList<SkyMapSchemeT> const &idList) {
            for (typename IdList::const_iterator inPtr = idList.begin(); inPtr != idList.end(); ++inPtr) {
                _pixelList.erase(*inPtr);
            }
        }
        
        /**
        * \brief Return the value of the specified pixel
        *
        * \raise lsst::pex::exceptions::NotFoundException if pixel is not present
        */
        PixelDataT get(PixelId const &pixelId) const {
            Pixel *pixelPtr = _pixelList.find(pixelId);
            if (pixelPtr == end()) {
                throw LSST_EXCEPT(lsst::pex::exceptions::NotFoundException, "Not found");
            }
            return pixelPtr->second;
        };
        
        /**
        * \brief Set the value of a specified pixel
        *
        * \raise lsst::pex::exceptions::NotFoundException if pixel is not present
        */
        void set(Pixel const &pixel) {
            Pixel *pixelPtr = _pixelList.find(pixel->first);
            if (pixelPtr == end()) {
                throw LSST_EXCEPT(lsst::pex::exceptions::NotFoundException, "Not found");
            }
            pixelPtr->second = pixel->second;
        }

        const_iterator begin() const { return _pixelList.begin(); };

        const_iterator end() const { return _pixelList.end(); };

        iterator begin() { return _pixelList.begin(); };

        iterator end() { return _pixelList.end(); };
        
        /**
        * \brief Make a sky map pixel
        */
        Pixel static makePixel(PixelId const &pixelId, PixelDataT const &pixelData) {
            return Pixel(pixelId, pixelData);
        }
        
    private:
        SkyMapSchemeT _schemePtr;
        std::map<PixelId, PixelDataT> _pixelList;
    };

}}} // namespace lsst::afw::image

#endif // LSST_AFW_IMAGE_SKYMAP_H
