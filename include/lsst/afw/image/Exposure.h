// -*- LSST-C++ -*- // fixed format comment for emacs

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
  * @brief Declaration of the templated Exposure Class for LSST.
  *
  * Create an Exposure from a lsst::afw::image::MaskedImage.
  *
  * @ingroup afw
  *
  * @author Nicole M. Silvestri, University of Washington
  *
  * Contact: nms@astro.washington.edu
  *
  * Created on: Mon Apr 23 1:01:14 2007
  *
  * @version 
  *
  * LSST Legalese here...
  */

#ifndef LSST_AFW_IMAGE_EXPOSURE_H
#define LSST_AFW_IMAGE_EXPOSURE_H

#include "boost/cstdint.hpp"
#include "boost/shared_ptr.hpp"
#include "boost/make_shared.hpp"

#include "lsst/base.h"
#include "lsst/daf/base.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/image/Wcs.h"
#include "lsst/afw/image/TanWcs.h"
#include "lsst/afw/cameraGeom/Detector.h"
#include "lsst/afw/image/Filter.h"

namespace lsst {
namespace afw {
namespace detection {
    class Psf;
}

namespace formatters {
    template<typename ImageT, typename MaskT, typename VarianceT> class ExposureFormatter;
}

namespace fits {
class MemFileManager;
class Fits;
} // namespace fits

namespace image {

class Calib;

/// A class to contain the data, WCS, and other information needed to describe an %image of the sky
template<typename ImageT, typename MaskT=lsst::afw::image::MaskPixel,
         typename VarianceT=lsst::afw::image::VariancePixel>
class Exposure : public lsst::daf::base::Persistable,
                 public lsst::daf::base::Citizen {
public:
    typedef MaskedImage<ImageT, MaskT, VarianceT> MaskedImageT;
    typedef boost::shared_ptr<Exposure> Ptr;
    typedef boost::shared_ptr<Exposure const> ConstPtr;
    
    // Class Constructors and Destructor
    explicit Exposure(
        unsigned int width, unsigned int height, 
        Wcs const& wcs=NoWcs
    );

    explicit Exposure(
        lsst::afw::geom::Extent2I const & dimensions=lsst::afw::geom::Extent2I(),
        Wcs const& wcs=NoWcs
    );

    explicit Exposure(
        lsst::afw::geom::Box2I const & bbox,
        Wcs const & wcs=NoWcs
    );

    explicit Exposure(MaskedImageT & maskedImage, Wcs const& wcs=NoWcs);

    //@{
    /**
     *  @brief Construct an Exposure from a FITS multi-extension file.
     *  
     *  @note The method warns the user if the Exposure does not have a Wcs.
     *
     *  See the MaskedImage Fits constructors for more information; Exposure
     *  simply parses the FITS header for its additional data members.
     */
    explicit Exposure(
        std::string const &baseName, 
        int const hdu=0, 
        geom::Box2I const& bbox=geom::Box2I(), 
        ImageOrigin const origin=LOCAL,
        bool const conformMasks=false
    );
    explicit Exposure(
        afw::fits::MemFileManager & manager,
        int const hdu=0, 
        geom::Box2I const& bbox=geom::Box2I(), 
        ImageOrigin const origin=LOCAL, 
        bool const conformMasks=false
    );
    explicit Exposure(
        afw::fits::Fits & fitsfile,
        geom::Box2I const& bbox=geom::Box2I(), 
        ImageOrigin const origin=LOCAL, 
        bool const conformMasks=false
    );
    //@}
    
    Exposure(
        Exposure const &src, 
        bool const deep=false
    );

    Exposure(
        Exposure const &src, 
        lsst::afw::geom::Box2I const& bbox, 
        ImageOrigin const origin=LOCAL, 
        bool const deep=false
    );

    /// generalised copy constructor; defined here in the header so that the compiler can instantiate
    /// N(N-1)/2 conversions between N ImageBase types.
    ///
    /// We only support converting the Image part
    template<typename OtherPixelT>
    Exposure(Exposure<OtherPixelT, MaskT, VarianceT> const& rhs, //!< Input Exposure
             const bool deep        //!< Must be true; needed to disambiguate
    ) :
        lsst::daf::base::Citizen(typeid(this)),
        _maskedImage(rhs.getMaskedImage(), deep),
        _wcs(rhs.getWcs()->clone()),
        _detector(rhs.getDetector()),
        _filter(rhs.getFilter()),
        _calib(_cloneCalib(rhs.getCalib())),
        _psf(_clonePsf(rhs.getPsf()))
    {
        if (not deep) {
            throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException,
                              "Exposure's converting copy constructor must make a deep copy");
        }

        setMetadata(deep ? rhs.getMetadata()->deepCopy() : rhs.getMetadata());
    }

    virtual ~Exposure(); 

    // Get Members
    /// Return the MaskedImage
    MaskedImageT getMaskedImage() { return _maskedImage; }
    /// Return the MaskedImage
    MaskedImageT getMaskedImage() const { return _maskedImage; }
    Wcs::Ptr getWcs() const;
    /// Return the Exposure's Detector information
    lsst::afw::cameraGeom::Detector::Ptr getDetector() const { return _detector; }
    /// Return the Exposure's filter
    Filter getFilter() const { return _filter; }
    /// Return flexible metadata
    lsst::daf::base::PropertySet::Ptr getMetadata() const { return _metadata; }
    void setMetadata(lsst::daf::base::PropertySet::Ptr metadata) { _metadata = metadata; }

    /// Return the Exposure's width
    int getWidth() const { return _maskedImage.getWidth(); }
    /// Return the Exposure's height
    int getHeight() const { return _maskedImage.getHeight(); }
    /// Return the Exposure's size
    geom::Extent2I getDimensions() const { return _maskedImage.getDimensions(); }
    
    /**
     * Return the Exposure's row-origin
     *
     * \sa getXY0()
     */
    int getX0() const { return _maskedImage.getX0(); }
    /**
     * Return the Exposure's column-origin
     *
     * \sa getXY0()
     */
    int getY0() const { return _maskedImage.getY0(); }

    /**
     * Return the Exposure's origin
     *
     * This will usually be (0, 0) except for images created using the
     * <tt>Exposure(fileName, hdu, BBox, mode)</tt> ctor or <tt>Exposure(Exposure, BBox)</tt> cctor
     * The origin can be reset with \c setXY0
     */
    geom::Point2I getXY0() const { return _maskedImage.getXY0(); }

    geom::Box2I getBBox(ImageOrigin const origin=LOCAL) const {
        return _maskedImage.getBBox(origin);
    }
    /**
     * Set the Exposure's origin (including correcting the Wcs)
     *
     * The origin is usually set by the constructor, so you shouldn't need this function
     *
     * \note There are use cases (e.g. memory overlays) that may want to set these values, but
     * don't do so unless you are an Expert.
     */
    void setXY0(geom::Point2I const & origin) {
        geom::Point2I old(_maskedImage.getXY0());
        _wcs->shiftReferencePixel(origin.getX() - old.getX(), origin.getY() - old.getY());
        _maskedImage.setXY0(origin);
    }

    // Set Members
    void setMaskedImage(MaskedImageT &maskedImage);
    void setWcs(Wcs const& wcs);
    /// Set the Exposure's Detector information
    void setDetector(lsst::afw::cameraGeom::Detector::Ptr detector) { _detector = detector; }
    /// Set the Exposure's filter
    void setFilter(Filter const& filter) { _filter = filter; }
    /// Set the Exposure's Calib object
    void setCalib(PTR(Calib) calib) { _calib = calib; }
    /// Return the Exposure's Calib object
    PTR(Calib) getCalib() { return _calib; }
    /// Return the Exposure's Calib object
    CONST_PTR(Calib) getCalib() const { return _calib; }
    /// Set the Exposure's Psf
    void setPsf(CONST_PTR(lsst::afw::detection::Psf) psf) { _psf = _clonePsf(psf); }

    /// Return the Exposure's Psf object
    PTR(lsst::afw::detection::Psf) getPsf() { return _psf; }
    /// Return the Exposure's Psf object
    CONST_PTR(lsst::afw::detection::Psf) getPsf() const { return _psf; }
    
    /// Does this Exposure have a Psf?
    bool hasPsf() const { return static_cast<bool>(_psf); }

    /// Does this Exposure have a Wcs?
    bool hasWcs() const { return *_wcs ? true : false; }
        
    /**
     *  @brief Write the Exposure as a multi-extension FITS file with the given filename.
     *
     *  @param[in] fileName   Name of the file.
     *  @param[in] mode       "w" to write a new file; "a" to append.
     *
     *  If mode is 'w', an empty Primary HDU will be created before appending three image Extension HDUs.
     *
     *  @note LSST and FITS use a different convention for WCS coordinates.
     *  FITS measures crpix relative to the bottom left hand corner of the image
     *  saved in that file (what ds9 calls image coordinates). LSST measures it 
     *  relative to the bottom left hand corner of the parent image (what 
     *  ds9 calls the physical coordinates). This may cause confusion when you
     *  write an image to disk and discover that the values of crpix in the header
     *  are not what you expect.
     *
     *  exposure = afwImage.ExposureF(filename) 
     *  fitsHeader = afwImage.readMetadata(filename)
     * 
     *  exposure.getWcs().getPixelOrigin() ---> (128,128)
     *  fitsHeader.get("CRPIX1") --> 108
     *
     *  This is expected. If you look at the value of
     *  fitsHeader.get("LTV1") --> -20
     *  you will find that CRPIX - LTV == getPixelOrigin.
     *
     *  This implementation means that if you open the image in ds9 (say)
     *  the wcs translations for a given pixel are correct
     */
    void writeFits(std::string const& fileName, std::string const& mode="w") const;

    /**
     *  @brief Write the image as a FITS memory file.
     *
     *  If mode is 'w', an empty Primary HDU will be created before appending three image Extension HDUs.
     *
     *  @param[in] manager    Object that manages the lifetime of the memory block.
     *  @param[in] mode       "w" to write a new file; "a" to append.
     *
     *  @note LSST and FITS use a different convention for WCS coordinates.
     *  FITS measures crpix relative to the bottom left hand corner of the image
     *  saved in that file (what ds9 calls image coordinates). LSST measures it 
     *  relative to the bottom left hand corner of the parent image (what 
     *  ds9 calls the physical coordinates). This may cause confusion when you
     *  write an image to disk and discover that the values of crpix in the header
     *  are not what you expect.
     *
     *  exposure = afwImage.ExposureF(filename) 
     *  fitsHeader = afwImage.readMetadata(filename)
     * 
     *  exposure.getWcs().getPixelOrigin() ---> (128,128)
     *  fitsHeader.get("CRPIX1") --> 108
     *
     *  This is expected. If you look at the value of
     *  fitsHeader.get("LTV1") --> -20
     *  you will find that CRPIX - LTV == getPixelOrigin.
     *
     *  This implementation means that if you open the image in ds9 (say)
     *  the wcs translations for a given pixel are correct
     */
    void writeFits(fits::MemFileManager & manager, std::string const& mode="w") const;

    /**
     *  @brief Write the image to the current HDU of the given FITS file.
     *
     *  Three image Extension HDUs will be appended to the current file; if the file
     *  is empty, an empty Primary HDU will be added first.
     *
     *  @param[in] fitsfile   Internal cfitsio pointer in thin afw wrapper.
     *
     *  This overload is intended primarily for internal use and is not avalable in Python.
     *
     *  @note LSST and FITS use a different convention for WCS coordinates.
     *  FITS measures crpix relative to the bottom left hand corner of the image
     *  saved in that file (what ds9 calls image coordinates). LSST measures it 
     *  relative to the bottom left hand corner of the parent image (what 
     *  ds9 calls the physical coordinates). This may cause confusion when you
     *  write an image to disk and discover that the values of crpix in the header
     *  are not what you expect.
     *
     *  exposure = afwImage.ExposureF(filename) 
     *  fitsHeader = afwImage.readMetadata(filename)
     * 
     *  exposure.getWcs().getPixelOrigin() ---> (128,128)
     *  fitsHeader.get("CRPIX1") --> 108
     *
     *  This is expected. If you look at the value of
     *  fitsHeader.get("LTV1") --> -20
     *  you will find that CRPIX - LTV == getPixelOrigin.
     *
     *  This implementation means that if you open the image in ds9 (say)
     *  the wcs translations for a given pixel are correct
     */
    void writeFits(fits::Fits & fitsfile) const;

private:
    LSST_PERSIST_FORMATTER(lsst::afw::formatters::ExposureFormatter<ImageT, MaskT, VarianceT>)
    
    /// Finish initialization after constructing from a FITS file
    void postFitsCtorInit(lsst::daf::base::PropertySet::Ptr metadata);

    MaskedImageT _maskedImage;             
    Wcs::Ptr _wcs;
    cameraGeom::Detector::Ptr _detector;
    Filter _filter;
    PTR(Calib) _calib;
    PTR(lsst::afw::detection::Psf) _psf;
    lsst::daf::base::PropertySet::Ptr _metadata;

    lsst::daf::base::PropertySet::Ptr generateOutputMetadata() const;    //Used by writeFits()
    static PTR(lsst::afw::detection::Psf) _clonePsf(CONST_PTR(lsst::afw::detection::Psf) psf);
    static PTR(Calib) _cloneCalib(CONST_PTR(Calib) calib);
};

/**
 * A function to return an Exposure of the correct type (cf. std::make_pair)
 */
template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
typename Exposure<ImagePixelT, MaskPixelT, VariancePixelT>::Ptr makeExposure(
    MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT> & mimage, ///< the Exposure's image
    Wcs const & wcs=NoWcs ///< the Exposure's WCS
) {
    return typename Exposure<ImagePixelT, MaskPixelT, VariancePixelT>::Ptr(
        new Exposure<ImagePixelT, MaskPixelT, VariancePixelT>(mimage, wcs));
}

}}} // lsst::afw::image

#endif // LSST_AFW_IMAGE_EXPOSURE_H
