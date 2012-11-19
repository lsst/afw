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

#ifndef LSST_AFW_IMAGE_EXPOSURE_H
#define LSST_AFW_IMAGE_EXPOSURE_H

#include "boost/cstdint.hpp"
#include "boost/shared_ptr.hpp"
#include "boost/make_shared.hpp"

#include "lsst/base.h"
#include "lsst/daf/base.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/image/Filter.h"

namespace lsst {
namespace afw {

namespace cameraGeom {
    class Detector;
}

namespace detection {
    class Psf;
}

namespace formatters {
    template<typename ImageT, typename MaskT, typename VarianceT> class ExposureFormatter;
}

namespace image {

    class Calib;
    class Wcs;

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
        CONST_PTR(Wcs) wcs = CONST_PTR(Wcs)()
    );

    explicit Exposure(
        lsst::afw::geom::Extent2I const & dimensions=lsst::afw::geom::Extent2I(),
        CONST_PTR(Wcs) wcs = CONST_PTR(Wcs)()
    );

    explicit Exposure(
        lsst::afw::geom::Box2I const & bbox,
        CONST_PTR(Wcs) wcs = CONST_PTR(Wcs)()
    );

    explicit Exposure(MaskedImageT & maskedImage,
                      CONST_PTR(Wcs) wcs = CONST_PTR(Wcs)());

    /**
     *  @brief Construct an Exposure by reading a regular FITS file.
     *
     *  @param[in]      fileName      File to read.
     *  @param[in]      hdu           First HDU to read, 1-indexed (i.e. 1=Primary HDU).  The special value
     *                                of 0 reads the Primary HDU unless it is empty, in which case it
     *                                reads the first extension HDU.
     *  @param[in]      bbox          If non-empty, read only the pixels within the bounding box.
     *  @param[in]      origin        Coordinate system of the bounding box; if PARENT, the bounding box
     *                                should take into account the xy0 saved with the image.
     *  @param[in]      conformMasks  If true, make Mask conform to the mask layout in the file.
     *
     *  Exposures may also be read from three separate files, in which the fileName argument is
     *  interpreted as the base file name and "_img.fits", "_msk.fits", and "_var.fits" are appended to it.
     *  This format is deprecated and is only provided temporarily for backwards compatibility.
     */
    explicit Exposure(
        std::string const & fileName, int hdu=0, geom::Box2I const& bbox=geom::Box2I(),
        ImageOrigin origin=LOCAL, bool conformMasks=false
    );

    /**
     *  @brief Construct an Exposure by reading a FITS image in memory.
     *
     *  @param[in]      manager       An object that manages the memory buffer to read.
     *  @param[in]      hdu           First HDU to read, 1-indexed (i.e. 1=Primary HDU).  The special value
     *                                of 0 reads the Primary HDU unless it is empty, in which case it
     *                                reads the first extension HDU.
     *  @param[in]      bbox          If non-empty, read only the pixels within the bounding box.
     *  @param[in]      origin        Coordinate system of the bounding box; if PARENT, the bounding box
     *                                should take into account the xy0 saved with the image.
     *  @param[in]      conformMasks  If true, make Mask conform to the mask layout in the file.
     */
    explicit Exposure(
        fits::MemFileManager & manager, int hdu=0, geom::Box2I const & bbox=geom::Box2I(),
        ImageOrigin origin=LOCAL, bool conformMasks=false
    );

    /**
     *  @brief Construct an Exposure from an already-open FITS object.
     *
     *  @param[in]      fitsfile      A FITS object to read from, already at the desired HDU.
     *  @param[in]      bbox          If non-empty, read only the pixels within the bounding box.
     *  @param[in]      origin        Coordinate system of the bounding box; if PARENT, the bounding box
     *                                should take into account the xy0 saved with the image.
     *  @param[in]      conformMasks  If true, make Mask conform to the mask layout in the file.
     */
    explicit Exposure(
        fits::Fits & fitsfile, geom::Box2I const & bbox=geom::Box2I(),
        ImageOrigin origin=LOCAL, bool conformMasks=false
    );

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
        _wcs(_cloneWcs(rhs.getWcs())),
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

    CONST_PTR(Wcs) getWcs() const { return _wcs; }
    /// Return the Exposure's Detector information
    CONST_PTR(lsst::afw::cameraGeom::Detector) getDetector() const { return _detector; }
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
    void setXY0(geom::Point2I const & origin);

    // Set Members
    void setMaskedImage(MaskedImageT &maskedImage);
    void setWcs(PTR(Wcs) wcs) { _wcs = wcs; }
    /// Set the Exposure's Detector information
    void setDetector(CONST_PTR(lsst::afw::cameraGeom::Detector) detector) { _detector = detector; }
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
    bool hasWcs() const { return static_cast<bool>(_wcs); }

    void writeFits(std::string const & fileName) const;
    void writeFits(fits::MemFileManager & manager) const;
    void writeFits(fits::Fits & fitsfile) const;

private:
    LSST_PERSIST_FORMATTER(lsst::afw::formatters::ExposureFormatter<ImageT, MaskT, VarianceT>)

    /// Finish initialization after constructing from a FITS file
    void postFitsCtorInit(lsst::daf::base::PropertySet::Ptr metadata);

    lsst::daf::base::PropertySet::Ptr generateOutputMetadata() const;    //Used by writeFits()

    MaskedImageT _maskedImage;             
    PTR(Wcs) _wcs;
    CONST_PTR(lsst::afw::cameraGeom::Detector) _detector;
    Filter _filter;
    PTR(Calib) _calib;
    PTR(lsst::afw::detection::Psf) _psf;
    lsst::daf::base::PropertySet::Ptr _metadata;

    static PTR(lsst::afw::detection::Psf) _clonePsf(CONST_PTR(lsst::afw::detection::Psf) psf);
    static PTR(Calib) _cloneCalib(CONST_PTR(Calib) calib);
    static PTR(Wcs) _cloneWcs(CONST_PTR(Wcs) wcs);
};

/**
 * A function to return an Exposure of the correct type (cf. std::make_pair)
 */
template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
typename Exposure<ImagePixelT, MaskPixelT, VariancePixelT>::Ptr makeExposure(
    MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT> & mimage, ///< the Exposure's image
    CONST_PTR(Wcs) wcs = CONST_PTR(Wcs)() ///< the Exposure's WCS
) {
    return typename Exposure<ImagePixelT, MaskPixelT, VariancePixelT>::Ptr(
        new Exposure<ImagePixelT, MaskPixelT, VariancePixelT>(mimage, wcs));
}

}}} // lsst::afw::image

#endif // LSST_AFW_IMAGE_EXPOSURE_H
