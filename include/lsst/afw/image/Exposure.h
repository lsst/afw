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

#include <memory>

#include "lsst/base.h"
#include "lsst/daf/base.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/image/ExposureInfo.h"
#include "lsst/geom/Extent.h"
#include "lsst/geom/SpherePoint.h"

namespace lsst {
namespace afw {

namespace formatters {
template <typename ImageT, typename MaskT, typename VarianceT>
class ExposureFormatter;
}

namespace image {

/** A class to contain the data, WCS, and other information needed to describe an %image of the sky.
 * Exposure Class Implementation for LSST: a templated framework class
 * for creating an Exposure from a MaskedImage and a Wcs.
 *
 * An Exposure is required to take one afwImage::MaskedImage or a region (col,
 * row) defining the size of a MaskedImage (this can be of size 0,0).  An
 * Exposure can (but is not required to) contain an afwImage::SkyWcs.
 *
 * The template types should optimally be a float, double, unsigned int 16 bit,
 * or unsigned int 32 bit for the image (pixel) type and an unsigned int 32 bit
 * for the mask type.  These types have been explicitly instantiated for the
 * Exposure class.  All MaskedImage and Wcs constructors are 'const' to allow
 * for views and copying.
 *
 * An Exposure can get and return its MaskedImage, SkyWcs, and a subExposure.
 * The getSubExposure member takes a BBox region defining the subRegion of
 * the original Exposure to be returned.  The member retrieves the MaskedImage
 * corresponding to the subRegion.  The MaskedImage class throws an exception
 * for any subRegion extending beyond the original MaskedImage bounding
 * box. This member is not yet fully implemented because it requires the SkyWcs
 * class to return the SkyWcs metadata to the member so the CRPIX values of the
 * SkyWcs can be adjusted to reflect the new subMaskedImage origin.  The
 * getSubExposure member will eventually return a subExposure consisting of
 * the subMAskedImage and the SkyWcs object with its corresponding adjusted
 * metadata.
 *
 * The hasWcs member is used to determine if the Exposure has a SkyWcs.  It is not
 * required to have one.
 */
template <typename ImageT, typename MaskT = lsst::afw::image::MaskPixel,
          typename VarianceT = lsst::afw::image::VariancePixel>
class Exposure : public lsst::daf::base::Persistable, public lsst::daf::base::Citizen {
public:
    typedef MaskedImage<ImageT, MaskT, VarianceT> MaskedImageT;

    // Class Constructors and Destructor
    /** @brief Construct an Exposure with a blank MaskedImage of specified size (default 0x0) and
     * a SkyWcs (which may be default constructed)
     *
     * @param width number of columns
     * @param height number of rows
     * @param wcs the SkyWcs
     */
    explicit Exposure(unsigned int width, unsigned int height,
                      std::shared_ptr<geom::SkyWcs const> wcs = std::shared_ptr<geom::SkyWcs const>());

    /** @brief Construct an Exposure with a blank MaskedImage of specified size (default 0x0) and
     * a SkyWcs (which may be default constructed)
     *
     * @param dimensions desired image width/height
     * @param wcs the SkyWcs
     */
    explicit Exposure(lsst::geom::Extent2I const& dimensions = lsst::geom::Extent2I(),
                      std::shared_ptr<geom::SkyWcs const> wcs = std::shared_ptr<geom::SkyWcs const>());

    /** @brief Construct an Exposure with a blank MaskedImage of specified size (default 0x0) and a SkyWcs
     *
     * @param bbox desired image width/height, and origin
     * @param wcs the SkyWcs
     */
    explicit Exposure(lsst::geom::Box2I const& bbox,
                      std::shared_ptr<geom::SkyWcs const> wcs = std::shared_ptr<geom::SkyWcs const>());

    /** Construct an Exposure from a MaskedImage and an optional SkyWcs
     *
     * @param maskedImage the MaskedImage
     * @param wcs the SkyWcs
     */
    explicit Exposure(MaskedImageT& maskedImage,
                      std::shared_ptr<geom::SkyWcs const> wcs = std::shared_ptr<geom::SkyWcs const>());

    /** Construct an Exposure from a MaskedImage and an ExposureInfo
     *
     * If the ExposureInfo is an empty pointer then a new empty ExposureInfo is used
     *
     * @param maskedImage the MaskedImage
     * @param info the ExposureInfo
     */
    explicit Exposure(MaskedImageT& maskedImage, std::shared_ptr<ExposureInfo> info);

    /**
     *  Construct an Exposure by reading a regular FITS file.
     *
     *  @param[in]      fileName      File to read.
     *  @param[in]      bbox          If non-empty, read only the pixels within the bounding box.
     *  @param[in]      origin        Coordinate system of the bounding box; if PARENT, the bounding box
     *                                should take into account the xy0 saved with the image.
     *  @param[in]      conformMasks  If true, make Mask conform to the mask layout in the file.
     */
    explicit Exposure(std::string const& fileName, lsst::geom::Box2I const& bbox = lsst::geom::Box2I(),
                      ImageOrigin origin = PARENT, bool conformMasks = false);

    /**
     *  Construct an Exposure by reading a FITS image in memory.
     *
     *  @param[in]      manager       An object that manages the memory buffer to read.
     *  @param[in]      bbox          If non-empty, read only the pixels within the bounding box.
     *  @param[in]      origin        Coordinate system of the bounding box; if PARENT, the bounding box
     *                                should take into account the xy0 saved with the image.
     *  @param[in]      conformMasks  If true, make Mask conform to the mask layout in the file.
     */
    explicit Exposure(fits::MemFileManager& manager, lsst::geom::Box2I const& bbox = lsst::geom::Box2I(),
                      ImageOrigin origin = PARENT, bool conformMasks = false);

    /**
     *  Construct an Exposure from an already-open FITS object.
     *
     *  @param[in]      fitsfile      A FITS object to read from.  Current HDU is ignored.
     *  @param[in]      bbox          If non-empty, read only the pixels within the bounding box.
     *  @param[in]      origin        Coordinate system of the bounding box; if PARENT, the bounding box
     *                                should take into account the xy0 saved with the image.
     *  @param[in]      conformMasks  If true, make Mask conform to the mask layout in the file.
     */
    explicit Exposure(fits::Fits& fitsfile, lsst::geom::Box2I const& bbox = lsst::geom::Box2I(),
                      ImageOrigin origin = PARENT, bool conformMasks = false);

    /** Copy an Exposure
     *
     * @param src Parent Exposure
     * @param deep Should we copy the pixels?
     */
    Exposure(Exposure const& src, bool const deep = false);
    Exposure(Exposure&& src);

    /** Construct a subExposure given an Exposure and a bounding box
     *
     * @param src Parent Exposure
     * @param bbox Desired region in Exposure
     * @param origin Coordinate system for bbox
     * @param deep Should we copy the pixels?
     *
     * @throws lsst::pex::exceptions::InvalidParameterError if the requested subRegion
     * is not fully contained by the original MaskedImage BBox.
     */
    Exposure(Exposure const& src, lsst::geom::Box2I const& bbox, ImageOrigin const origin = PARENT,
             bool const deep = false);

    /// generalised copy constructor; defined here in the header so that the compiler can instantiate
    /// N(N-1)/2 conversions between N ImageBase types.
    ///
    /// We only support converting the Image part
    template <typename OtherPixelT>
    Exposure(Exposure<OtherPixelT, MaskT, VarianceT> const& rhs,  ///< Input Exposure
             const bool deep                                      ///< Must be true; needed to disambiguate
             )
            : lsst::daf::base::Citizen(typeid(this)),
              _maskedImage(rhs.getMaskedImage(), deep),
              _info(new ExposureInfo(*rhs.getInfo(), deep)) {
        if (not deep) {
            throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError,
                              "Exposure's converting copy constructor must make a deep copy");
        }
    }

    Exposure& operator=(Exposure const&);
    Exposure& operator=(Exposure&&);

    /**
     * Return a subimage corresponding to the given box.
     *
     * @param  bbox   Bounding box of the subimage returned.
     * @param  origin Origin bbox is rleative to; PARENT accounts for xy0, LOCAL does not.
     * @return        A subimage view into this.
     *
     * This method is wrapped as __getitem__ in Python.
     *
     * @note This method permits mutable views to be obtained from const
     *       references to images (just as the copy constructor does).
     *       This is an intrinsic flaw in Image's design.
     */
    Exposure subset(lsst::geom::Box2I const& bbox, ImageOrigin origin = PARENT) const {
        return Exposure(*this, bbox, origin, false);
    }

    /// Return a subimage corresponding to the given box (interpreted as PARENT coordinates).
    Exposure operator[](lsst::geom::Box2I const& bbox) const { return subset(bbox); }

    /** Destructor
     */
    virtual ~Exposure();

    // Get Members
    /// Return the MaskedImage
    MaskedImageT getMaskedImage() { return _maskedImage; }
    /// Return the MaskedImage
    MaskedImageT getMaskedImage() const { return _maskedImage; }

    std::shared_ptr<geom::SkyWcs const> getWcs() const { return _info->getWcs(); }

    /// Return the Exposure's Detector information
    std::shared_ptr<lsst::afw::cameraGeom::Detector const> getDetector() const {
        return _info->getDetector();
    }
    /// Return the Exposure's filter
    Filter getFilter() const { return _info->getFilter(); }
    /// Return flexible metadata
    std::shared_ptr<lsst::daf::base::PropertySet> getMetadata() const { return _info->getMetadata(); }
    void setMetadata(std::shared_ptr<lsst::daf::base::PropertySet> metadata) { _info->setMetadata(metadata); }

    /// Return the Exposure's width
    int getWidth() const { return _maskedImage.getWidth(); }
    /// Return the Exposure's height
    int getHeight() const { return _maskedImage.getHeight(); }
    /// Return the Exposure's size
    lsst::geom::Extent2I getDimensions() const { return _maskedImage.getDimensions(); }

    /**
     * Return the Exposure's column-origin
     *
     * @see getXY0()
     */
    int getX0() const { return _maskedImage.getX0(); }
    /**
     * Return the Exposure's row-origin
     *
     * @see getXY0()
     */
    int getY0() const { return _maskedImage.getY0(); }

    /**
     * Return the Exposure's origin
     *
     * This will usually be (0, 0) except for images created using the
     * `Exposure(fileName, hdu, BBox, mode)` ctor or `Exposure(Exposure, BBox)` cctor
     * The origin can be reset with `setXY0`
     */
    lsst::geom::Point2I getXY0() const { return _maskedImage.getXY0(); }

    lsst::geom::Box2I getBBox(ImageOrigin const origin = PARENT) const {
        return _maskedImage.getBBox(origin);
    }
    /**
     * Set the Exposure's origin (including correcting the Wcs)
     *
     * The origin is usually set by the constructor, so you shouldn't need this function
     *
     * @note There are use cases (e.g. memory overlays) that may want to set these values, but
     * don't do so unless you are an Expert.
     */
    void setXY0(lsst::geom::Point2I const& origin);

    // Set Members
    /** Set the MaskedImage of the Exposure.
     */
    void setMaskedImage(MaskedImageT& maskedImage);
    void setWcs(std::shared_ptr<geom::SkyWcs> wcs) { _info->setWcs(wcs); }
    /// Set the Exposure's Detector information
    void setDetector(std::shared_ptr<lsst::afw::cameraGeom::Detector const> detector) {
        _info->setDetector(detector);
    }
    /// Set the Exposure's filter
    void setFilter(Filter const& filter) { _info->setFilter(filter); }
    /// Set the Exposure's Calib object
    void setCalib(std::shared_ptr<Calib> calib) { _info->setCalib(calib); }
    /// Return the Exposure's Calib object
    std::shared_ptr<Calib> getCalib() { return _info->getCalib(); }
    /// Return the Exposure's Calib object
    std::shared_ptr<Calib const> getCalib() const { return _info->getCalib(); }
    /// Set the Exposure's Psf
    void setPsf(std::shared_ptr<lsst::afw::detection::Psf const> psf) { _info->setPsf(psf); }

    /// Return the Exposure's Psf object
    std::shared_ptr<lsst::afw::detection::Psf> getPsf() { return _info->getPsf(); }
    /// Return the Exposure's Psf object
    std::shared_ptr<lsst::afw::detection::Psf const> getPsf() const { return _info->getPsf(); }

    /// Does this Exposure have a Psf?
    bool hasPsf() const { return _info->hasPsf(); }

    /// Does this Exposure have a Wcs?
    bool hasWcs() const { return _info->hasWcs(); }

    /// Get the ExposureInfo that aggregates all the non-image components.  Never null.
    std::shared_ptr<ExposureInfo> getInfo() { return _info; }

    /// Get the ExposureInfo that aggregates all the non-image components.  Never null.
    std::shared_ptr<ExposureInfo const> getInfo() const { return _info; }

    /// Set the ExposureInfo that aggregates all the non-image components.
    void setInfo(std::shared_ptr<ExposureInfo> exposureInfo) { _info = exposureInfo; }

    /**
     *  Write an Exposure to a regular multi-extension FITS file.
     *
     *  @param[in] fileName      Name of the file to write.
     *
     *  As with MaskedImage persistence, an empty primary HDU will be created and all images planes
     *  will be saved to extension HDUs.  Most metadata will be saved only to the header of the
     *  main image HDU, but the WCS will be saved to the header of the mask and variance as well.
     *  If present, the Psf will be written to one or more additional HDUs.
     *
     *  Note that the LSST pixel origin differs from the FITS convention by one, so the values
     *  of CRPIX and LTV saved in the file are not the same as those in the C++ objects in memory,
     *  but are rather modified so they are interpreted by external tools (like ds9).
     */
    void writeFits(std::string const& fileName) const;

    /**
     *  Write an Exposure to a multi-extension FITS file in memory.
     *
     *  @param[in] manager       Manager for the memory to write to.
     *
     *  @see writeFits
     */
    void writeFits(fits::MemFileManager& manager) const;

    /**
     *  Write an Exposure to an already-open FITS file object.
     *
     *  @param[in] fitsfile       FITS object to write.
     *
     *  @see writeFits
     */
    void writeFits(fits::Fits& fitsfile) const;

    /**
     *  Write an Exposure to a regular multi-extension FITS file.
     *
     *  @param[in] fileName        Name of the file to write.
     *  @param[in] imageOptions    Options controlling writing of image as FITS.
     *  @param[in] maskOptions     Options controlling writing of mask as FITS.
     *  @param[in] varianceOptions Options controlling writing of variance as FITS.
     */
    void writeFits(std::string const& fileName, fits::ImageWriteOptions const& imageOptions,
                   fits::ImageWriteOptions const& maskOptions,
                   fits::ImageWriteOptions const& varianceOptions) const;

    /**
     *  Write an Exposure to a regular multi-extension FITS file.
     *
     *  @param[in] manager         Manager for the memory to write to.
     *  @param[in] imageOptions    Options controlling writing of image as FITS.
     *  @param[in] maskOptions     Options controlling writing of mask as FITS.
     *  @param[in] varianceOptions Options controlling writing of variance as FITS.
     */
    void writeFits(fits::MemFileManager& manager, fits::ImageWriteOptions const& imageOptions,
                   fits::ImageWriteOptions const& maskOptions,
                   fits::ImageWriteOptions const& varianceOptions) const;

    /**
     *  Write an Exposure to a regular multi-extension FITS file.
     *
     *  @param[in] fitsfile        FITS object to which to write.
     *  @param[in] imageOptions    Options controlling writing of image as FITS.
     *  @param[in] maskOptions     Options controlling writing of mask as FITS.
     *  @param[in] varianceOptions Options controlling writing of variance as FITS.
     */
    void writeFits(fits::Fits& fitsfile, fits::ImageWriteOptions const& imageOptions,
                   fits::ImageWriteOptions const& maskOptions,
                   fits::ImageWriteOptions const& varianceOptions) const;

    /**
     *  Read an Exposure from a regular FITS file.
     *
     *  @param[in] filename    Name of the file to read.
     */
    static Exposure readFits(std::string const& filename) {
        return Exposure<ImageT, MaskT, VarianceT>(filename);
    }

    /**
     *  Read an Exposure from a FITS RAM file.
     *
     *  @param[in] manager     Object that manages the memory to be read.
     */
    static Exposure readFits(fits::MemFileManager& manager) {
        return Exposure<ImageT, MaskT, VarianceT>(manager);
    }

    /**
     * Return an Exposure that is a small cutout of the original.
     *
     * @param center desired center of cutout (in RA and Dec)
     * @param size width and height (in that order) of cutout in pixels
     *
     * @return An Exposure of the requested size centered on `center` to within
     *     half a pixel in either dimension. Pixels past the edge of the original
     *     exposure will be zero.
     *
     * @throws lsst::pex::exceptions::LogicError Thrown if this Exposure does
     *     not have a WCS.
     * @throws lsst::pex::exceptions::InvalidParameterError Thrown if ``center``
     *     falls outside this Exposure or if ``size`` is not a valid size.
     */
    Exposure getCutout(lsst::geom::SpherePoint const& center, lsst::geom::Extent2I const& size) const;

private:
    LSST_PERSIST_FORMATTER(lsst::afw::formatters::ExposureFormatter<ImageT, MaskT, VarianceT>)

    void _readFits(fits::Fits& fitsfile, lsst::geom::Box2I const& bbox, ImageOrigin origin,
                   bool conformMasks);

    MaskedImageT _maskedImage;
    std::shared_ptr<ExposureInfo> _info;
};

/**
 * A function to return an Exposure of the correct type (cf. std::make_pair)
 */
template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
std::shared_ptr<Exposure<ImagePixelT, MaskPixelT, VariancePixelT>> makeExposure(
        MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT>& mimage,  ///< the Exposure's image
        std::shared_ptr<geom::SkyWcs const> wcs =
                std::shared_ptr<geom::SkyWcs const>()  ///< the Exposure's WCS
) {
    return typename std::shared_ptr<Exposure<ImagePixelT, MaskPixelT, VariancePixelT>>(
            new Exposure<ImagePixelT, MaskPixelT, VariancePixelT>(mimage, wcs));
}
}  // namespace image
}  // namespace afw
}  // namespace lsst

#endif  // LSST_AFW_IMAGE_EXPOSURE_H
