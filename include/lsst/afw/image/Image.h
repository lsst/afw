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
 * 2-D images
 *
 * This file contains the 2-d image support for LSST
 */
#ifndef LSST_AFW_IMAGE_IMAGE_H
#define LSST_AFW_IMAGE_IMAGE_H

#include <string>
#include <utility>
#include <functional>

#include "boost/mpl/bool.hpp"
#include <climits>
#include <memory>

#include "lsst/geom.h"
#include "lsst/afw/image/ImageBase.h"
#include "lsst/afw/image/lsstGil.h"
#include "lsst/afw/image/ImageUtils.h"
#include "lsst/afw/image/Mask.h"
#include "lsst/afw/math/Function.h"
#include "lsst/afw/fitsDefaults.h"
#include "lsst/daf/base.h"
#include "lsst/daf/base/Citizen.h"
#include "lsst/pex/exceptions.h"
#include "ndarray.h"

namespace lsst {
namespace afw {

namespace formatters {
template <typename PixelT>
class ImageFormatter;
template <typename PixelT>
class DecoratedImageFormatter;
}  // namespace formatters

namespace image {

/// A class to represent a 2-dimensional array of pixels
template <typename PixelT>
class Image : public ImageBase<PixelT> {
public:
    template <typename, typename, typename>
    friend class MaskedImage;

    typedef detail::Image_tag image_category;

    /// A templated class to return this classes' type (present in Image/Mask/MaskedImage)
    template <typename ImagePT = PixelT>
    struct ImageTypeFactory {
        /// Return the desired type
        typedef Image<ImagePT> type;
    };
    template <typename OtherPixelT>
    friend class Image;  // needed by generalised copy constructors

    /**
     * Create an initialised Image of the specified size
     *
     * @param width number of columns
     * @param height number of rows
     * @param initialValue Initial value
     *
     * @note Many lsst::afw::image and lsst::afw::math objects define a `dimensions` member
     * which may be conveniently used to make objects of an appropriate size
     */
    explicit Image(unsigned int width, unsigned int height, PixelT initialValue = 0);
    /**
     * Create an initialised Image of the specified size
     *
     * @param dimensions Number of columns, rows
     * @param initialValue Initial value
     *
     * @note Many lsst::afw::image and lsst::afw::math objects define a `dimensions` member
     * which may be conveniently used to make objects of an appropriate size
     */
    explicit Image(lsst::geom::Extent2I const& dimensions = lsst::geom::Extent2I(), PixelT initialValue = 0);
    /**
     * Create an initialized Image of the specified size
     *
     * @param bbox dimensions and origin of desired Image
     * @param initialValue Initial value
     */
    explicit Image(lsst::geom::Box2I const& bbox, PixelT initialValue = 0);

    /**
     * Copy constructor to make a copy of part of an Image.
     *
     * The bbox ignores X0/Y0 if origin == LOCAL, and uses it if origin == PARENT.
     *
     * @param rhs Right-hand-side Image
     * @param bbox Specify desired region
     * @param origin Coordinate system of the bbox
     * @param deep If false, new ImageBase shares storage with rhs; if true make a new, standalone, ImageBase
     *
     * @note Unless `deep` is `true`, the new %image will share the old %image's pixels;
     * this is probably what you want
     */
    explicit Image(Image const& rhs, lsst::geom::Box2I const& bbox, ImageOrigin const origin = PARENT,
                   const bool deep = false);
    /**
     * Copy constructor.
     *
     * @param rhs Right-hand-side Image
     * @param deep If false, new Image shares storage with rhs; if true make a new, standalone, ImageBase
     *
     * @note Unless `deep` is `true`, the new %image will share the old %image's pixels;
     * this may not be what you want.  See also assign(rhs) to copy pixels between Image%s
     */
    Image(const Image& rhs, const bool deep = false);
    Image(Image&& rhs);

    /**
     *  Construct an Image by reading a regular FITS file.
     *
     *  @param[in]      fileName    File to read.
     *  @param[in]      hdu         HDU to read, 0-indexed (i.e. 0=Primary HDU).  The special value
     *                              of afw::fits::DEFAULT_HDU reads the Primary HDU unless it is empty,
     *                              in which case it reads the first extension HDU.
     *  @param[in,out]  metadata    Metadata read from the header (may be null).
     *  @param[in]      bbox        If non-empty, read only the pixels within the bounding box.
     *  @param[in]      origin      Coordinate system of the bounding box; if PARENT, the bounding box
     *                              should take into account the xy0 saved with the image.
     */
    explicit Image(std::string const& fileName, int hdu = fits::DEFAULT_HDU,
                   std::shared_ptr<lsst::daf::base::PropertySet> metadata =
                           std::shared_ptr<lsst::daf::base::PropertySet>(),
                   lsst::geom::Box2I const& bbox = lsst::geom::Box2I(), ImageOrigin origin = PARENT);

    /**
     *  Construct an Image by reading a FITS image in memory.
     *
     *  @param[in]      manager     An object that manages the memory buffer to read.
     *  @param[in]      hdu         HDU to read, 0-indexed (i.e. 0=Primary HDU).  The special value
     *                              of afw::fits::DEFAULT_HDU reads the Primary HDU unless it is empty,
     *                              in which case it reads the first extension HDU.
     *  @param[in,out]  metadata    Metadata read from the header (may be null).
     *  @param[in]      bbox        If non-empty, read only the pixels within the bounding box.
     *  @param[in]      origin      Coordinate system of the bounding box; if PARENT, the bounding box
     *                              should take into account the xy0 saved with the image.
     */
    explicit Image(fits::MemFileManager& manager, int hdu = fits::DEFAULT_HDU,
                   std::shared_ptr<lsst::daf::base::PropertySet> metadata =
                           std::shared_ptr<lsst::daf::base::PropertySet>(),
                   lsst::geom::Box2I const& bbox = lsst::geom::Box2I(), ImageOrigin origin = PARENT);

    /**
     *  Construct an Image from an already-open FITS object.
     *
     *  @param[in]      fitsfile    A FITS object to read from, already at the desired HDU.
     *  @param[in,out]  metadata    Metadata read from the header (may be null).
     *  @param[in]      bbox        If non-empty, read only the pixels within the bounding box.
     *  @param[in]      origin      Coordinate system of the bounding box; if PARENT, the bounding box
     *                              should take into account the xy0 saved with the image.
     */
    explicit Image(fits::Fits& fitsfile,
                   std::shared_ptr<lsst::daf::base::PropertySet> metadata =
                           std::shared_ptr<lsst::daf::base::PropertySet>(),
                   lsst::geom::Box2I const& bbox = lsst::geom::Box2I(), ImageOrigin origin = PARENT);

    // generalised copy constructor
    template <typename OtherPixelT>
    Image(Image<OtherPixelT> const& rhs, const bool deep) : image::ImageBase<PixelT>(rhs, deep) {}

    explicit Image(ndarray::Array<PixelT, 2, 1> const& array, bool deep = false,
                   lsst::geom::Point2I const& xy0 = lsst::geom::Point2I())
            : image::ImageBase<PixelT>(array, deep, xy0) {}

    ~Image() override = default;
    //
    // Assignment operators are not inherited
    //
    /// Set the %image's pixels to rhs
    Image& operator=(const PixelT rhs);
    /**
     * Assignment operator.
     *
     * @note that this has the effect of making the lhs share pixels with the rhs which may not be what you
     * intended;  to copy the pixels, use assign(rhs)
     *
     * @note this behaviour is required to make the swig interface work, otherwise I'd declare this function
     * private
     */
    Image& operator=(const Image& rhs);
    Image& operator=(Image&& rhs);

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
    Image subset(lsst::geom::Box2I const & bbox, ImageOrigin origin=PARENT) const {
        return Image(*this, bbox, origin, false);
    }

    /// Return a subimage corresponding to the given box (interpreted as PARENT coordinates).
    Image operator[](lsst::geom::Box2I const & bbox) const {
        return subset(bbox);
    }

    using ImageBase<PixelT>::operator[];

    /**
     *  Write an image to a regular FITS file.
     *
     *  @param[in] fileName      Name of the file to write.
     *  @param[in] metadata      Additional values to write to the header (may be null).
     *  @param[in] mode          "w"=Create a new file; "a"=Append a new HDU.
     */
    void writeFits(std::string const& fileName,
                   std::shared_ptr<lsst::daf::base::PropertySet const> metadata =
                           std::shared_ptr<lsst::daf::base::PropertySet const>(),
                   std::string const& mode = "w") const;

    /**
     *  Write an image to a FITS RAM file.
     *
     *  @param[in] manager       Manager object for the memory block to write to.
     *  @param[in] metadata      Additional values to write to the header (may be null).
     *  @param[in] mode          "w"=Create a new file; "a"=Append a new HDU.
     */
    void writeFits(fits::MemFileManager& manager,
                   std::shared_ptr<lsst::daf::base::PropertySet const> metadata =
                           std::shared_ptr<lsst::daf::base::PropertySet const>(),
                   std::string const& mode = "w") const;

    /**
     *  Write an image to an open FITS file object.
     *
     *  @param[in] fitsfile      A FITS file already open to the desired HDU.
     *  @param[in] metadata      Additional values to write to the header (may be null).
     */
    void writeFits(fits::Fits& fitsfile, std::shared_ptr<lsst::daf::base::PropertySet const> metadata =
                                                 std::shared_ptr<lsst::daf::base::PropertySet const>()) const;

    /**
     *  Write an image to a regular FITS file.
     *
     *  @param[in] filename      Name of the file to write.
     *  @param[in] options       Options controlling writing of FITS image.
     *  @param[in] mode          "w"=Create a new file; "a"=Append a new HDU.
     *  @param[in] header        Additional values to write to the header (may be null).
     *  @param[in] mask          Mask, for calculation of statistics.
     */
    void writeFits(std::string const& filename, fits::ImageWriteOptions const& options,
                   std::string const& mode = "w",
                   std::shared_ptr<daf::base::PropertySet const> header = nullptr,
                   std::shared_ptr<Mask<MaskPixel> const> mask = nullptr) const;

    /**
     *  Write an image to a FITS RAM file.
     *
     *  @param[in] manager       Manager object for the memory block to write to.
     *  @param[in] options       Options controlling writing of FITS image.
     *  @param[in] header        Additional values to write to the header (may be null).
     *  @param[in] mode          "w"=Create a new file; "a"=Append a new HDU.
     *  @param[in] mask          Mask, for calculation of statistics.
     */
    void writeFits(fits::MemFileManager& manager, fits::ImageWriteOptions const& options,
                   std::string const& mode = "w",
                   std::shared_ptr<daf::base::PropertySet const> header = nullptr,
                   std::shared_ptr<Mask<MaskPixel> const> mask = nullptr) const;

    /**
     *  Write an image to an open FITS file object.
     *
     *  @param[in] fitsfile      A FITS file already open to the desired HDU.
     *  @param[in] options       Options controlling writing of FITS image.
     *  @param[in] header        Additional values to write to the header (may be null).
     *  @param[in] mask          Mask, for calculation of statistics.
     */
    void writeFits(fits::Fits& fitsfile, fits::ImageWriteOptions const& options,
                   std::shared_ptr<daf::base::PropertySet const> header = nullptr,
                   std::shared_ptr<Mask<MaskPixel> const> mask = nullptr) const;

    /**
     *  Read an Image from a regular FITS file.
     *
     *  @param[in] filename    Name of the file to read.
     *  @param[in] hdu         Number of the "header-data unit" to read (where 0 is the Primary HDU).
     *                         The default value of afw::fits::DEFAULT_HDU is interpreted as
     *                         "the first HDU with NAXIS != 0".
     */
    static Image readFits(std::string const& filename, int hdu = fits::DEFAULT_HDU) {
        return Image<PixelT>(filename, hdu);
    }

    /**
     *  Read an Image from a FITS RAM file.
     *
     *  @param[in] manager     Object that manages the memory to be read.
     *  @param[in] hdu         Number of the "header-data unit" to read (where 0 is the Primary HDU).
     *                         The default value of afw::fits::DEFAULT_HDU is interpreted as
     *                         "the first HDU with NAXIS != 0".
     */
    static Image readFits(fits::MemFileManager& manager, int hdu = fits::DEFAULT_HDU) {
        return Image<PixelT>(manager, hdu);
    }

    void swap(Image& rhs);
    //
    // Operators etc.
    //
    /// Add scalar rhs to lhs
    Image& operator+=(PixelT const rhs);
    /// Add Image rhs to lhs
    virtual Image& operator+=(Image<PixelT> const& rhs);
    /**
     * Add a Function2(x, y) to an Image
     *
     * @param function function to add
     */
    Image& operator+=(lsst::afw::math::Function2<double> const& function);
    /// Add Image c*rhs to lhs
    void scaledPlus(double const c, Image<PixelT> const& rhs);
    /// Subtract scalar rhs from lhs
    Image& operator-=(PixelT const rhs);
    /// Subtract Image rhs from lhs
    Image& operator-=(Image<PixelT> const& rhs);
    /**
     * Subtract a Function2(x, y) from an Image
     *
     * @param function function to add
     */
    Image& operator-=(lsst::afw::math::Function2<double> const& function);
    /// Subtract Image c*rhs from lhs
    void scaledMinus(double const c, Image<PixelT> const& rhs);
    /// Multiply lhs by scalar rhs
    Image& operator*=(PixelT const rhs);
    /// Multiply lhs by Image rhs (i.e. %pixel-by-%pixel multiplication)
    Image& operator*=(Image<PixelT> const& rhs);
    /// Multiply lhs by Image c*rhs (i.e. %pixel-by-%pixel multiplication)
    void scaledMultiplies(double const c, Image<PixelT> const& rhs);
    /**
     * Divide lhs by scalar rhs
     *
     * @note Floating point types implement this by multiplying by the 1/rhs
     */
    Image& operator/=(PixelT const rhs);
    /// Divide lhs by Image rhs (i.e. %pixel-by-%pixel division)
    Image& operator/=(Image<PixelT> const& rhs);
    /// Divide lhs by Image c*rhs (i.e. %pixel-by-%pixel division)
    void scaledDivides(double const c, Image<PixelT> const& rhs);

    // In-place per-pixel sqrt().  Useful when handling variance planes.
    void sqrt();

protected:
    using ImageBase<PixelT>::_getRawView;

private:
    LSST_PERSIST_FORMATTER(lsst::afw::formatters::ImageFormatter<PixelT>)
};

/// Add lhs to Image rhs (i.e. %pixel-by-%pixel addition) where types are different
template <typename LhsPixelT, typename RhsPixelT>
Image<LhsPixelT>& operator+=(Image<LhsPixelT>& lhs, Image<RhsPixelT> const& rhs);
/// Subtract lhs from Image rhs (i.e. %pixel-by-%pixel subtraction) where types are different
template <typename LhsPixelT, typename RhsPixelT>
Image<LhsPixelT>& operator-=(Image<LhsPixelT>& lhs, Image<RhsPixelT> const& rhs);
/// Multiply lhs by Image rhs (i.e. %pixel-by-%pixel multiplication) where types are different
template <typename LhsPixelT, typename RhsPixelT>
Image<LhsPixelT>& operator*=(Image<LhsPixelT>& lhs, Image<RhsPixelT> const& rhs);
/// Divide lhs by Image rhs (i.e. %pixel-by-%pixel division) where types are different
template <typename LhsPixelT, typename RhsPixelT>
Image<LhsPixelT>& operator/=(Image<LhsPixelT>& lhs, Image<RhsPixelT> const& rhs);

template <typename PixelT>
void swap(Image<PixelT>& a, Image<PixelT>& b);

/**
 * A container for an Image and its associated metadata
 */
template <typename PixelT>
class DecoratedImage : public lsst::daf::base::Persistable, public lsst::daf::base::Citizen {
public:
    /**
     * Create an %image of the specified size
     *
     * @param dimensions desired number of columns. rows
     */
    explicit DecoratedImage(const lsst::geom::Extent2I& dimensions = lsst::geom::Extent2I());
    /**
     * Create an %image of the specified size
     *
     * @param bbox (width, height) and origin of the desired Image
     *
     * @note Many lsst::afw::image and lsst::afw::math objects define a `dimensions` member
     * which may be conveniently used to make objects of an appropriate size
     */
    explicit DecoratedImage(const lsst::geom::Box2I& bbox);
    /**
     * Create a DecoratedImage wrapping `rhs`
     *
     * Note that this ctor shares pixels with the rhs; it isn't a deep copy
     *
     * @param rhs Image to go into DecoratedImage
     */
    explicit DecoratedImage(std::shared_ptr<Image<PixelT>> rhs);
    /**
     * Copy constructor
     *
     * Note that the lhs will share memory with the rhs unless `deep` is true
     *
     * @param rhs right hand side
     * @param deep Make deep copy?
     */
    DecoratedImage(DecoratedImage const& rhs, const bool deep = false);
    /**
     * Create a DecoratedImage from a FITS file
     *
     * @param fileName File to read
     * @param hdu The HDU to read
     * @param bbox Only read these pixels
     * @param origin Coordinate system of the bbox
     */
    explicit DecoratedImage(std::string const& fileName, const int hdu = fits::DEFAULT_HDU,
                            lsst::geom::Box2I const& bbox = lsst::geom::Box2I(),
                            ImageOrigin const origin = PARENT);

    /**
     * Assignment operator
     *
     * N.b. this is a shallow assignment; use set(src) if you want to copy the pixels
     */
    DecoratedImage& operator=(const DecoratedImage& image);

    std::shared_ptr<lsst::daf::base::PropertySet> getMetadata() const { return _metadata; }
    void setMetadata(std::shared_ptr<lsst::daf::base::PropertySet> metadata) { _metadata = metadata; }

    /// Return the number of columns in the %image
    int getWidth() const { return _image->getWidth(); }
    /// Return the number of rows in the %image
    int getHeight() const { return _image->getHeight(); }

    /// Return the %image's column-origin
    int getX0() const { return _image->getX0(); }
    /// Return the %image's row-origin
    int getY0() const { return _image->getY0(); }

    /// Return the %image's size;  useful for passing to constructors
    const lsst::geom::Extent2I getDimensions() const { return _image->getDimensions(); }

    void swap(DecoratedImage& rhs);

    /**
     * Write a FITS file
     *
     * @param fileName the file to write
     * @param metadata metadata to write to header; or NULL
     * @param mode "w" to write a new file; "a" to append
     */
    void writeFits(std::string const& fileName,
                   std::shared_ptr<lsst::daf::base::PropertySet const> metadata =
                           std::shared_ptr<lsst::daf::base::PropertySet const>(),
                   std::string const& mode = "w") const;

    /**
     * Write a FITS file
     *
     * @param[in] fileName the file to write
     * @param[in] options       Options controlling writing of FITS image.
     * @param[in] metadata metadata to write to header; or NULL
     * @param[in] mode "w" to write a new file; "a" to append
     */
    void writeFits(std::string const& fileName, fits::ImageWriteOptions const& options,
                   std::shared_ptr<lsst::daf::base::PropertySet const> metadata =
                           std::shared_ptr<lsst::daf::base::PropertySet const>(),
                   std::string const& mode = "w") const;

    /// Return a shared_ptr to the DecoratedImage's Image
    std::shared_ptr<Image<PixelT>> getImage() { return _image; }
    /// Return a shared_ptr to the DecoratedImage's Image as const
    std::shared_ptr<Image<PixelT> const> getImage() const { return _image; }

    /**
     * Return the DecoratedImage's gain
     * @note This is mostly just a place holder for other properties that we might
     * want to associate with a DecoratedImage
     */
    double getGain() const { return _gain; }
    /// Set the DecoratedImage's gain
    void setGain(double gain) { _gain = gain; }

private:
    LSST_PERSIST_FORMATTER(lsst::afw::formatters::DecoratedImageFormatter<PixelT>)
    std::shared_ptr<Image<PixelT>> _image;
    std::shared_ptr<lsst::daf::base::PropertySet> _metadata;

    double _gain;

    void init();
};

template <typename PixelT>
void swap(DecoratedImage<PixelT>& a, DecoratedImage<PixelT>& b);

/// Determine the image bounding box from its metadata (FITS header)
///
/// Note that this modifies the metadata, stripping the WCS headers that
/// provide the xy0.
lsst::geom::Box2I bboxFromMetadata(daf::base::PropertySet& metadata);

/**
 * Return true if the pixels for two images or masks overlap in memory.
 */
template <typename T1, typename T2>
bool imagesOverlap(ImageBase<T1> const& image1, ImageBase<T2> const& image2);

}  // namespace image
}  // namespace afw
}  // namespace lsst

#endif
