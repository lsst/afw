/*
 * Developed for the LSST Data Management System.
 * This product includes software developed by the LSST Project
 * (https://www.lsst.org).
 * See the COPYRIGHT file at the top-level directory of this distribution
 * for details of code ownership.
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
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef LSST_AFW_IMAGE_MASKEDIMAGEFITSREADER_H
#define LSST_AFW_IMAGE_MASKEDIMAGEFITSREADER_H


#include "lsst/afw/image/ImageFitsReader.h"
#include "lsst/afw/image/MaskFitsReader.h"
#include "lsst/afw/image/MaskedImage.h"

namespace lsst { namespace afw { namespace image {

/**
 * A FITS reader class for MaskedImages and their components.
 *
 * @exceptsafe All MaskedImageFitsReader methods provide strong exception
 *             safety, but exceptions thrown by the internal fits::Fits object
 *             itself may change its status variable or HDU pointer;
 *             MaskedImageFitsReader guards against this by resetting those
 *             before any use of the Fits object.
 */
class MaskedImageFitsReader final {
public:

    /**
     * Construct a FITS reader object.
     *
     * @param  fileName Name of a file to open.
     * @param  hdu      HDU index for the image plane, where 0 is the primary
     *                  HDU and DEFAULT_HDU is the first non-empty HDU.
     */
    explicit MaskedImageFitsReader(std::string const& fileName, int hdu=fits::DEFAULT_HDU);

    /**
     * Construct a FITS reader object.
     *
     * @param  manager  Memory block containing a FITS file.
     * @param  hdu      HDU index for the image plane, where 0 is the primary
     *                  HDU and DEFAULT_HDU is the first non-empty HDU.
     */
    explicit MaskedImageFitsReader(fits::MemFileManager& manager, int hdu=fits::DEFAULT_HDU);

    /**
     * Construct a FITS reader object.
     *
     * @param  fitsFile  Pointer to a CFITSIO file object.  Lifetime will not
     *                   be managed by the Reader object.
     */
    explicit MaskedImageFitsReader(fits::Fits * fitsFile);

    // FITS readers are not copyable, movable, or assignable.
    MaskedImageFitsReader(MaskedImageFitsReader const &) = delete;
    MaskedImageFitsReader(MaskedImageFitsReader &&) = delete;
    MaskedImageFitsReader & operator=(MaskedImageFitsReader const &) = delete;
    MaskedImageFitsReader & operator=(MaskedImageFitsReader &&) = delete;

    ~MaskedImageFitsReader() noexcept;

    ///@{
    /**
     * Read a string describing the pixel type of the on-disk image plane.
     *
     * @return A string of the form `[u](int|float)<bits>` (e.g. "uint16",
     *         "float64").
     */
    std::string readImageDType() const;
    std::string readMaskDType() const;
    std::string readVarianceDType() const;
    ///@}

    /**
     * Read the bounding box of the on-disk image.
     *
     * @param  origin  Coordinate system convention for the returned box.
     *                 If LOCAL, the returned box will always have a minimum
     *                 of (0, 0).
     */
    lsst::geom::Box2I readBBox(ImageOrigin origin=PARENT);

    /**
     * Read the image origin from the on-disk image or a subimage thereof.
     *
     * @param  bbox   A bounding box used to defined a subimage, or an empty
     *                box (default) to use the whole image.
     * @param  origin Coordinate system convention for the given box.  Ignored
     *                if `bbox` is empty.
     */
    lsst::geom::Point2I readXY0(
        lsst::geom::Box2I const & bbox=lsst::geom::Box2I(),
        ImageOrigin origin=PARENT
    );

    ///@{
    /**
     * Read the FITS header of one of the HDUs.
     */
    std::shared_ptr<daf::base::PropertyList> readPrimaryMetadata();
    std::shared_ptr<daf::base::PropertyList> readImageMetadata();
    std::shared_ptr<daf::base::PropertyList> readMaskMetadata();
    std::shared_ptr<daf::base::PropertyList> readVarianceMetadata();
    ///@}

    /**
     * Read the image plane.
     *
     * @param  bbox   A bounding box used to defined a subimage, or an empty
     *                box (default) to read the whole image.
     * @param  origin Coordinate system convention for the given box.
     * @param  allowUnsafe   Permit reading into the requested pixel type even
     *                       when on-disk values may overflow or truncate.
     *
     * In Python, this templated method is wrapped with an additional `dtype`
     * argument to provide the type to read.  This defaults to the type of the
     * on-disk image.
     */
    template <typename ImagePixelT>
    Image<ImagePixelT> readImage(lsst::geom::Box2I const & bbox=lsst::geom::Box2I(),
                                 ImageOrigin origin=PARENT, bool allowUnsafe=false);

    /**
     * Read the mask plane.
     *
     * @param  bbox          A bounding box used to defined a subimage, or an
     *                       empty box (default) to read the whole image.
     * @param  origin        Coordinate system convention for the given box.
     * @param  conformMasks  If True, conform the global mask dict to match
     *                       this file.
     * @param  allowUnsafe   Permit reading into the requested pixel type even
     *                       when on-disk values may overflow or truncate.
     *
     * In Python, this templated method is wrapped with an additional `dtype`
     * argument to provide the type to read.  This defaults to the type of the
     * on-disk image.
     */
    template <typename MaskPixelT>
    Mask<MaskPixelT> readMask(lsst::geom::Box2I const & bbox=lsst::geom::Box2I(),
                              ImageOrigin origin=PARENT, bool conformMasks=false,
                              bool allowUnsafe=false);

    /**
     * Read the variance plane.
     *
     * @param  bbox   A bounding box used to defined a subimage, or an empty
     *                box (default) to read the whole image.
     * @param  origin Coordinate system convention for the given box.
     * @param  allowUnsafe   Permit reading into the requested pixel type even
     *                       when on-disk values may overflow or truncate.
     *
     * In Python, this templated method is wrapped with an additional `dtype`
     * argument to provide the type to read.  This defaults to the type of the
     * on-disk image.
     */
    template <typename VariancePixelT>
    Image<VariancePixelT> readVariance(lsst::geom::Box2I const & bbox=lsst::geom::Box2I(),
                                       ImageOrigin origin=PARENT, bool allowUnsafe=false);

    /**
     * Read the full MaskedImage.
     *
     * @param  bbox          A bounding box used to defined a subimage, or an
     *                       empty box (default) to read the whole image.
     * @param  origin        Coordinate system convention for the given box.
     * @param  conformMasks  If True, conform the global mask dict to match
     *                       this file.
     * @param  needAllHdus   If True, refuse to read the image if the mask
     *                       or variance plane is not present (the image plane
     *                       is always required).
     * @param  allowUnsafe   Permit reading into the requested pixel type even
     *                       when on-disk values may overflow or truncate.
     *
     * In Python, this templated method is wrapped with an additional `dtype`
     * argument to provide the type to read (for the image plane).  This
     * defaults to the type of the on-disk image.
     */
    template <typename ImagePixelT, typename MaskPixelT=MaskPixel, typename VariancePixelT=VariancePixel>
    MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT> read(
        lsst::geom::Box2I const & bbox=lsst::geom::Box2I(), ImageOrigin origin=PARENT,
        bool conformMasks=false, bool needAllHdus=false, bool allowUnsafe=false
    );

    /**
     * Return the name of the file this reader targets.
     */
    std::string getFileName() const { return _imageReader.getFileName(); }

private:

    friend class ExposureFitsReader;

    fits::Fits * _getFitsFile() { return _imageReader._fitsFile; }

    std::shared_ptr<daf::base::PropertyList> _imageMetadata;
    std::shared_ptr<daf::base::PropertyList> _maskMetadata;
    std::shared_ptr<daf::base::PropertyList> _varianceMetadata;
    ImageFitsReader _imageReader;
    MaskFitsReader _maskReader;
    ImageFitsReader _varianceReader;
};

}}} // namespace lsst::afw::image

#endif // !LSST_AFW_IMAGE_MASKEDIMAGEFITSREADER_H
