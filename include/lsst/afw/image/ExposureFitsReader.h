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

#ifndef LSST_AFW_IMAGE_EXPOSUREFITSREADER_H
#define LSST_AFW_IMAGE_EXPOSUREFITSREADER_H

#include "lsst/afw/image/MaskedImageFitsReader.h"
#include "lsst/afw/image/ExposureInfo.h"
#include "lsst/afw/image/Exposure.h"

namespace lsst {
namespace afw {
namespace image {

/**
 * A FITS reader class for Exposures and their components.
 *
 * @exceptsafe All ExposureFitsReader methods provide strong exception safety,
 *             but exceptions thrown by the internal fits::Fits object itself
 *             may change its status variable or HDU pointer;
 *             ExposureFitsReader guards against this by resetting those
 *             before any use of the Fits object.
 */
class ExposureFitsReader {
public:
    /**
     * Construct a FITS reader object.
     *
     * @param  fileName Name of a file to open.
     */
    explicit ExposureFitsReader(std::string const &fileName);

    /**
     * Construct a FITS reader object.
     *
     * @param  manager  Memory block containing a FITS file.
     */
    explicit ExposureFitsReader(fits::MemFileManager &manager);

    /**
     * Construct a FITS reader object.
     *
     * @param  fitsFile  Pointer to a CFITSIO file object.  Lifetime will not
     *                   be managed by the Reader object.
     */
    explicit ExposureFitsReader(fits::Fits *fitsFile);

    // FITS readers are not copyable, movable, or assignable.
    ExposureFitsReader(ExposureFitsReader const &) = delete;
    ExposureFitsReader(ExposureFitsReader &&) = delete;
    ExposureFitsReader &operator=(ExposureFitsReader const &) = delete;
    ExposureFitsReader &operator=(ExposureFitsReader &&) = delete;

    ~ExposureFitsReader() noexcept;

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
    lsst::geom::Box2I readBBox(ImageOrigin origin = PARENT);

    /**
     * Read the image origin from the on-disk image or a subimage thereof.
     *
     * @param  bbox   A bounding box used to defined a subimage, or an empty
     * @param  origin Coordinate system convention for the given box.  Ignored
     *                if `bbox` is empty.
     */
    lsst::geom::Point2I readXY0(lsst::geom::Box2I const &bbox = lsst::geom::Box2I(),
                                ImageOrigin origin = PARENT);

    /**
     * Read the flexible metadata associated with the Exposure.
     *
     * FITS header keys used to construct other Exposure components will be
     * stripped.
     */
    std::shared_ptr<daf::base::PropertyList> readMetadata();

    /// Read the Exposure's world coordinate system.
    std::shared_ptr<afw::geom::SkyWcs> readWcs();

    /// Read the Exposure's filter.
    Filter readFilter();

    /// Read the Exposure's photometric calibration.
    std::shared_ptr<PhotoCalib> readPhotoCalib();

    /// Read the Exposure's point-spread function.
    std::shared_ptr<detection::Psf> readPsf();

    /// Read the polygon describing the region of validity for the Exposure.
    std::shared_ptr<afw::geom::polygon::Polygon> readValidPolygon();

    /// Read the Exposure's aperture correction map.
    std::shared_ptr<ApCorrMap> readApCorrMap();

    /// Read the Exposure's coadd input catalogs.
    std::shared_ptr<CoaddInputs> readCoaddInputs();

    /// Read the Exposure's visit metadata.
    std::shared_ptr<VisitInfo> readVisitInfo();

    /// Read the Exposure's transmission curve.
    std::shared_ptr<TransmissionCurve> readTransmissionCurve();

    /// Read the Exposure's detector.
    std::shared_ptr<cameraGeom::Detector> readDetector();

    /// Read the Exposure's non-standard components
    std::map<std::string, std::shared_ptr<table::io::Persistable>> readExtraComponents();

    /// Read the ExposureInfo containing all non-image components.
    std::shared_ptr<ExposureInfo> readExposureInfo();

    ///@{
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
    Image<ImagePixelT> readImage(lsst::geom::Box2I const &bbox = lsst::geom::Box2I(),
                                 ImageOrigin origin = PARENT, bool allowUnsafe = false);
    template <typename ImagePixelT>
    ndarray::Array<ImagePixelT, 2, 2> readImageArray(lsst::geom::Box2I const &bbox = lsst::geom::Box2I(),
                                                     ImageOrigin origin = PARENT, bool allowUnsafe = false);
    ///@}

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
    Mask<MaskPixelT> readMask(lsst::geom::Box2I const &bbox = lsst::geom::Box2I(),
                              ImageOrigin origin = PARENT, bool conformMasks = false,
                              bool allowUnsafe = false);

    /**
     * Read the mask plane.
     *
     * @param  bbox          A bounding box used to defined a subimage, or an
     *                       empty box (default) to read the whole image.
     * @param  origin        Coordinate system convention for the given box.
     * @param  allowUnsafe   Permit reading into the requested pixel type even
     *                       when on-disk values may overflow or truncate.
     *
     * In Python, this templated method is wrapped with an additional `dtype`
     * argument to provide the type to read.  This defaults to the type of the
     * on-disk image.
     */
    template <typename MaskPixelT>
    ndarray::Array<MaskPixelT, 2, 2> readMaskArray(lsst::geom::Box2I const &bbox = lsst::geom::Box2I(),
                                                   ImageOrigin origin = PARENT, bool allowUnsafe = false);

    ///@{
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
    Image<VariancePixelT> readVariance(lsst::geom::Box2I const &bbox = lsst::geom::Box2I(),
                                       ImageOrigin origin = PARENT, bool allowUnsafe = false);
    template <typename VariancePixelT>
    ndarray::Array<VariancePixelT, 2, 2> readVarianceArray(
            lsst::geom::Box2I const &bbox = lsst::geom::Box2I(), ImageOrigin origin = PARENT,
            bool allowUnsafe = false);
    ///@}

    /**
     * Read the MaskedImage.
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
     * argument to provide the type to read (for the image plane).  This
     * defaults to the type of the on-disk image.
     */
    template <typename ImagePixelT, typename MaskPixelT = MaskPixel, typename VariancePixelT = VariancePixel>
    MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT> readMaskedImage(
            lsst::geom::Box2I const &bbox = lsst::geom::Box2I(), ImageOrigin origin = PARENT,
            bool conformMasks = false, bool allowUnsafe = false);

    /**
     * Read the full Exposure.
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
     * argument to provide the type to read (for the image plane).  This
     * defaults to the type of the on-disk image.
     */
    template <typename ImagePixelT, typename MaskPixelT = MaskPixel, typename VariancePixelT = VariancePixel>
    Exposure<ImagePixelT, MaskPixelT, VariancePixelT> read(
            lsst::geom::Box2I const &bbox = lsst::geom::Box2I(), ImageOrigin origin = PARENT,
            bool conformMasks = false, bool allowUnsafe = false);

    /**
     * Return the name of the file this reader targets.
     */
    std::string getFileName() const { return _maskedImageReader.getFileName(); }

private:
    class MetadataReader;
    class ArchiveReader;

    void _ensureReaders();

    fits::Fits *_getFitsFile() { return _maskedImageReader._getFitsFile(); }

    MaskedImageFitsReader _maskedImageReader;
    std::unique_ptr<MetadataReader> _metadataReader;
    std::unique_ptr<ArchiveReader> _archiveReader;
};

}  // namespace image
}  // namespace afw
}  // namespace lsst

#endif  // !LSST_AFW_IMAGE_EXPOSUREFITSREADER_H
