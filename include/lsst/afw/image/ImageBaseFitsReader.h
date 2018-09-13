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

#ifndef LSST_AFW_IMAGE_IMAGEBASEFITSREADER_H
#define LSST_AFW_IMAGE_IMAGEBASEFITSREADER_H

#include <string>
#include <memory>

#include "lsst/geom/Box.h"
#include "lsst/daf/base/PropertyList.h"
#include "lsst/afw/fits.h"

namespace lsst { namespace afw { namespace image {

/**
 * Base class for image FITS readers.
 *
 * This class should be considered an implementation detail of ImageFitsReader
 * and MaskFitsReader that provides their common methods, not the definition
 * of an interface.
 *
 * @exceptsafe All ImageBaseFitsReader methods provide strong exception safety,
 *             but exceptions thrown by the internal fits::Fits object itself
 *             may change its status variable or HDU pointer;
 *             ImageBaseFitsReader guards against this by resetting those
 *             before any use of the Fits object.
 */
class ImageBaseFitsReader {
public:

    /**
     * Construct a FITS reader object.
     *
     * @param  fileName Name of a file to open.
     * @param  hdu      HDU index, where 0 is the primary HDU and DEFAULT_HDU
     *                  is the first non-empty HDU.
     */
    explicit ImageBaseFitsReader(std::string const& fileName, int hdu=fits::DEFAULT_HDU);

    /**
     * Construct a FITS reader object.
     *
     * @param  manager  Memory block containing a FITS file.
     * @param  hdu      HDU index, where 0 is the primary HDU and DEFAULT_HDU
     *                  is the first non-empty HDU.
     */
    explicit ImageBaseFitsReader(fits::MemFileManager& manager, int hdu=fits::DEFAULT_HDU);

    /**
     * Construct a FITS reader object.
     *
     * @param  fitsFile  Pointer to a CFITSIO file object.  Lifetime will not
     *                   be managed by the Reader object.
     */
    explicit ImageBaseFitsReader(fits::Fits * fitsFile);

    // FITS readers are not copyable, movable, or assignable.
    ImageBaseFitsReader(ImageBaseFitsReader const &) = delete;
    ImageBaseFitsReader(ImageBaseFitsReader &&) = delete;
    ImageBaseFitsReader & operator=(ImageBaseFitsReader const &) = delete;
    ImageBaseFitsReader & operator=(ImageBaseFitsReader &&) = delete;

    /**
     * Read a string describing the pixel type of the on-disk image.
     *
     * @return A string of the form `[u](int|float)<bits>` (e.g. "uint16",
     *         "float64").
     */
    std::string readDType() const;

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

    /**
     * Read the image's FITS header.
     */
    std::shared_ptr<daf::base::PropertyList> readMetadata();

    /**
     * Read the image's data array.
     *
     * @param  bbox   A bounding box used to defined a subimage, or an empty
     *                box (default) to read the whole image.
     * @param  origin Coordinate system convention for the given box.
     */
    template <typename T>
    ndarray::Array<T, 2, 2> readArray(
        lsst::geom::Box2I const & bbox,
        ImageOrigin origin=PARENT
    );

    /**
     * Return the HDU this reader targets.
     */
    int getHdu() const noexcept { return _hdu; }

    /**
     * Return the name of the file this reader targets.
     */
    std::string getFileName() const { return _fitsFile->getFileName(); }

protected:

    // This class should never be directly instantiated, and should never
    // be deleted through a base-class pointer.
    ~ImageBaseFitsReader() noexcept;

private:

    friend class MaskedImageFitsReader;

    bool _ownsFitsFile;
    int _hdu;
    fits::Fits * _fitsFile;
    lsst::geom::Box2I _bbox;
    std::shared_ptr<daf::base::PropertyList> _metadata;
};

}}} // namespace lsst::afw::image

#endif // !LSST_AFW_IMAGE_IMAGEBASEFITSREADER_H
