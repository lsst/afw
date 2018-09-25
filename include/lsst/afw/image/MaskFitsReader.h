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

#ifndef LSST_AFW_IMAGE_MASKFITSREADER_H
#define LSST_AFW_IMAGE_MASKFITSREADER_H

#include "lsst/afw/image/ImageBaseFitsReader.h"
#include "lsst/afw/image/Mask.h"

namespace lsst { namespace afw { namespace image {

/**
 * A FITS reader class for Masks.
 *
 * @exceptsafe All MaskFitsReader methods provide strong exception safety, but
 *             exceptions thrown by the internal fits::Fits object itself may
 *             change its status variable or HDU pointer; MaskFitsReader
 *             guards against this by resetting those before any use of the
 *             Fits object.
 */
class MaskFitsReader final : public ImageBaseFitsReader {
public:

    using ImageBaseFitsReader::ImageBaseFitsReader;

    /**
     * Read the Mask.
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
    template <typename PixelT>
    Mask<PixelT> read(lsst::geom::Box2I const & bbox=lsst::geom::Box2I(), ImageOrigin origin=PARENT,
                      bool conformMasks=false, bool allowUnsafe=false);

};

}}} // namespace lsst::afw::image

#endif // !LSST_AFW_IMAGE_MASKFITSREADER_H
