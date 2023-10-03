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

#include "lsst/afw/image/MaskFitsReader.h"
#include "lsst/afw/image/detail/MaskDict.h"

namespace lsst {
namespace afw {
namespace image {

template <typename PixelT>
Mask<PixelT> MaskFitsReader::read(lsst::geom::Box2I const &bbox, ImageOrigin origin, bool conformMasks,
                                  bool allowUnsafe) {
    // TODO: deprecate conformMasks?
    Mask<PixelT> result(readArray<PixelT>(bbox, origin, allowUnsafe), false, readXY0(bbox, origin));
    auto metadata = readMetadata();
    // look for mask planes in the file
    detail::MaskDict fileMaskDict = Mask<PixelT>::parseMaskPlaneMetadata(metadata);
    // detail::MaskDict fileMD = fileMaskDict.getDefaultIfEmpty(fileMaskDict);
    // if (fileMD == detail::MaskDict(Mask::getNumPlanesMax())) {  // file is already consistent with Mask
    //     return result;
    // }
    // TODO: deprecate conformMasks?

    // if (conformMasks) {  // adopt the definitions in the file
    //     detail::MaskDict::setDefault(fileMD);
    //     result._maskDict = fileMD;
    // }
    // TODO: do we care about this?
    result.conformMaskPlanes(fileMaskDict);  // convert planes defined by fileMaskDict to the order
                                             // defined by Mask::_maskPlaneDict
    return result;
}

#define INSTANTIATE(T) \
    template Mask<T> MaskFitsReader::read(lsst::geom::Box2I const &, ImageOrigin, bool, bool)

INSTANTIATE(MaskPixel);

}  // namespace image
}  // namespace afw
}  // namespace lsst
