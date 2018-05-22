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
 * Rotate an Image (or Mask or MaskedImage) by a fixed angle or number of quarter turns
 */
#include <cstdint>

#include "lsst/geom.h"
#include "lsst/afw/math/offsetImage.h"
#include "lsst/afw/image/LsstImageTypes.h"

namespace afwImage = lsst::afw::image;

namespace lsst {
namespace afw {
namespace math {

template <typename ImageT>
std::shared_ptr<ImageT> rotateImageBy90(ImageT const& inImage, int nQuarter) {
    std::shared_ptr<ImageT> outImage;  // output image

    while (nQuarter < 0) {
        nQuarter += 4;
    }

    switch (nQuarter % 4) {
        case 0:
            outImage.reset(new ImageT(inImage, true));  // a deep copy of inImage
            break;
        case 1:
            outImage.reset(new ImageT(lsst::geom::Extent2I(inImage.getHeight(), inImage.getWidth())));

            for (int y = 0; y != inImage.getHeight(); ++y) {
                typename ImageT::y_iterator optr = outImage->col_begin(inImage.getHeight() - y - 1);
                for (typename ImageT::x_iterator iptr = inImage.row_begin(y), end = inImage.row_end(y);
                     iptr != end; ++iptr, ++optr) {
                    *optr = *iptr;
                }
            }

            break;
        case 2:
            outImage.reset(new ImageT(inImage.getDimensions()));

            for (int y = 0; y != inImage.getHeight(); ++y) {
                typename ImageT::x_iterator optr =
                        outImage->x_at(inImage.getWidth() - 1, inImage.getHeight() - y - 1);
                for (typename ImageT::x_iterator iptr = inImage.row_begin(y), end = inImage.row_end(y);
                     iptr != end; ++iptr, optr -= 1) {
                    *optr = *iptr;
                }
            }
            break;
        case 3:
            outImage.reset(new ImageT(lsst::geom::Extent2I(inImage.getHeight(), inImage.getWidth())));

            for (int y = 0; y != inImage.getHeight(); ++y) {
                typename ImageT::y_iterator optr = outImage->y_at(y, inImage.getWidth() - 1);
                for (typename ImageT::x_iterator iptr = inImage.row_begin(y), end = inImage.row_end(y);
                     iptr != end; ++iptr, optr -= 1) {
                    *optr = *iptr;
                }
            }

            break;
    }

    return outImage;
}

template <typename ImageT>
std::shared_ptr<ImageT> flipImage(ImageT const& inImage, bool flipLR, bool flipTB) {
    std::shared_ptr<ImageT> outImage(new ImageT(inImage, true));  // Output image

    if (flipLR) {
        if (flipTB) {
            for (int y = 0; y != inImage.getHeight(); ++y) {
                typename ImageT::x_iterator optr =
                        outImage->x_at(inImage.getWidth() - 1, inImage.getHeight() - y - 1);
                for (typename ImageT::x_iterator iptr = inImage.row_begin(y), end = inImage.row_end(y);
                     iptr != end; ++iptr, optr -= 1) {
                    *optr = *iptr;
                }
            }
        } else {
            for (int y = 0; y != inImage.getHeight(); ++y) {
                typename ImageT::x_iterator optr = outImage->x_at(inImage.getWidth() - 1, y);
                for (typename ImageT::x_iterator iptr = inImage.row_begin(y), end = inImage.row_end(y);
                     iptr != end; ++iptr, optr -= 1) {
                    *optr = *iptr;
                }
            }
        }
    } else {
        if (flipTB) {
            for (int y = 0; y != inImage.getHeight(); ++y) {
                typename ImageT::x_iterator optr = outImage->row_begin(inImage.getHeight() - y - 1);
                for (typename ImageT::x_iterator iptr = inImage.row_begin(y), end = inImage.row_end(y);
                     iptr != end; ++iptr, ++optr) {
                    *optr = *iptr;
                }
            }
        } else {
            ;  // nothing to do
        }
    }

    return outImage;
}

//
// Explicit instantiations
//
/// @cond
#define INSTANTIATE(TYPE)                                                                                \
    template std::shared_ptr<afwImage::Image<TYPE>> rotateImageBy90(afwImage::Image<TYPE> const&, int);  \
    template std::shared_ptr<afwImage::MaskedImage<TYPE>> rotateImageBy90(                               \
            afwImage::MaskedImage<TYPE> const&, int);                                                    \
    template std::shared_ptr<afwImage::Image<TYPE>> flipImage(afwImage::Image<TYPE> const&, bool flipLR, \
                                                              bool flipTB);                              \
    template std::shared_ptr<afwImage::MaskedImage<TYPE>> flipImage(afwImage::MaskedImage<TYPE> const&,  \
                                                                    bool flipLR, bool flipTB);

INSTANTIATE(std::uint16_t)
INSTANTIATE(int)
INSTANTIATE(float)
INSTANTIATE(double)
template std::shared_ptr<afwImage::Mask<afwImage::MaskPixel>>
    rotateImageBy90(afwImage::Mask<afwImage::MaskPixel> const&, int);
template std::shared_ptr<afwImage::Mask<afwImage::MaskPixel>>
    flipImage(afwImage::Mask<afwImage::MaskPixel> const&, bool flipLR, bool flipTB);
/// @endcond
}
}
}
