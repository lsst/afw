/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */
 
#include <cstdio>

#include "lsst/afw/image.h"
#include "lsst/afw/detection.h"

namespace afwDetect = lsst::afw::detection;
namespace afwImage = lsst::afw::image;
namespace afwGeom = lsst::afw::geom;

namespace {
    template <typename MaskT>
    class FindSetBits : public afwDetect::FootprintFunctor<MaskT> {
    public:
        FindSetBits(MaskT const& mask       // The Mask the source lives in
                   ) : afwDetect::FootprintFunctor<MaskT>(mask), _bits(0) {}

        // method called for each pixel by apply()
        void operator()(typename MaskT::xy_locator loc,        // locator pointing at the pixel
                        int,                                   // column-position of pixel
                        int                                    // row-position of pixel
                       ) {
            _bits |= *loc;
        }

        // Return the bits set
        typename MaskT::Pixel getBits() const { return _bits; }
        // Clear the accumulator
        void reset() { _bits = 0; }
        void reset(afwDetect::Footprint const&) { ; }
    private:
        typename MaskT::Pixel _bits;
    };
}

void printBits(afwImage::Mask<afwImage::MaskPixel> mask,
               afwDetect::FootprintSet::FootprintList const& feet) {
    FindSetBits<afwImage::Mask<afwImage::MaskPixel> > count(mask);

    for (afwDetect::FootprintSet::FootprintList::const_iterator fiter = feet.begin();
        fiter != feet.end(); ++fiter) {
        count.apply(**fiter);

        printf("0x%x\n", count.getBits());
    }
}

int main() {
    afwImage::MaskedImage<float> mimage(afwGeom::Extent2I(20, 30));

    (*mimage.getImage())(5, 6) = 100;
    (*mimage.getImage())(5, 7) = 110;

    *mimage.getMask() = 0x1;
    (*mimage.getMask())(5, 6) |= 0x2;
    (*mimage.getMask())(5, 7) |= 0x4;

    afwDetect::FootprintSet ds(mimage, 10);

    printBits(*mimage.getMask(), *ds.getFootprints());
}
