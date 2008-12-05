#include "lsst/afw/image.h"
#include "lsst/afw/detection.h"

namespace detection = lsst::afw::detection;
namespace image = lsst::afw::image;

namespace {
    template <typename MaskT>
    class FindSetBits : public detection::FootprintFunctor<MaskT> {
    public:
        FindSetBits(MaskT const& mask       // The Mask the source lives in
                   ) : detection::FootprintFunctor<MaskT>(mask), _bits(0) {}

        // method called for each pixel by apply()
        void operator()(typename MaskT::xy_locator loc,        // locator pointing at the pixel
                        int x,                                 // column-position of pixel
                        int y                                  // row-position of pixel
                       ) {
            _bits |= *loc;
        }

        // Return the bits set
        typename MaskT::Pixel getBits() const { return _bits; }
        // Clear the accumulator
        void reset() { _bits = 0; }
    private:
        typename MaskT::Pixel _bits;
    };
}

void printBits(image::Mask<image::MaskPixel> mask, detection::DetectionSet<float>::FootprintList& feet) {
    FindSetBits<image::Mask<image::MaskPixel> > count(mask);

    for (detection::DetectionSet<float>::FootprintList::iterator fiter = feet.begin(); fiter != feet.end(); ++fiter) {
        count.reset();
        count.apply(**fiter);

        printf("0x%x\n", count.getBits());
    }
}

int main() {
    image::MaskedImage<float> mimage(20, 30);

    (*mimage.getImage())(5, 6) = 100;
    (*mimage.getImage())(5, 7) = 110;

    *mimage.getMask() = 0x1;
    (*mimage.getMask())(5, 6) |= 0x2;
    (*mimage.getMask())(5, 7) |= 0x4;

    detection::DetectionSet<float> ds(mimage, 10);

    printBits(*mimage.getMask(), ds.getFootprints());
}
