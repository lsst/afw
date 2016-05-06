/**
 * \file
 *
 * Handle saturated pixels when making colour images
 */
#include <cmath>
#include "boost/format.hpp"
#include "lsst/afw/detection.h"
#include "lsst/afw/image/MaskedImage.h"
#include "Rgb.h"

namespace lsst { namespace afw { namespace display {

namespace {
    template <typename ImageT>
    class SetPixels : public detection::FootprintFunctor<ImageT> {
    public:
        explicit SetPixels(ImageT const& img     // The image the source lives in
                          ) : detection::FootprintFunctor<ImageT>(img), _value(0) {}

        void setValue(float value) { _value = value; }

        // method called for each pixel by apply()
        void operator()(typename ImageT::xy_locator loc,        // locator pointing at the pixel
                        int,                                    // column-position of pixel
                        int                                     // row-position of pixel
                       ) {
            *loc = _value;
        }
    private:
        float _value;
    };
}

template<typename ImageT>
void
replaceSaturatedPixels(ImageT & rim,    // R image (e.g. i)
                       ImageT & gim,    // G image (e.g. r)
                       ImageT & bim,    // B image (e.g. g)
                       int borderWidth,	// width of border used to estimate colour of saturated regions
                       float saturatedPixelValue // the brightness of a saturated pixel, once fixed
                      )
{
    int const width = rim.getWidth(), height = rim.getHeight();
    int const x0 = rim.getX0(), y0 = rim.getY0();

    if (width != gim.getWidth() || height != gim.getHeight() || x0 != gim.getX0() || y0 != gim.getY0()) {
        throw LSST_EXCEPT(pex::exceptions::InvalidParameterError,
                          str(boost::format("R image has different size/origin from G image "
                                            "(%dx%d+%d+%d v. %dx%d+%d+%d") %
                              width % height % x0 % y0 %
                              gim.getWidth() % gim.getHeight() % gim.getX0() % gim.getY0()));

    }
    if (width != bim.getWidth() || height != bim.getHeight() || x0 != bim.getX0() || y0 != bim.getY0()) {
        throw LSST_EXCEPT(pex::exceptions::InvalidParameterError,
                          str(boost::format("R image has different size/origin from B image "
                                            "(%dx%d+%d+%d v. %dx%d+%d+%d") %
                              width % height % x0 % y0 %
                              bim.getWidth() % bim.getHeight() % bim.getX0() % bim.getY0()));

    }

    bool const useMaxPixel = !std::isfinite(saturatedPixelValue);

    SetPixels<typename ImageT::Image>
        setR(*rim.getImage()),
        setG(*gim.getImage()),
        setB(*bim.getImage()); // functors used to set pixel values

    // Find all the saturated pixels in any of the three image
    int const npixMin = 1;              // minimum number of pixels in an object
    afw::image::MaskPixel const SAT = rim.getMask()->getPlaneBitMask("SAT");
    detection::Threshold const satThresh(SAT, detection::Threshold::BITMASK);

    detection::FootprintSet       sat(*rim.getMask(), satThresh, npixMin);
    sat.merge(detection::FootprintSet(*gim.getMask(), satThresh, npixMin));
    sat.merge(detection::FootprintSet(*bim.getMask(), satThresh, npixMin));
    // go through the list of saturated regions, determining the mean colour of the surrounding pixels
    typedef detection::FootprintSet::FootprintList FootprintList;
    PTR(FootprintList) feet = sat.getFootprints();
    for (FootprintList::const_iterator ptr = feet->begin(), end = feet->end(); ptr != end; ++ptr) {
        PTR(detection::Footprint) const foot = *ptr;
        PTR(detection::Footprint) const bigFoot = growFootprint(*foot, borderWidth);

        double sumR = 0, sumG = 0, sumB = 0; // sum of all non-saturated adjoining pixels
        double maxR = 0, maxG = 0, maxB = 0; // maximum of non-saturated adjoining pixels

        for (detection::Footprint::SpanList::const_iterator sptr = bigFoot->getSpans().begin(),
                 send = bigFoot->getSpans().end(); sptr != send; ++sptr) {
            PTR(detection::Span) const span = *sptr;

            int const y = span->getY() - y0;
            if (y < 0 || y >= height) {
                continue;
            }
            int sx0 = span->getX0() - x0;
            if (sx0 < 0) {
                sx0 = 0;
            }
            int sx1 = span->getX1() - x0;
            if (sx1 >= width) {
                sx1 = width - 1;
            }

            for (typename ImageT::iterator
                     rptr = rim.at(sx0, y),
                     rend = rim.at(sx1 + 1, y),
                     gptr = gim.at(sx0, y),
                     bptr = bim.at(sx0, y); rptr != rend; ++rptr, ++gptr, ++bptr) {
                if (!((rptr.mask() | gptr.mask() | bptr.mask()) & SAT)) {
                    float val = rptr.image();
                    sumR += val;
                    if (val > maxR) {
                        maxR = val;
                    }
                    
                    val = gptr.image();
                    sumG += val;
                    if (val > maxG) {
                        maxG = val;
                    }

                    val = bptr.image();
                    sumB += val;
                    if (val > maxB) {
                        maxB = val;
                    }
                }
            }
        }
        // OK, we have the mean fluxes for the pixels surrounding this set of saturated pixels
        // so we can figure out the proper values to use for the saturated ones
        float R = 0, G = 0, B = 0;      // mean intensities
        if (sumR + sumB + sumG > 0) {
            if (sumR > sumG) {
                if (sumR > sumB) {
                    R = useMaxPixel ? maxR : saturatedPixelValue;
                    
                    G = (R*sumG)/sumR;
                    B = (R*sumB)/sumR;
                } else {
                    B = useMaxPixel ? maxB : saturatedPixelValue;
                    R = (B*sumR)/sumB;
                    G = (B*sumG)/sumB;
                }
            } else {
                if (sumG > sumB) {
                    G = useMaxPixel ? maxG : saturatedPixelValue;
                    R = (G*sumR)/sumG;
                    B = (G*sumB)/sumG;
                } else {
                    B = useMaxPixel ? maxB : saturatedPixelValue;
                    R = (B*sumR)/sumB;
                    G = (B*sumG)/sumB;
                }
            }
        }
        // Now that we know R, G, and B we can fix the values
        setR.setValue(R); setR.apply(*foot);
        setG.setValue(G); setG.apply(*foot);
        setB.setValue(B); setB.apply(*foot);
    }
}

template
void
replaceSaturatedPixels(image::MaskedImage<float> & rim,
                       image::MaskedImage<float> & gim,
                       image::MaskedImage<float> & bim,
                       int borderWidth,
                       float saturatedPixelValue
                      );

}}}
