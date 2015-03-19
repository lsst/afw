#if !defined(LSST_AFW_DISPLAY_RGB_H)
#define LSST_AFW_DISPLAY_RGB_H 1

namespace lsst { namespace afw { namespace display {
                
template<typename ImageT>
void
replaceSaturatedPixels(ImageT & rim,    //< R image (e.g. i)
                       ImageT & gim,    //< G image (e.g. r)
                       ImageT & bim,    //< B image (e.g. g)
                       int borderWidth = 2, //< width of border used to estimate colour of saturated regions
                       float saturatedPixelValue = 65535 //< the brightness of a saturated pixel, once fixed
                      );

}}}

#endif
