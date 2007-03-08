// -*- lsst-c++ -*-
#include "lsst/MaskedImage.h"
#include <typeinfo>

using namespace lsst;

template <typename ImagePixelT, typename MaskPixelT> class testPixProcFunc : public PixelProcessingFunc<ImagePixelT, MaskPixelT> {
public:
    typedef typename PixelChannelType<ImagePixelT>::type ImageChannelT;
    typedef typename PixelChannelType<MaskPixelT>::type MaskChannelT;
    typedef PixelLocator<ImagePixelT> ImageIteratorT;
    typedef PixelLocator<MaskPixelT> MaskIteratorT;
     
    testPixProcFunc(MaskedImage<ImagePixelT, MaskPixelT>& m) : PixelProcessingFunc<ImagePixelT, MaskPixelT>(m) {}
    
    void init() {
        PixelProcessingFunc<ImagePixelT, MaskPixelT>::_maskPtr->getPlaneBitMask("CR", bitsCR);
        testCount = 0;
    }
        
    void operator ()(ImageIteratorT &i,MaskIteratorT &m ) { 
        //  In general, do something to the pixel values
        ImageIteratorT j = i;
        if (++testCount < 10) {
            std::cout << *i << " " << *m << std::endl;
            *i = 1;
            int dx = 1;
            int dy = 10;
//             *(i.advance(dx,dy)) = 10;
            j.advance(dx,dy);
            *j = 10;
         }
     }

private:
     MaskChannelT bitsCR;
     int testCount;
};




int main()
{
     typedef PixelGray<uint8> MaskPixelType;
     typedef PixelGray<float32> ImagePixelType;

     MaskedImage<ImagePixelType,MaskPixelType > testMaskedImage1(272, 1037);
     MaskedImage<ImagePixelType,MaskPixelType > testMaskedImage2(272, 1037);
     testMaskedImage2 += testMaskedImage1;

     testPixProcFunc<ImagePixelType, MaskPixelType> fooFunc(testMaskedImage1);   // should be a way to automatically convey template types
                                                                                 // from testMaskedImage1 to fooFunc
     fooFunc.init();

     testMaskedImage1.processPixels(fooFunc);

     fooFunc.init();
     testMaskedImage1.processPixels(fooFunc);

}
