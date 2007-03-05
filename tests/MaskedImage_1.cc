// -*- lsst-c++ -*-
#include "lsst/MaskedImage.h"
#include <typeinfo>


namespace lsst {
template <typename ImagePixelT, typename MaskPixelT> class testPixProcFunc : public PixelProcessingFunc<ImagePixelT, MaskPixelT> {
public:
     typedef typename PixelChannelType<ImagePixelT>::type ImageChannelT;
     typedef typename PixelChannelType<MaskPixelT>::type MaskChannelT;
     
     testPixProcFunc(MaskedImage<ImagePixelT, MaskPixelT>& m) : PixelProcessingFunc<ImagePixelT, MaskPixelT>(m) {}
     
     void init() {
	  PixelProcessingFunc<ImagePixelT, MaskPixelT>::_maskPtr->getPlaneBitMask("CR", bitsCR);
	  testCount = 0;
     }
        
     void operator ()(boost::tuple<ImagePixelT&, MaskPixelT&> t) { 
         //  In general, do something to the pixel values
	  if (++testCount < 10) std::cout << "yup" << std::endl;
     }

private:
     MaskChannelT bitsCR;
     int testCount;
};

  template <typename ImagePixelT, typename MaskPixelT> void testFunc(PixelProcessingFunc<ImagePixelT, MaskPixelT> &func) {
    std::cout << "testFunc got: " << typeid(func).name() << std::endl;
  };

};

using namespace lsst;

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

}
