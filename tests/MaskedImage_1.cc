// -*- lsst-c++ -*-
#include <typeinfo>

#include <vw/Image.h>

#include <lsst/daf/data.h>
#include <lsst/pex/exceptions.h>
#include <lsst/pex/utils/Trace.h>
#include <lsst/afw/image.h>

using namespace std;
using lsst::pex::utils::Trace;
using lsst::daf::data::Citizen;
using lsst::daf::data::FitsFormatter;
namespace mwie = lsst::pex::exceptions;

template <typename ImagePixelT, typename MaskPixelT> 
class testPixProcFunc : public lsst::afw::image::PixelProcessingFunc<ImagePixelT, MaskPixelT> {
public:
    typedef typename vw::PixelChannelType<ImagePixelT>::type ImageChannelT;
    typedef typename vw::PixelChannelType<MaskPixelT>::type MaskChannelT;
    typedef lsst::afw::image::PixelLocator<ImagePixelT> ImageIteratorT;
    typedef lsst::afw::image::PixelLocator<MaskPixelT> MaskIteratorT;
    
    testPixProcFunc(lsst::afw::image::MaskedImage<ImagePixelT, MaskPixelT>& m) : 
        lsst::afw::image::PixelProcessingFunc<ImagePixelT, MaskPixelT>(m), initCount(0) {}
    
    void init() {
       lsst::afw::image::PixelProcessingFunc<ImagePixelT, MaskPixelT>::_maskPtr->getPlaneBitMask("CR", bitsCR);
       testCount = 0;
       initCount++;
    }
       
    void operator ()(ImageIteratorT &i,MaskIteratorT &m ) { 
       //  In general, do something to the pixel values
//         ImageIteratorT j = i;
//         if (++testCount < 10) {
//             cout << *i << " " << *m << endl;
//             *j = 1;
//             int dx = 1;
//             int dy = 0;
//             if (initCount <2) *(j.advance(dx,dy)) = 2*testCount;
//             cout << "modified: " << *j << endl;
//          }
       if (*i > 15000) {
           *m = *m | bitsCR;
           testCount++;
       }
    }

    int getCount() { return testCount; }

private:
    MaskChannelT bitsCR;
    int testCount;
    int initCount;
};




int test(int argc, char**argv) {
    if (argc < 5) {
       cerr << "Usage: inputBaseName1 inputBaseName2 outputBaseName1  outputBaseName2" << endl;
       return 1;
    }
    
    Trace::setDestination(cout);
    Trace::setVerbosity(".", 0);
    
    typedef uint8 MaskPixelType;
    typedef float32 ImagePixelType;

    lsst::afw::image::MaskedImage<ImagePixelType,MaskPixelType > testMaskedImage1;
    try {
        testMaskedImage1.readFits(argv[1]);
    } catch (mwie::ExceptionStack &e) {
        cerr << "Failed to open " << argv[1] << ": " << e.what() << endl;
        return 1;
    }

    testMaskedImage1.setDefaultVariance();
    testMaskedImage1.getMask()->addMaskPlane("CR");
    
    testPixProcFunc<ImagePixelType, MaskPixelType> fooFunc(testMaskedImage1);   // should be a way to automatically convey template types
                                                                                // from testMaskedImage1 to fooFunc

    fooFunc.init();
    testMaskedImage1.processPixels(fooFunc);
    cout << fooFunc.getCount() << " mask pixels were set" << endl;

    // verify that copy constructor and operator= build and do not leak
    lsst::afw::image::Image<ImagePixelType> testImage(100, 100);
    lsst::afw::image::Image<ImagePixelType> imageCopy(testImage);
    imageCopy = testImage;

    lsst::afw::image::MaskedImage<ImagePixelType,MaskPixelType > testMaskedImage2(testMaskedImage1);
    testMaskedImage2 = testMaskedImage1;

    lsst::afw::image::MaskedImage<ImagePixelType,MaskPixelType > testFlat;

    try {
        testFlat.readFits(argv[2]);
    } catch (mwie::ExceptionStack &e) {
        cerr << "Failed to open " << argv[2] << ": " << e.what() << endl;
        return 1;
    }
    testFlat.setDefaultVariance();

    testFlat /= 20000.0;

    testMaskedImage2 *= testFlat;

    // test of fits write

    testMaskedImage2.writeFits(argv[3]);

    // test of subImage

    lsst::afw::image::MaskedImage<ImagePixelType,MaskPixelType>::MaskedImagePtrT subMaskedImagePtr1;

    vw::BBox2i region(100, 600, 200, 300);
    subMaskedImagePtr1 = testMaskedImage1.getSubImage(region);
    *subMaskedImagePtr1 *= 0.5;
    subMaskedImagePtr1->writeFits(argv[4]);


    testMaskedImage1.replaceSubImage(region, subMaskedImagePtr1, true, true, true);

    // Check whether offsets have been correctly saved

    lsst::afw::image::MaskedImage<ImagePixelType,MaskPixelType>::MaskedImagePtrT subMaskedImagePtr2;

    vw::BBox2i region2(80, 110, 20, 30);
    subMaskedImagePtr2 = subMaskedImagePtr1->getSubImage(region2);

    cout << "Offsets: " << subMaskedImagePtr2->getOffsetCols() << " " << 
        subMaskedImagePtr2->getOffsetRows() << endl;

    testMaskedImage1.writeFits(argv[3]);

    DataProperty::PtrType metaDataPtr = testMaskedImage1.getImage()->getMetaData();

    cout << "Number of FITS header cards: " 
       << FitsFormatter::countFITSHeaderCards(metaDataPtr, false) << endl;
    cout << FitsFormatter::formatDataProperty(metaDataPtr, false) << endl;
    
    return 0;
}

int main(int argc, char **argv) {
    try {
       try {
           test(argc, argv);
       } catch (mwie::ExceptionStack &e) {
           throw mwie::Runtime(std::string("In handler\n") + e.what());
       }
    } catch (mwie::ExceptionStack &e) {
       clog << e.what() << endl;
    }

    //
    // Check for memory leaks
    //
    if (Citizen::census(0) == 0) {
        cerr << "No leaks detected" << endl;
    } else {
        cerr << "Leaked memory blocks:" << endl;
        Citizen::census(cerr);
    }
}
