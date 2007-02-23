// -*- lsst-c++ -*-
#include "lsst/Mask.h"

using namespace lsst;

template <typename MaskPixelT> class testCrFunc : public MaskPixelBooleanFunc<MaskPixelT> {
public:
    typedef typename PixelChannelType<MaskPixelT>::type MaskChannelT;
    testCrFunc(Mask<MaskPixelT>& m) : MaskPixelBooleanFunc<MaskPixelT>(m) {}
    void init() {
        MaskPixelBooleanFunc<MaskPixelT>::_mask.getPlaneBitMask("CR", bitsCR);
    }        
    bool operator ()(MaskPixelT pixel) { 
        return ((pixel.v() & bitsCR) !=0 ); 
    }
private:
    MaskChannelT bitsCR;
};

int main(int argc, char *argv[])
{
// ------------- Test constructors

     typedef PixelGray<uint8> MaskPixelType;
     typedef ImageView<MaskPixelType> MaskImageType;
     typedef boost::shared_ptr<MaskImageType> MaskImagePtrType;
     typedef boost::shared_ptr<Mask<MaskPixelType> > MaskPtrType;

     
     MaskImagePtrType maskImage(new MaskImageType(300,400));
     cout << maskImage.use_count() << endl;

     Mask<MaskPixelType > testMask(maskImage);
     cout << maskImage.use_count() << endl;

     typedef PixelGray<uint16> MaskPixelType2;
     typedef ImageView<MaskPixelType2> MaskImageType2;
     typedef boost::shared_ptr<MaskImageType2> MaskImagePtrType2;

     MaskImagePtrType2 maskImage2(new MaskImageType2(300,400));

     Mask<MaskPixelType2 > testMask2(maskImage2);

     Mask<MaskPixelType > testMask3(300,400);

// ------------- Test mask plane addition

     int iPlane;

     iPlane = testMask.addMaskPlane("CR");
     cout << "Assigned CR to plane " << iPlane << endl;

     iPlane = testMask.addMaskPlane("BP");
     cout << "Assigned BP to plane " << iPlane << endl;

     int planeCR, planeBP;

     if (testMask.getMaskPlane("CR", planeCR) == false) {
	  cout << "No CR plane found" << endl;
     } else {
	  cout << "CR plane is " << planeCR << endl;
     }

     if (testMask.getMaskPlane("BP", planeBP) == false) {
	  cout << "No BP plane found" << endl;
     } else {
	  cout << "BP plane is " << planeBP << endl;
     }


// ------------ Test mask plane operations

     testMask.clearMaskPlane(planeCR);

     PixelCoord coord;
     list<PixelCoord> pixelList;

     for (int x=0; x<300; x+=1) {
	  for (int y=300; y<400; y+=20) {
	       coord.x = x;
	       coord.y = y;
	       pixelList.push_back(coord);
	  }
     }

     testMask.setMaskPlaneValues(planeCR, pixelList);

     testMask.setMaskPlaneValues(planeBP, pixelList);

     for (int x=250; x<300; x+=10) {
	  for (int y=300; y<400; y+=20) {
	       cout << x << " " << y << " " << (int)testMask(x, y) << " " << testMask(x, y, planeCR) << endl;
	  }
     }

     testMask.clearMaskPlane(planeCR);

     cout << endl;
     for (int x=250; x<300; x+=10) {
	  for (int y=300; y<400; y+=20) {
	       cout << x << " " << y << " " << (int)testMask(x, y) << " " << testMask(x, y, planeCR) << endl;
	  }
     }

// ------------------ Test |= operator
   
     iPlane = testMask3.addMaskPlane("CR");
     cout << "Assigned CR to plane " << iPlane << endl;

     testMask |= testMask3;

     cout << "Applied |= operator" << endl;
     
// -------------- Test mask plane removal

     testMask.clearMaskPlane(planeBP);
     testMask.removeMaskPlane("BP");

     if (testMask.getMaskPlane("CR", planeCR) == false) {
	  cout << "No CR plane found" << endl;
     } else {
	  cout << "CR plane is " << planeCR << endl;
     }

     if (testMask.getMaskPlane("BP", planeBP) == false) {
	  cout << "No BP plane found" << endl;
     } else {
	  cout << "BP plane is " << planeBP << endl;
     }

// --------------- Test submask methods

     testMask.setMaskPlaneValues(planeCR, pixelList);

     MaskPtrType subTestMask;

     BBox2i region(100, 300, 10, 40);
     subTestMask = testMask.getSubMask(region);

     testMask.clearMaskPlane(planeCR);

     testMask.replaceSubMask(region, *subTestMask);

     cout << endl;
     for (int x=90; x<120; x+=1) {
	  for (int y=295; y<350; y+=5) {
	       cout << x << " " << y << " " << (int)testMask(x, y) << " " << testMask(x, y, planeCR) << endl;
	  }
     }


// --------------------- Test MaskPixelBooleanFunc

     testCrFunc<MaskPixelType> testCrFuncInstance(testMask);

     // Calling init gets the latest plane info from testMask
     testCrFuncInstance.init();

     // Now use testCrFuncInstance
     int count = testMask.countMask(testCrFuncInstance, region);
     cout << count << " pixels had CR set in region" << endl;

     // This should generate a vw exception - dimensions of region and submask must be =

     cout << "This should throw an exception:" << endl;

     region.expand(10);
     testMask.replaceSubMask(region, *subTestMask);


}
