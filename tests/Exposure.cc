// -*- LSST-C++ -*- // fixed format comment for emacs
/**
  * @file
  *
  * @ingroup afw
  *
  * @brief Test code for the LSST Exposure Class.
  *
  * This test code runs some very straightforward tests on the Exposure Class
  * members and (to some extent) its related classes (MaskedImage & Wcs) - it
  * reads in a fits file as a MaskedImage, gets the Wcs and creates an Exposure
  * along with a few other mundane tasks.
  * 
  * Additional tests will eventually include convolution of a Wcs with an
  * Exposure and an attempt to patch up the Wcs. See examples/wcsTests.cc for
  * additional Wcs Class tests.
  *        
  * @author Nicole M. Silvestri, University of Washington
  *
  * Contact nms@astro.washington.edu 
  *
  * Created on: Wed Jun 06 13:15:00 2007
  *
  * @version
  *
  * LSST Legalese here...
  */

#include <iostream>
#include <sstream>
#include <string>

#include "boost/format.hpp"

#include "lsst/daf/base.h"
#include "lsst/pex/exceptions.h"
#include "lsst/afw/image/Exposure.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/pex/logging/Trace.h"
#include "lsst/afw/image/Wcs.h"

namespace image = lsst::afw::image;
namespace logging = lsst::pex::logging;
namespace base = lsst::daf::base;

// FROM POLICY FILE: INPUT AND OUTPUT FILE NAMES FOR EXPOSURES/MASKEDIMAGES
const std::string miOutFile1("miOutFile1"); // output maskedImage
const std::string expMIOutFile1("expMIOutFile1"); // output MI subExposure
const std::string miOutFile2("miOutFile2"); // output maskedImage
const std::string expMIOutFile2("expMIOutFile2"); // output MI subExposure
const std::string expOutFile1("expOutFile1"); // output Exposure

//typedef double pixelType;
typedef float pixelType;
std::string afwdata("");

void doWork() {                         // Block to allow shared_ptrs to go out of scope
    // MASKEDIMAGE CLASS METHOD TESTS: There are 10 tests.  Some of the
    // tests generate images that can be viewed and compared to each other.
    // These have been commented out upon merging to the trunk.
    
    // Read a fits file in as a MaskedImage
    base::PropertySet::Ptr miMetadata(new base::PropertySet());
    int const hdu = 0;
    image::MaskedImage<pixelType> mImageFOO(afwdata + "/small_MI"); // input CFHT MI
    image::MaskedImage<pixelType> mImage(afwdata + "/small_MI", hdu, miMetadata); // input CFHT MI
        
    // Write it back out...
        
    mImage.writeFits(miOutFile1);
       
    // EXPOSURE CLASS METHOD TESTS:

    // (1) Construct a blank Exposure

    image::Exposure<pixelType> blankExpImage;
    image::MaskedImage<pixelType> blankMaskedImage = blankExpImage.getMaskedImage();
    logging::Trace("lsst.afw.Exposure", 5, boost::format("Number of columns, rows in Blank Exposure: %s, %s") %
                   blankMaskedImage.getWidth() % blankMaskedImage.getHeight());
                                  
        
    // (2) Construct an Exposure with only a MaskedImage.

    image::Exposure<pixelType> miExpImage(mImage);
    image::MaskedImage<pixelType> miMaskedImage = miExpImage.getMaskedImage();
    logging::Trace("lsst.afw.Exposure", 5, boost::format("Number of columns, rows in MiExposure: %s, %s")
                   % miMaskedImage.getWidth() % miMaskedImage.getHeight());

    logging::Trace("lsst.afw.Exposure", 5, boost::format("Number of columns, rows in original MaskedImage, 'mImage': %s, %s") % mImage.getWidth() % mImage.getHeight());

    // (3) Construct an Exposure from a MaskedImage and a Wcs.  Need to
    // construct a Wcs first.  The Wcs class takes the Exposure metadata
    // as a DataPropertyPtrT.  getMetadata() returns a pointer to the metadata.

    base::PropertySet::Ptr mData = miMetadata;

    // make sure it can be copied.
    image::Wcs myWcs(mData);  

    image::Wcs wcs2;
    wcs2 = myWcs;
        
    // Now use Exposure class to create an Exposure from a MaskedImage and a
    // Wcs.
       
    image::Exposure<pixelType> miWcsExpImage(mImage, myWcs);
             
    image::Wcs wcsCopy(myWcs); 
      
    //image::Wcs wcsAssigned();
    //wcsAssigned = myWcs; 

    // (4) Construct an Exposure from a given region (col, row) and a Wcs.

    int miWidth = 5;
    int miHeight = 5;
    image::Exposure<pixelType> regWcsExpImage(miWidth, miHeight, myWcs);
       
    // (5) Construct an Exposure from a given region (col, row) with no Wcs.

    int mi2Width = 5;
    int mi2Height = 5;
    image::Exposure<pixelType> regExpImage(mi2Width, mi2Height);
       
    // try to get the Wcs when there isn't one
    try {
        image::Wcs noWcs = *regExpImage.getWcs();
    } catch (lsst::pex::exceptions::Exception &e) {
        logging::Trace("lsst.afw.Exposure", 5, "Caught Exception for getting a null Wcs: %s", e.what());
    }

    // (6) Get a MaskedImage and write it out.

    image::MaskedImage<pixelType> newMiImage =  miWcsExpImage.getMaskedImage();
    newMiImage.writeFits(miOutFile2);
        
    // (7) Get a Wcs. 
         
    image::Wcs newWcs = *miWcsExpImage.getWcs();

    // try to get a Wcs from an image where I have corrupted the Wcs
    // information (removed the CRPIX1/2 header keywords/values.  Lets see
    // what Wcs class does.  It should fail miserably because there is no
    // exception handling for this in the Wcs class.
       
    try {
        int const hdu = 0;          // the HDU to read
        base::PropertySet::Ptr mCorData(new base::PropertySet());
        image::MaskedImage<pixelType> mCorruptImage("tests/data/small_MI_corrupt", hdu,
                                                    mCorData); // CFHT MI with corrupt header
        image::Wcs wcs = image::Wcs(mCorData);
            
        image::Exposure<pixelType> newCorExposure(mCorruptImage, wcs);
       
    } catch (lsst::pex::exceptions::Exception &e) {
        logging::Trace("lsst.afw.Exposure", 1,
                       "Reading Corrupted MaskedImage Failed: %s", e.what());
    }

    // (8) Get a subExposure once the Wcs Class is ready for this to be
    // implemented.

    // Test that this throws the appropriate daf Exception that it should
    // receive from the MaskedImage Class when requested subRegion is
    // outside of the original Exposure's BBox.  Write out the
    // subMaskedImage.
    int orx = 0;             ///< x origin of the subMaskedImage
    int ory = 0;             ///< y origin of the subMaskedImage
    int width = 0;           ///< # of columns in subMaskedImage 
    int height = 0;          ///< # of rows in the subMaskedImage
    width = newMiImage.getWidth();  ///< # of cols in orig MaskedImage
    height = newMiImage.getHeight(); ///< # of rows in orig MaskedImage
        
    // This subRegion should not trigger an exception.  It is completely within
    // the original Exposure's MaskedImage BBox. 
    int subWidth = width - 5;
    int subHeight = height - 5;
    image::BBox subRegion = image::BBox(image::PointI(orx, ory), subWidth, subHeight);
        
    try {
        image::Exposure<pixelType> subExpImage(miWcsExpImage, subRegion);
           
        image::MaskedImage<pixelType> subExpMI = subExpImage.getMaskedImage();
        subExpMI.writeFits(expMIOutFile1);
    } catch (lsst::pex::exceptions::Exception &e) {
        logging::Trace("lsst.afw.Exposure", 5, "Caught Exception for requested subRegion: %s", e.what());
    }
       
    // This subRegion should trigger an exception.  It extends beyond the
    // original Exposure's MaskedImage BBox.  
    int subWidth2 = width + 5;
    int subHeight2 = height + 5;
    const image::BBox subRegion2 = image::BBox(image::PointI(orx, ory), subWidth2, subHeight2); 
     
    try {
        image::Exposure<pixelType> subExpImage2(miWcsExpImage, subRegion2);
            
        image::MaskedImage<pixelType> subExpMI2 = subExpImage2.getMaskedImage();
        subExpMI2.writeFits(expMIOutFile2);
    } catch (lsst::pex::exceptions::Exception &e) {
        logging::Trace("lsst.afw.Exposure", 5, "Caught Exception for requested subRegion2: %s", e.what());
    }
       
    // (9) Check if the Exposure has a Wcs.  Doesn't have to have one.  Bool
    // true/false should be returned.  Warning should be received if Wcs is
    // not present.
        
    // This Exposure should have a Wcs.
    miWcsExpImage.hasWcs(); 
        
    // This Exposure should not have a Wcs.        
    if (!miExpImage.hasWcs()) {
            
    }
    // This Exposure should have a MaskedImage with a size of 0,0 and no Wcs.
    blankExpImage.hasWcs();
        
    // (10) Test readFits/writeFits functionality for Exposure
    // Class...writeFits still needs to be implemented.
    try {
        image::Exposure<pixelType> exposure(afwdata + "/871034p_1_MI"); // input CFHT Exposure
    } catch (lsst::pex::exceptions::Exception &e) {
        logging::Trace("lsst.afw.Exposure", 5, "Reading MaskedImage Failed - caught Exception: %s", e.what());
    }

    try {
        image::Exposure<pixelType> exposure;
        exposure.writeFits(expOutFile1);
    } catch (lsst::pex::exceptions::Exception &e) {
        logging::Trace("lsst.afw.Exposure", 5, "Writing MaskedImage Failed - caught Exception: %s", e.what());
    }

} // close memory (de)allocation block

int main() {
    char *afwdataCStr = getenv("AFWDATA_DIR");
    if (afwdataCStr == 0) {
        std::cerr << "afwdata must be set up" << std::endl;
        return EXIT_FAILURE;
    }
    afwdata = afwdataCStr;
                
    logging::Trace::setDestination(std::cerr);
    logging::Trace::setVerbosity("lsst.afw.Exposure", 4);

    doWork();

    // Checking for memory leaks...
    if (base::Citizen::census(0) != 0) {
        std::cerr << "Leaked memory blocks:" << std::endl;
        base::Citizen::census(std::cerr);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;

} //close main
