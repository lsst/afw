// -*- LSST-C++ -*- // fixed format comment for emacs
/**
  * \file
  *
  * \ingroup afw
  *
  * \brief Test code for the LSST Exposure Class.
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
  * \author Nicole M. Silvestri, University of Washington
  *
  * Contact nms@astro.washington.edu 
  *
  * Created on: Wed Jun 06 13:15:00 2007
  *
  * \version
  *
  * LSST Legalese here...
  */

#include <iostream>
#include <sstream>
#include <string>

#include <boost/format.hpp>
#include "vw/Core.h"
#include "vw/Math/BBox.h"

#include "lsst/daf/base.h"
#include "lsst/pex/exceptions.h"
#include "lsst/afw/image/Exposure.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/pex/logging/Trace.h"
#include "lsst/afw/image/Wcs.h"

// FROM POLICY FILE: INPUT AND OUTPUT FILE NAMES FOR EXPOSURES/MASKEDIMAGES
const std::string miOutFile1("miOutFile1"); // output maskedImage
const std::string expMIOutFile1("expMIOutFile1"); // output MI subExposure
const std::string miOutFile2("miOutFile2"); // output maskedImage
const std::string expMIOutFile2("expMIOutFile2"); // output MI subExposure
const std::string expOutFile1("expOutFile1"); // output Exposure

int main() {
    //typedef double pixelType;
    typedef float pixelType;
    typedef float viewType;

    char *afwdataCStr = getenv("AFWDATA_DIR");
    if (afwdataCStr == 0) {
        std::cout << "afwdata must be set up" << std::endl;
        exit(1);
    }
    std::string afwdata(afwdataCStr);
                
    lsst::pex::logging::Trace::setDestination(std::cout);
    lsst::pex::logging::Trace::setVerbosity("lsst.afw", 4);

    { //memory (de)allocation block

        // MASKEDIMAGE CLASS METHOD TESTS: There are 10 tests.  Some of the
        // tests generate images that can be viewed and compared to each other.
        // These have been commented out upon mergeing to the trunk.
    
        // Read a fits file in as a MaskedImage
        lsst::afw::image::MaskedImage<pixelType, lsst::afw::image::maskPixelType> mImage;
        const std::string inMIFile(afwdata + "/small_MI"); // input CFHT MI
        
        mImage.readFits(inMIFile);
       
        // Write it back out...
        
        mImage.writeFits(miOutFile1);
       
        // EXPOSURE CLASS METHOD TESTS:

        // (1) Construct a blank Exposure

        lsst::afw::image::Exposure<pixelType, lsst::afw::image::maskPixelType> blankExpImage;
        lsst::afw::image::MaskedImage<pixelType, lsst::afw::image::maskPixelType> blankMaskedImage = blankExpImage.getMaskedImage();
        int numCols = blankMaskedImage.getCols();
        int numRows = blankMaskedImage.getRows();
        lsst::pex::logging::Trace("lsst.afw.Exposure", 5, boost::format("Number of columns, rows in Blank Exposure: %s, %s") % numCols % numRows);
        
        // (2) Construct an Exposure with only a MaskedImage.

        lsst::afw::image::Exposure<pixelType, lsst::afw::image::maskPixelType> miExpImage(mImage);
        lsst::afw::image::MaskedImage<pixelType, lsst::afw::image::maskPixelType> miMaskedImage = miExpImage.getMaskedImage();
        int numMiCols = miMaskedImage.getCols();
        int numMiRows = miMaskedImage.getRows();
        int numOrigMiCols = mImage.getCols();
        int numOrigMiRows = mImage.getRows();
        lsst::pex::logging::Trace("lsst.afw.Exposure", 5, boost::format("Number of columns, rows in MiExposure: %s, %s") % numMiCols % numMiRows);
        lsst::pex::logging::Trace("lsst.afw.Exposure", 5, boost::format("Number of columns, rows in original MaskedImage, 'mImage': %s, %s") % numOrigMiCols % numOrigMiRows);

        // (3) Construct an Exposure from a MaskedImage and a Wcs.  Need to
        // construct a Wcs first.  The Wcs class takes the MaskedImage metadata
        // as a DataPropertyPtrT.  The getImage()->getMetaData() member returns a pointer to
        // the metadata.

        lsst::daf::base::DataProperty::PtrType mData = mImage.getImage()->getMetaData();

        // make sure it can be copied.
        lsst::afw::image::Wcs myWcs(mData);  

        lsst::afw::image::Wcs wcs2;
        wcs2 = myWcs;
        
        // Now use Exposure class to create an Exposure from a MaskedImage and a
        // Wcs.
       
       lsst::afw::image::Exposure<pixelType, lsst::afw::image::maskPixelType> miWcsExpImage(mImage, myWcs);
             
	lsst::afw::image::Wcs wcsCopy(myWcs); 
      
	//lsst::afw::image::Wcs wcsAssigned();
	//wcsAssigned = myWcs; 

        // (4) Construct an Exposure from a given region (col, row) and a Wcs.

        unsigned miCols = 5;
        unsigned miRows = 5;
        lsst::afw::image::Exposure<pixelType, lsst::afw::image::maskPixelType> regWcsExpImage(miCols, miRows, myWcs);
       
        // (5) Construct an Exposure from a given region (col, row) with no Wcs.

        unsigned mi2Cols = 5;
        unsigned mi2Rows = 5;
        lsst::afw::image::Exposure<pixelType, lsst::afw::image::maskPixelType> regExpImage(mi2Cols, mi2Rows);
       
        // try to get the Wcs when there isn't one
        try {
        lsst::afw::image::Wcs noWcs = regExpImage.getWcs();
        } catch (lsst::pex::exceptions::NotFound e) {
            lsst::pex::logging::Trace("lsst.afw.Exposure", 5, "Caught lsst::pex NotFound Exception for getting a null Wcs");
        }

        // (6) Get a MaskedImage and write it out.

        lsst::afw::image::MaskedImage<pixelType, lsst::afw::image::maskPixelType> newMiImage =  miWcsExpImage.getMaskedImage();
        newMiImage.writeFits(miOutFile2);
        
        // (7) Get a Wcs. 
         
       lsst::afw::image::Wcs newWcs = miWcsExpImage.getWcs();

        // try to get a Wcs from an image where I have corrupted the Wcs
        // information (removed the CRPIX1/2 header keywords/values.  Lets see
        // what Wcs class does.  It should fail miserably because there is no
        // exception handling for this in the Wcs class.
        
        lsst::afw::image::MaskedImage<pixelType, lsst::afw::image::maskPixelType> mCorruptImage;
        const std::string inCorrMIFile("tests/data/small_MI_corrupt"); // input CFHT MI with corrupt header
       
        try {
        mCorruptImage.readFits(inCorrMIFile);
        lsst::daf::base::DataProperty::PtrType  mCorData = mCorruptImage.getImage()->getMetaData();
        lsst::afw::image::Wcs wcs = lsst::afw::image::Wcs(mCorData);
       
        lsst::afw::image::Exposure<pixelType, lsst::afw::image::maskPixelType> newCorExposure(mCorruptImage, wcs);
       
        } catch (lsst::pex::exceptions::NotFound error) {
            lsst::pex::logging::Trace("lsst.afw.Exposure", 1, "Reading Corrupted MaskedImage Failed - caught lsst::pex NotFound Exception.");
        }

        // (8) Get a subExposure once the Wcs Class is ready for this to be
        // implemented.

        // Test that this throws the appropriate daf Exception that it should
        // receive from the MaskedImage Class when requested subRegion is
        // outside of the original Exposure's BBox.  Write out the
        // subMaskedImage.
        vw::int32 orx = 0;             ///< x origin of the subMaskedImage
        vw::int32 ory = 0;             ///< y origin of the subMaskedImage
        vw::int32 width = 0;           ///< # of columns in subMaskedImage 
        vw::int32 height = 0;          ///< # of rows in the subMaskedImage
        width = newMiImage.getCols();  ///< # of cols in orig MaskedImage
        height = newMiImage.getRows(); ///< # of rows in orig MaskedImage
        
        // This subRegion should not trigger an exception.  It is completely within
        // the original Exposure's MaskedImage BBox. 
        vw::int32 subWidth = width - 5;
        vw::int32 subHeight = height - 5;
        const vw::BBox2i &subRegion = vw::BBox2i(orx, ory, subWidth, subHeight);
        
        try {
            lsst::afw::image::Exposure<pixelType, lsst::afw::image::maskPixelType> subExpImage =  miWcsExpImage.getSubExposure(subRegion);
           
            lsst::afw::image::MaskedImage<pixelType, lsst::afw::image::maskPixelType> subExpMI = subExpImage.getMaskedImage();
            subExpMI.writeFits(expMIOutFile1);
        } catch (lsst::pex::exceptions::InvalidParameter ex) {
            lsst::pex::logging::Trace("lsst.afw.Exposure", 5, "Caught InvalidParameter Exception for requested subRegion");
        }
       
        // This subRegion should trigger an exception.  It extends beyond the
        // original Exposure's MaskedImage BBox.  
        vw::int32 subWidth2 = width + 5;
        vw::int32 subHeight2 = height + 5;
        const vw::BBox2i &subRegion2 = vw::BBox2i(orx, ory, subWidth2, subHeight2); 
     
        try {
            lsst::afw::image::Exposure<pixelType, lsst::afw::image::maskPixelType> subExpImage2 =  miWcsExpImage.getSubExposure(subRegion2); 
            
            lsst::afw::image::MaskedImage<pixelType, lsst::afw::image::maskPixelType> subExpMI2 = subExpImage2.getMaskedImage();
            subExpMI2.writeFits(expMIOutFile2);
        } catch (lsst::pex::exceptions::InvalidParameter err) {
            lsst::pex::logging::Trace("lsst.afw.Exposure", 5, "Caught InvalidParameter Exception for requested subRegion2");
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
        lsst::afw::image::Exposure<pixelType, lsst::afw::image::maskPixelType> exposure;
        
        const std::string inExFile(afwdata + "/871034p_1_MI"); // input CFHT Exposure
        try {
        exposure.readFits(afwdata + "/871034p_1_MI");        
        } catch (lsst::pex::exceptions::NotFound error) {
           lsst::pex::logging::Trace("lsst.afw.Exposure", 5, "Reading MaskedImage Failed - caught lsst::pex NotFound Exception.");
        }

        try {
        exposure.writeFits(expOutFile1);
        } catch (lsst::pex::exceptions::InvalidParameter error) {
            lsst::pex::logging::Trace("lsst.afw.Exposure", 5, "Writing MaskedImage Failed - caught lsst::pex InvalidParameter Exception.");
        }

    } // close memory (de)allocation block

    // Checking for memory leaks...
    if (lsst::daf::base::Citizen::census(0) != 0) {
        std::cerr << "Leaked memory blocks:" << std::endl;
        lsst::daf::base::Citizen::census(std::cerr);
    }

} //close main
