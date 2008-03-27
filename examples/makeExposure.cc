// -*- LSST-C++ -*- // fixed format comment for emacs
/**
  * \file makeExposure.cc
  *
  * \ingroup fw
  *
  * \brief Test code for the LSST Exposure Class.
  *
  * This test code runs some very straightforward tests on the Exposure Class
  * members and (to some extent) its related classes (MaskedImage & WCS) - it
  * reads in a fits file as a MaskedImage, gets the WCS and creates an Exposure
  * along with a few other mundane tasks.
  * 
  * Additional tests will eventually include convolution of a WCS with an
  * Exposure and an attempt to patch up the WCS. See examples/wcsTests.cc for
  * additional WCS Class tests.
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

#include <boost/cstdint.hpp>
#include <boost/format.hpp>
#include <boost/shared_ptr.hpp>
#include <vw/Core.h>
#include <vw/Image.h>
#include <vw/Math/BBox.h>

#include <lsst/daf/data/Citizen.h>
#include <lsst/daf/data/DataProperty.h>
#include <lsst/afw/image/DiskImageResourceFITS.h>
#include <lsst/pex/exceptions.h>
#include <lsst/afw/image/Exposure.h>
#include <lsst/afw/image/Mask.h>
#include <lsst/afw/image/Image.h>
#include <lsst/afw/image/MaskedImage.h>
#include <lsst/pex/utils/Trace.h>
#include <lsst/afw/math/WCS.h>

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

    char *fwDataCStr = getenv("FWDATA_DIR");
    if (fwDataCStr == 0) {
        std::cout << "fwData must be set up" << std::endl;
        exit(1);
    }
    std::string fwData(fwDataCStr);
                
    lsst::pex::utils::Trace::setDestination(std::cout);
    lsst::pex::utils::Trace::setVerbosity("lsst.fw", 4);

    { //memory (de)allocation block

        // MASKEDIMAGE CLASS METHOD TESTS: There are 10 tests.  Some of the
        // tests generate images that can be viewed and compared to each other.
        // These have been commented out upon mergeing to the trunk.
    
        // Read a fits file in as a MaskedImage
        lsst::afw::image::MaskedImage<pixelType, lsst::afw::maskPixelType> mImage;
        const std::string inMIFile(fwData + "/small_MI"); // input CFHT MI
        
        mImage.readFits(inMIFile);
       
        // Write it back out...
        
        mImage.writeFits(miOutFile1);
       
        // EXPOSURE CLASS METHOD TESTS:

        // (1) Construct a blank Exposure

        lsst::afw::Exposure<pixelType, lsst::afw::maskPixelType> blankExpImage;
        lsst::afw::image::MaskedImage<pixelType, lsst::afw::maskPixelType> blankMaskedImage = blankExpImage.getMaskedImage();
        int numCols = blankMaskedImage.getCols();
        int numRows = blankMaskedImage.getRows();
        lsst::pex::utils::Trace("lsst.fw.Exposure", 5, boost::format("Number of columns, rows in Blank Exposure: %s, %s") % numCols % numRows);
        
        // (2) Construct an Exposure with only a MaskedImage.

        lsst::afw::Exposure<pixelType, lsst::afw::maskPixelType> miExpImage(mImage);
        lsst::afw::image::MaskedImage<pixelType, lsst::afw::maskPixelType> miMaskedImage = miExpImage.getMaskedImage();
        int numMiCols = miMaskedImage.getCols();
        int numMiRows = miMaskedImage.getRows();
        int numOrigMiCols = mImage.getCols();
        int numOrigMiRows = mImage.getRows();
        lsst::pex::utils::Trace("lsst.fw.Exposure", 5, boost::format("Number of columns, rows in MiExposure: %s, %s") % numMiCols % numMiRows);
        lsst::pex::utils::Trace("lsst.fw.Exposure", 5, boost::format("Number of columns, rows in original MaskedImage, 'mImage': %s, %s") % numOrigMiCols % numOrigMiRows);

        // (3) Construct an Exposure from a MaskedImage and a WCS.  Need to
        // construct a WCS first.  The WCS class takes the MaskedImage metadata
        // as a DataPropertyPtrT.  The getImage()->getMetaData() member returns a pointer to
        // the metadata.

        lsst::daf::data::DataProperty::PtrType mData = mImage.getImage()->getMetaData();

        // make sure it can be copied.
        lsst::afw::math::WCS myWcs(mData);  

        lsst::afw::math::WCS wcs2;
        wcs2 = myWcs;
        
        // Now use Exposure class to create an Exposure from a MaskedImage and a
        // WCS.
       
       lsst::afw::Exposure<pixelType, lsst::afw::maskPixelType> miWcsExpImage(mImage, myWcs);
             
	lsst::afw::math::WCS wcsCopy(myWcs); 
      
	//lsst::afw::math::WCS wcsAssigned();
	//wcsAssigned = myWcs; 

        // (4) Construct an Exposure from a given region (col, row) and a WCS.

        unsigned miCols = 5;
        unsigned miRows = 5;
        lsst::afw::Exposure<pixelType, lsst::afw::maskPixelType> regWcsExpImage(miCols, miRows, myWcs);
       
        // (5) Construct an Exposure from a given region (col, row) with no WCS.

        unsigned mi2Cols = 5;
        unsigned mi2Rows = 5;
        lsst::afw::Exposure<pixelType, lsst::afw::maskPixelType> regExpImage(mi2Cols, mi2Rows);
       
        // try to get the WCS when there isn't one
        try {
        lsst::afw::math::WCS noWcs = regExpImage.getWcs();
        } catch (lsst::pex::exceptions::NotFound e) {
            lsst::pex::utils::Trace("lsst.fw.Exposure", 5, "Caught lsst::pex NotFound Exception for getting a null WCS");
        }

        // (6) Get a MaskedImage and write it out.

        lsst::afw::image::MaskedImage<pixelType, lsst::afw::maskPixelType> newMiImage =  miWcsExpImage.getMaskedImage();
        newMiImage.writeFits(miOutFile2);
        
        // (7) Get a WCS. 
         
       lsst::afw::math::WCS newWcs = miWcsExpImage.getWcs();

        // try to get a WCS from an image where I have corrupted the WCS
        // information (removed the CRPIX1/2 header keywords/values.  Lets see
        // what WCS class does.  It should fail miserably because there is no
        // exception handling for this in the WCS class.
        
        lsst::afw::image::MaskedImage<pixelType, lsst::afw::maskPixelType> mCorruptImage;
        const std::string inCorrMIFile("tests/data/small_MI_corrupt"); // input CFHT MI with corrupt header
       
        try {
        mCorruptImage.readFits(inCorrMIFile);
        lsst::daf::data::DataProperty::PtrType  mCorData = mCorruptImage.getImage()->getMetaData();
        lsst::afw::math::WCS wcs = lsst::afw::math::WCS(mCorData);
       
        lsst::afw::Exposure<pixelType, lsst::afw::maskPixelType> newCorExposure(mCorruptImage, wcs);
       
        } catch (lsst::pex::exceptions::NotFound error) {
            lsst::pex::utils::Trace("lsst.fw.Exposure", 1, "Reading Corrupted MaskedImage Failed - caught lsst::pex NotFound Exception.");
        }

        // (8) Get a subExposure once the WCS Class is ready for this to be
        // implemented.

        // Test that this throws the appropriate mwi Exception that it should
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
            lsst::afw::Exposure<pixelType, lsst::afw::maskPixelType> subExpImage =  miWcsExpImage.getSubExposure(subRegion);
           
            lsst::afw::image::MaskedImage<pixelType, lsst::afw::maskPixelType> subExpMI = subExpImage.getMaskedImage();
            subExpMI.writeFits(expMIOutFile1);
        } catch (lsst::pex::exceptions::InvalidParameter ex) {
            lsst::pex::utils::Trace("lsst.fw.Exposure", 5, "Caught InvalidParameter Exception for requested subRegion");
        }
       
        // This subRegion should trigger an exception.  It extends beyond the
        // original Exposure's MaskedImage BBox.  
        vw::int32 subWidth2 = width + 5;
        vw::int32 subHeight2 = height + 5;
        const vw::BBox2i &subRegion2 = BBox2i(orx, ory, subWidth2, subHeight2); 
     
        try {
            lsst::afw::Exposure<pixelType, lsst::afw::maskPixelType> subExpImage2 =  miWcsExpImage.getSubExposure(subRegion2); 
            
            lsst::afw::image::MaskedImage<pixelType, lsst::afw::maskPixelType> subExpMI2 = subExpImage2.getMaskedImage();
            subExpMI2.writeFits(expMIOutFile2);
        } catch (lsst::pex::exceptions::InvalidParameter err) {
            lsst::pex::utils::Trace("lsst.fw.Exposure", 5, "Caught InvalidParameter Exception for requested subRegion2");
        }
       
        // (9) Check if the Exposure has a WCS.  Doesn't have to have one.  Bool
        // true/false should be returned.  Warning should be received if WCS is
        // not present.
        
        // This Exposure should have a WCS.
        miWcsExpImage.hasWcs(); 
        
        // This Exposure should not have a WCS.        
        if (!miExpImage.hasWcs()) {
            
        }
        // This Exposure should have a MaskedImage with a size of 0,0 and no WCS.
        blankExpImage.hasWcs();
        
        // (10) Test readFits/writeFits functionality for Exposure
        // Class...writeFits still needs to be implemented.
        lsst::afw::Exposure<pixelType, lsst::afw::maskPixelType> exposure;
        
        const std::string inExFile(fwData + "/871034p_1_MI"); // input CFHT Exposure
        try {
        exposure.readFits(fwData + "/871034p_1_MI");        
        } catch (lsst::pex::exceptions::NotFound error) {
           lsst::pex::utils::Trace("lsst.fw.Exposure", 5, "Reading MaskedImage Failed - caught lsst::pex NotFound Exception.");
        }

        try {
        exposure.writeFits(expOutFile1);
        } catch (lsst::pex::exceptions::InvalidParameter error) {
            lsst::pex::utils::Trace("lsst.fw.Exposure", 5, "Writing MaskedImage Failed - caught lsst::pex InvalidParameter Exception.");
        }

    } // close memory (de)allocation block

    // Checking for memory leaks...
    if (lsst::daf::data::Citizen::census(0) != 0) {
        std::cerr << "Leaked memory blocks:" << std::endl;
        lsst::daf::data::Citizen::census(std::cerr);
    }

} //close main
