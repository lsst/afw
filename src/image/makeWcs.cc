// -*- lsst-c++ -*-




#include "lsst/afw/image/makeWcs.h"

namespace except = lsst::pex::exceptions; 
namespace afwImg = lsst::afw::image;

using namespace std;

typedef lsst::daf::base::PropertySet PropertySet;
typedef lsst::afw::image::Wcs Wcs;
typedef lsst::afw::image::PointD PointD;


/// Use this function to create a Wcs object using a fits header. It examines the header and determines the 
/// most suitable object to return, either a general Wcs object, or a more specific object specialised to a 
/// given coordinate system (e.g TanWcs)
/// 
/// Note that this function violates the standard for Wcs in one minor way. The standard states that none
/// of the CRPIX or CRVAL keywords are required, for the header to be valid, and the appropriate values
/// should be set to 0.0 if the keywords are absent. This is a recipie for painful bugs in analysis, so
/// we violate the standard by insisting that the keywords CRPIX[1,2] and CRVAL[1,2] are present
/// (keywords CRPIX1a etc are also accepted)
afwImg::Wcs::Ptr afwImg::makeWcs(PropertySet::Ptr fitsMetadata){
    std::string ctype1;
    if( fitsMetadata->exists("CTYPE1")) {
        ctype1 = fitsMetadata->getAsString("CTYPE1");
    }
    else {
        throw LSST_EXCEPT(except::RuntimeErrorException, "No CTYPE1 keyword found. Can't determine coordinate system");
    }
    
    if( ctype1.substr(5, 3) == "TAN") {
        return afwImg::Wcs::Ptr(new afwImg::TanWcs(fitsMetadata));
    }
    
    //Default Wcs class
    return afwImg::Wcs::Ptr(new afwImg::Wcs(fitsMetadata));
}
    
